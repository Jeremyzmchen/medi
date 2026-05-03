# 预诊流程完整走读

基于测试场景：用户输入"我孩子发烧了"，8轮问诊后生成分诊报告。

---

## 节点拓扑

```
[START] → intake → intake_review → (END 等待用户输入)
                                 ↘ clinical → output → [END]
                    ↑___back-loop from clinical (最多1次)___↑
```

---

## Step 1：Runner.handle() — Safety Gate

**文件**：`runner.py:50-65`

```python
urgency = evaluate_urgency_by_rules(user_input)
if urgency and urgency.level == UrgencyLevel.EMERGENCY:
    # 直接 emit ESCALATION + RESULT，不进图
    self.reset_graph_state()
    return
```

- 规则层关键词匹配（"胸痛+晕倒"、"大量出血"等），命中则绕过图直接返回急救响应
- "我孩子发烧了"不触发，通过安全门，继续往下

---

## Step 2：第一轮 — 构建 initial_state，invoke 图

**文件**：`runner.py:72-100`

`_is_first_turn=True`，构建完整 `TriageGraphState`：

```python
initial_state = TriageGraphState(
    messages=[{"role": "user", "content": "我孩子发烧了"}],
    intake_protocol_id="generic_opqrst",  # 占位，等 IntakeNode 识别
    intake_facts=[],
    requested_slots=[],                   # operator.add reducer，累积已问槽
    intake_complete=False,
    next_node="intake",
    graph_iteration=0,
    ...
)
```

调用 `graph.ainvoke(initial_state, config)`，进入 `intake` 节点。

---

## Step 3：IntakeNode — 协议识别 + 事实提取

**文件**：`graph/nodes/intake_node.py`

**日志**：`[intake] round=1 protocol=fever overlays=['pediatric']`

### 3a. 协议识别

```python
fixed_protocol_id = _locked_protocol_id(state)
# state["intake_protocol_id"] == "generic_opqrst"（占位值）→ 返回 None
intake_plan = resolve_intake_plan(messages, health_profile, fixed_protocol_id=None)
```

`resolve_intake_plan` 只取 `role=="user"` 的消息做关键词匹配（防止护士问题污染匹配）：

- "发烧" → 命中 `fever` 协议
- "孩子" → 激活 `pediatric` overlay

输出：`intake_plan.protocol_id = "fever"`, `overlay_ids = ["pediatric"]`

### 3b. LLM 事实提取

用 `fast_chain` 做 JSON 事实抽取（`temperature=0`，JSON mode）：

```json
[{"slot": "hpi.chief_complaint", "status": "present", "value": "发烧", "confidence": 0.9}]
```

同时跑 `extract_deterministic_facts`（规则层，提取年龄暗示等）。

`FactStore.merge_fact()` 合并：新槽直接写入，旧槽按 `should_replace()` 判断是否覆盖。

### 3c. 写回 state

```python
return {
    "intake_protocol_id": "fever",     # 从此轮起锁定协议
    "intake_overlays": ["pediatric"],
    "intake_facts": store.to_state(),
    "next_node": "intake_review",
    "intake_complete": False,
    ...
}
```

---

## Step 4：路由到 IntakeReviewNode

**文件**：`graph/builder.py`，`_route_from_intake`

```python
def _route_from_intake(state) -> str:
    if state.get("next_node") == "intake_review":
        return "review"
    return "end"
```

IntakeNode 永远输出 `next_node="intake_review"`，所以永远路由到 review。

---

## Step 5：IntakeReviewNode — 质量打分，第一轮

**文件**：`graph/nodes/intake_review_node.py`

**日志**：`[intake_review] score=12 ready=False next_slot=specific.max_temperature`

`review_intake_quality()` 打分：

| 槽位 | 状态 | 得分 |
|------|------|------|
| chief_complaint | complete | +12 |
| 其余所有 | missing | 0 |
| **总分** | | **12** |

fever 协议阈值 74，12 < 74，`intake_complete=False`。

`_pick_next_slot()` 从未收集槽中按优先级排序，选出 `specific.max_temperature`，查 `REVIEW_SLOT_QUESTIONS` 得问题文本。

图路由到 `END`，emit `FOLLOW_UP` 事件，CLI 打印：**"最高体温到多少度？大概是什么时候测到的？"**

---

## Step 6：第二轮输入 — Checkpointer 恢复

**文件**：`runner.py:101-104`

```python
# _is_first_turn=False，只传增量消息
input_data = {"messages": [{"role": "user", "content": "昨天晚上开始的，最高烧到39度"}]}
```

`MemorySaver` 以 `session_id` 为 `thread_id`，自动恢复上一轮完整 state。`messages` 字段通过 `operator.add` reducer 追加（不覆盖）。

图从 `intake` 节点重新开始，state 里已有：`intake_protocol_id="fever"`（已锁定）、`intake_facts`（含 chief_complaint）。

---

## Step 7：IntakeNode 第二轮 — 协议锁定生效

**日志**：`[intake] round=2 protocol=fever overlays=['pediatric']`

```python
fixed_protocol_id = _locked_protocol_id(state)
# intake_protocol_id = "fever"（非 generic）→ 返回 "fever"
resolve_intake_plan(..., fixed_protocol_id="fever")  # 直接用，不重新匹配
```

LLM 提取新事实并合并到 FactStore：

```json
[
  {"slot": "hpi.onset", "status": "present", "value": "昨天晚上"},
  {"slot": "specific.max_temperature", "status": "present", "value": "39度"}
]
```

**日志**：`[intake_review] score=41 ready=False next_slot=specific.associated_fever_symptoms`

onset(+14) + max_temperature 相关分数，从 12 升到 41，下一槽：`specific.associated_fever_symptoms`。

---

## Step 8：第三~五轮 — 逐槽采集

每轮结构相同：`intake → intake_review → END`，每次 `ainvoke()` 都走完整图，Checkpointer 负责跨轮持久化。

| 轮次 | 用户输入 | 新增槽 | score | next_slot |
|------|---------|--------|-------|-----------|
| 3 | 有呕吐 | `specific.associated_fever_symptoms` | 57 | `specific.age` |
| 4 | 5岁 | `specific.age` | 60 | `specific.mental_status` |
| 5 | 没啥精神 | `specific.mental_status` | 63 | `specific.intake_urination` |

---

## Step 9：第六轮 — 重复问题 bug

**日志**：
```
轮6: score=63 next_slot=specific.intake_urination
轮7: score=63 next_slot=specific.intake_urination  ← 同一个槽
```

**原因**：

用户回答"不爱吃饭，其他都挺正常的"。`specific.intake_urination` 要求进食/饮水/尿量三项，LLM 认为只有进食，把该槽抽成 `status: "partial"`。

```python
def is_collected(self, slot: str) -> bool:
    return fact.status not in {"", "missing", "partial"}
    # partial 不算已收集 → ReviewNode 认为仍是 missing → 继续追问
```

**第七轮自动解脱**：

"我刚回答了啊"无有效事实，`specific.intake_urination` 仍 missing。但 `requested_slots`（operator.add 累积）现在记录了该槽被问过两次，`_pick_next_slot()` 判断已多次请求，跳过，选 `safety.current_medications`。

---

## Step 10：第八轮 — 达标，进入 ClinicalNode

用户输入："吃了布洛芬，没有过敏"

LLM 提取：
```json
[
  {"slot": "safety.current_medications", "status": "present", "value": "布洛芬"},
  {"slot": "safety.allergies", "status": "absent"}
]
```

**日志**：`[intake_review] score=82 ready=True next_slot=None`

medications(+7) + allergies(+8)，从 63 跳到 82，超过阈值 74。

`intake_complete=True`，路由到 `clinical`：

```python
def _route_from_review(state) -> str:
    if state.get("intake_complete"):
        return "clinical"
    return "end"
```

---

## Step 11：ClinicalNode

**文件**：`graph/nodes/clinical_node.py`

### 11a. 科室路由

`router.route(query_text, top_k=3)` 向量检索，输出：儿科（76%）。

### 11b. LLM 紧急程度评估

`evaluate_urgency_by_llm()`：发烧39度 + 呕吐 + 精神差 + 5岁 → `watchful`（较急）。

### 11c. 风险因子评估

`evaluate_risk_factors(symptom_data, health_profile=None)`：无 health_profile，`elevated_urgency=False`，紧急程度维持 watchful。

### 11d. 鉴别诊断

`_generate_differential()` 调用 smart_chain，JSON 输出：

| 诊断 | likelihood |
|------|-----------|
| 病毒性胃肠炎 | high |
| 细菌性脑膜炎 | medium |
| 食物中毒 | medium |
| 上呼吸道感染 | low |

### 11e. back-loop 判断

```python
has_high_likelihood = True   # 病毒性胃肠炎 == "high"
clinical_missing_slots = _missing_for_diagnosis(...)

# 条件：not has_high_likelihood AND clinical_missing_slots AND graph_iteration < 2
# → False，不回追问
next_node = "output"
```

---

## Step 12：OutputNode — 双输出单次 LLM 调用

**文件**：`graph/nodes/output_node.py`

一次 `smart_chain` 调用，`response_format={"type": "json_object"}`，同时生成：

```json
{
  "patient_output": {
    "recommended_department": "儿科",
    "urgency": "较急",
    "advice": "请尽快带孩子去儿科就诊...",
    "red_flags": ["持续高热不退", "精神反应异常", "尿量明显减少"]
  },
  "doctor_hpi": {
    "chief_complaint": "孩子发烧并伴有呕吐",
    "hpi_narrative": "5岁孩子昨晚开始发热，最高体温39度...",
    "onset": "昨晚",
    "differential_diagnoses": [...]
  }
}
```

**设计原因**：两个输出同一次调用，保证患者版紧急程度和医生版 HPI 叙述数据一致，不会互相矛盾。

---

## Step 13：Runner._process_result() — 会话结束

**文件**：`runner.py:165-185`

```python
if result.get("patient_output") is not None:
    self._ctx.transition(DialogueState.INIT)
    self.reset_graph_state()
    # 重置 MemorySaver、_is_first_turn=True、清空 _cached_symptom_data
```

CLI 收到 `RESULT` 事件，打印分诊结果和医生 HPI 报告，会话结束。

---

## 完整流程统计

| 指标 | 值 |
|------|---|
| intake → intake_review → END 循环次数 | 8 |
| ClinicalNode 执行次数 | 1 |
| back-loop 触发 | 否（有 high likelihood 诊断） |
| 协议锁定生效轮次 | 第2轮起 |
| 重复问题轮次 | 第6~7轮（intake_urination partial） |
| 最终得分 | 82 / 阈值 74 |

---

## 已知问题

### Bug 1：`specific.intake_urination` 重复追问

**现象**：用户给出部分回答，LLM 抽成 `partial`，同一问题连问两轮。  
**根因**：`is_collected()` 把 `partial` 视为未收集。  
**可选修复**：`partial` 且已在 `requested_slots` 出现过时，降级为"接受现有值"。

### Bug 2：`specific.intake_urination` 最终仍为 missing

**现象**：最终 missing 列表仍含该槽，但系统跳过后正常完成。  
**影响**：得分未计入该槽分数（约 +5），整体得分偏低但已超阈值，不影响本次结果。  
**边界情况**：若分数恰好卡在阈值附近，该 bug 可能导致不必要的额外追问轮次。
