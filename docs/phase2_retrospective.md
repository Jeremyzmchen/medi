# Medi Phase 2 — 开发回顾文档

## 一、Phase 2 目标与完成情况

### 计划目标

| 任务 | 状态 |
|------|------|
| HealthProfile SQLite 持久化 | 完成 |
| OrchestratorAgent 意图识别与路由 | 完成 |
| 对话历史（ctx.messages）多轮上下文 | 完成 |
| OPQRST 症状收集标准 | 完成 |
| LLM 追问生成（替代硬编码模板） | 完成 |
| LLM 部位推断兜底（_enrich_region） | 完成 |
| new_symptom 意图 + 历史重置 | 完成 |
| EpisodicMemory 分诊记录持久化 | 完成 |
| MedicationAgent 用药咨询 | 完成（GPT-4o 直接回答，后续接药物数据库 API） |
| decompose_input 混合意图拆分 | 完成 |

**未完成项（顺延 Phase 3）**：
- MemoryDistiller（设计有缺陷，等历史数据积累后重新设计）
- MedicationAgent 接药物数据库 API（NMPA/RxNorm/OpenFDA）+ 多模态输入（拍药盒）
- 全量向量索引（17.7 万条，当前仍为 2000 条测试量）

---

## 二、系统架构

### 2.1 整体分层（Phase 2 新增部分用 ★ 标注）

```
用户层
  CLI（typer + rich）
    ★ 首次使用引导填写健康档案
    ★ 会话开始展示最近就诊记录

Agent 层
  ★ OrchestratorAgent（输入分解 + 意图分类 + 路由）
    ├── decompose_input()：检测混合输入，拆分为独立子问题
    ├── symptom     → TriageAgent
    ├── new_symptom → 清空历史 → TriageAgent
    ├── medication  → MedicationAgent
    ├── followup    → OrchestratorAgent 直接回答
    └── out_of_scope → 边界提示

  ★ MedicationAgent
    ├── 处理药物查询、副作用、药物冲突三类问题
    ├── 注入 HealthProfile 硬约束（过敏史 + 当前用药）
    └── GPT-4o 直接回答（后续替换为药物数据库 API）

  TriageAgent（TAOR 主流程）
    ├── Safety 层（UrgencyEvaluator 规则扫描）
    ├── Think
    │     ├── NER 提取 OPQRST 实体
    │     ├── 关键词规则补充 O/Q/S/P 字段
    │     └── ★ _enrich_region()：LLM 推断缺失部位
    ├── Act
    │     ├── LLM 生成 OPQRST 追问（替代硬编码模板）
    │     └── ★★ LLM Tool Use → ToolRuntime → DepartmentRouter（Phase 3 升级）
    ├── Observe（透传）
    └── Respond
          ├── ★ 注入 OPQRST 症状摘要
          ├── ★ 注入历史就诊记录（软参考）
          └── GPT-4o 生成建议

基础设施层
  ★ UnifiedContext    — 新增 messages 对话历史
  AsyncStreamBus      — 异步事件总线
  ★★ ToolRuntime      — Phase 1 已实现，Phase 3 正式接入（注册 search_symptom_kb）

记忆层
  ★ HealthProfile     — SQLite 用户静态档案（硬约束）
  ★ EpisodicMemory    — SQLite 分诊记录（软参考）

知识层
  ChromaDB            — 症状-科室向量索引
  bge-large-zh-v1.5   — 症状文本 embedding
  bert-NER            — 症状实体识别
```

### 2.2 对话状态机（Phase 2 无变更，附完整流程）

```
INIT
  │
  ├── OrchestratorAgent 意图分类
  │     ├── out_of_scope  → 边界回复 → INIT
  │     ├── followup      → 带历史直接回答 → INIT
  │     ├── new_symptom   → 清空 ctx.messages + 重置 SymptomInfo → 走下方正常流程
  │     └── symptom / medication / new_symptom（重置后）→ 走下方正常流程
  │
  ├── Safety 层：evaluate_urgency_by_rules(text)
  │     └── EMERGENCY → ESCALATING → 输出急救提示 → 重置 → INIT
  │
  └── 正常流程（TAOR）
        │
      COLLECTING（Think）
        ├── NER + 关键词 + LLM 兜底提取 OPQRST
        ├── is_sufficient()=False & can_follow_up() → LLM 生成追问 → 等待输入
        └── is_sufficient()=True or 追问次数耗尽
              │
            SUFFICIENT
              │
            SEARCHING（Act：LLM Tool Use → ToolRuntime → 向量检索）
              │
            RESPONDING（Respond：LLM 生成建议）
              │
            INIT（重置，保存 EpisodicMemory）
```

### 2.3 意图分类流程（新增）

```
用户输入
    │
    ▼
OrchestratorAgent.decompose_input(user_input)
    ├── 单一问题 → [原文]
    └── 混合问题 → ["子问题1", "子问题2", ...]

对每个子问题：
    │
    ▼
OrchestratorAgent.classify_intent(question, symptom_summary)
    │
    ├── 注入信号
    │     ├── dialogue_state（COLLECTING / INIT）+ 语义提示
    │     ├── symptom_summary（当前 OPQRST 摘要）
    │     └── 近 6 条对话历史
    │
    ├── gpt-4o-mini 分类（max_tokens=15，temperature=0）
    │
    └── 返回 Intent 枚举，路由到对应 Agent
          symptom / new_symptom → TriageAgent
          medication            → MedicationAgent
          followup              → OrchestratorAgent.handle_followup()
          out_of_scope          → 边界提示
```

**保守策略**：分类失败时默认返回 `Intent.SYMPTOM`，避免误拒用户的就医需求。

**new_symptom vs symptom 判断依据**：
- `dialogue_state=COLLECTING`（分诊进行中）→ 新输入倾向为 `symptom`（补充当前主诉）
- `dialogue_state=INIT`（分诊已完成）→ 新输入倾向为 `new_symptom`（全新主诉）

**混合输入拆分示例**：
```
输入："我昨天吃了海鲜，身上过敏了，而且还有点感冒，氯雷他定和999感冒灵可以一起吃吗？"
拆分：["我昨天吃了海鲜，身上过敏了，而且还有点感冒", "氯雷他定和999感冒灵可以一起吃吗"]
路由：symptom → TriageAgent（皮肤科）, medication → MedicationAgent（药物冲突）
执行：顺序执行（保证回答有逻辑顺序，分诊建议先于用药建议）
```

---

## 三、核心模块详解

### 3.1 UnifiedContext — 新增 messages 对话历史

Phase 2 在 `UnifiedContext` 中增加了 `messages: list[dict]` 字段，记录完整对话历史。

```python
@dataclass
class UnifiedContext:
    # ... Phase 1 字段 ...
    messages: list[dict] = field(default_factory=list)  # ★ 新增

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
```

**作用**：
- `TriageAgent._respond()` 把 `ctx.messages` 完整传给 GPT-4o，让建议结合多轮上下文
- `OrchestratorAgent.handle_followup()` 带完整历史回答追问
- `OrchestratorAgent.classify_intent()` 带近 6 条历史做意图判断

**生命周期**：
- `new_symptom` 触发时 `ctx.messages.clear()`（避免多主诉串扰）
- ESCALATION 不清空（保留急救上下文供 followup）

### 3.2 OrchestratorAgent — 意图识别与路由

```
classify_intent()
    ↓
Intent.SYMPTOM / NEW_SYMPTOM  →  TriageAgent.handle()
Intent.FOLLOWUP               →  handle_followup()（带 ctx.messages 直接回答）
Intent.OUT_OF_SCOPE           →  handle_out_of_scope()（固定边界回复）
Intent.MEDICATION             →  TriageAgent.handle()（Phase 3 换 MedicationAgent）
```

**followup 处理**：不走 TriageAgent，OrchestratorAgent 直接用 gpt-4o-mini + 完整对话历史回答，max_tokens=300。

**out_of_scope 回复**（硬编码）：
```
我是分诊助手，只能帮您判断应就诊的科室和紧急程度。
您的问题超出了我的服务范围，建议通过搜索引擎或医院官网获取相关信息。
```

### 3.3 OPQRST 症状收集标准

Phase 1 仅收集 `location`（部位）和 `duration`（时间），Phase 2 扩展为 OPQRST 六字段：

| 字段 | 含义 | 提取方式 |
|------|------|----------|
| O - Onset | 发作时间与诱因 | 关键词（吃、跑、受凉、摔…） |
| P - Provocation | 加重/缓解因素 | 关键词（加重、缓解、饭后…） |
| Q - Quality | 症状性质 | 关键词（刺痛、钝痛、烧灼…） |
| R - Region | 部位与放射痛 | NER 模型 → LLM 兜底推断 |
| S - Severity | 严重程度 0-10 | 正则（`[0-9]\s*分`） |
| T - Time | 时间特征 | 关键词（天、周、昨、最近…） |

**充分标准**（`is_sufficient()`）：
- Phase 1：`location AND duration`（过于严格，"腹泻"这类症状词无法触发 location）
- Phase 2：`R AND (T OR O)`（R 必须有，T 或 O 至少一个）

**R 字段两级提取策略**：
1. NER 模型优先（识别解剖部位实体）
2. NER 未提取到 → `_enrich_region()` 调用 gpt-4o-mini 推断（"腹泻"→"腹部"，"头晕"→"头部"）

**LLM 追问生成**（`build_follow_up_question()`）：
- Phase 1：硬编码模板（"请问您的不适主要在哪个部位？"）
- Phase 2：gpt-4o-mini 根据已知信息 + 缺失字段描述生成自然追问
- 每次最多追问 2 个字段，temperature=0.3（保留一定自然感）

### 3.4 HealthProfile — SQLite 持久化

```
data/medi.db
  └── profiles（一人一行）
        user_id, age, gender, chronic_conditions, allergies, current_medications, updated_at
  └── visit_records（一人多行）
        id, user_id, visit_date, department, chief_complaint, conclusion
```

**硬约束注入**：每次 `_respond()` 调用时，`build_constraint_prompt()` 把档案转为 prompt 片段注入 system：

```
[用户健康约束 - 必须严格遵守，不得忽略]
- 过敏史：青霉素（涉及药物建议时必须过滤）
- 慢性病：高血压（影响科室优先级判断）
- 当前用药：二甲双胍（避免药物冲突建议）
- 年龄：33岁，性别：男
```

**首次使用引导**：`user_id != "guest"` 且档案未填写时，CLI 引导用户填写年龄、性别、过敏史、慢性病、当前用药。

### 3.5 MedicationAgent — 用药咨询

```python
class MedicationAgent:
    async def handle(self, user_input: str) -> None:
        # 注入 HealthProfile 硬约束
        constraint_prompt = self._ctx.build_constraint_prompt()
        system = MEDICATION_SYSTEM_PROMPT + constraint_prompt
        # 带完整对话历史，单次 GPT-4o 调用
        messages = [system] + ctx.messages
        response = gpt-4o(messages, max_tokens=600)
```

**三类问题**：药物用途查询、副作用咨询、药物冲突检查。

**硬约束注入**：过敏史和当前用药强制写入 system prompt，LLM 回答药物冲突时必须参考。

**免责声明**（硬编码在 system prompt 里）：
```
⚠️ 重要声明：以下信息仅供参考，实际用药请以药品说明书、医生或执业药师的建议为准。
```

**已知局限**：
- GPT-4o 直接回答，训练数据有截止日期，新药/撤市药/最新冲突研究可能不准确
- temperature 默认 1.0，同一问题两次回答详细程度可能不同
- 模糊问题（"我能吃止痛药吗"）由 LLM 在回答里自然引导，无显式追问状态机

**后续扩展**（未实现）：
- 接药物数据库 API（NMPA/RxNorm/OpenFDA），结构化数据 + LLM 组织语言
- 多模态输入：拍药盒/说明书图片，视觉模型提取药物名后查库
- 接 API 后需加追问状态机，提取标准化药物名

### 3.6 EpisodicMemory — 分诊记录

分诊结束后自动保存，下次会话软参考。

```
保存时机：TriageAgent._respond() 最后，emit(RESULT) 之后
保存内容：OPQRST 摘要 + 建议科室（candidates[0].department）+ 建议文本（截断 500 字）
展示时机：CLI 会话开始时，非 guest 用户展示最近 3 条
注入时机：每次 _respond() 时 build_history_prompt() 注入 system prompt（软参考）
```

**硬约束 vs 软参考**：

| | HealthProfile | EpisodicMemory |
|---|---|---|
| 内容 | 过敏史、慢性病、用药 | 历次分诊记录 |
| 性质 | 静态，不常变 | 动态，每次就诊更新 |
| 注入方式 | system prompt 强制声明 | system prompt 供参考 |
| 措辞 | "必须严格遵守，不得忽略" | "供参考，非硬约束" |
| 失效行为 | LLM 忽略 = 医疗安全问题 | LLM 忽略 = 体验略差 |

---

## 四、数据流转全流程（Phase 2）

### 4.1 完整对话流转

```
用户输入："我腹泻，从昨天开始吃了韩国酱蟹"
    │
    ▼
[OrchestratorAgent.classify_intent()]
    注入：dialogue_state=INIT + symptom_summary="（无）" + 近期历史
    gpt-4o-mini 判断：symptom
    │
    ▼
[TriageAgent.handle()]
    ctx.add_user_message("我腹泻...")
    evaluate_urgency_by_rules() → 未命中
    │
    ▼
[_think()]
    transition(COLLECTING)
    _extract_symptom_info()
      NER: "腹泻" → 症状实体（非部位），region = None
      关键词 O: "吃" → onset = "我腹泻，从昨天开始吃了韩国酱蟹"
      关键词 T: "昨" → time_pattern = "我腹泻，从昨天开始吃了韩国酱蟹"
    region 为空 → _enrich_region()
      gpt-4o-mini: "腹泻" → "腹部"
      region = "腹部"
    is_sufficient(): region=腹部 + onset=有 → True
    │
    ▼
[_act_search()]
    transition(SEARCHING)
    to_query_text() = "我腹泻... 腹部 我腹泻... 吃了韩国酱蟹"
    DepartmentRouter.route(query) → [消化内科(82%), 普通内科(71%), ...]
    │
    ▼
[_observe()] → 透传
    │
    ▼
[_respond()]
    constraint_prompt = "年龄：33岁，性别：男"
    history_prompt = "2026-04-28 | 消化内科 | 部位：腹部..."（上次记录）
    symptom_summary = "部位：腹部\n时间：...\n诱因：..."
    
    messages = [system] + ctx.messages + [retrieval_note]
    gpt-4o → "建议消化内科，紧急程度：普通..."
    
    ctx.add_assistant_message(content)
    episodic.save(symptom_summary, content, "消化内科")
    transition(INIT)
    │
    ▼
bus.emit(RESULT) → CLI 打印建议
```

### 4.2 followup 流转

```
用户输入："消化内科在几楼"
    │
    ▼
[OrchestratorAgent.classify_intent()]
    注入：dialogue_state=INIT + 近期历史（含上次建议）
    gpt-4o-mini 判断：followup
    │
    ▼
[OrchestratorAgent.handle_followup()]
    ctx.add_user_message("消化内科在几楼")
    messages = [system] + ctx.messages
    gpt-4o-mini → "具体楼层请向医院服务台询问..."
    ctx.add_assistant_message(content)
    │
    ▼
bus.emit(RESULT) → CLI 打印回答
```

### 4.3 new_symptom 重置流转

```
用户输入（分诊已完成后）："我头很疼"
    │
    ▼
[OrchestratorAgent.classify_intent()]
    注入：dialogue_state=INIT（上轮已完成）
    gpt-4o-mini 判断：new_symptom
    │
    ▼
[CLI]
    ctx.messages.clear()        ← 清空历史，避免串扰
    agent._symptom_info = SymptomInfo()  ← 重置症状信息
    │
    ▼
[TriageAgent.handle()] → 正常 TAOR 流程（干净状态）
```

---

## 五、已知局限与设计权衡

### 5.1 分诊完成后伴随症状误判为 new_symptom

**场景**：
```
用户：我腹泻，昨天开始    → 分诊给出建议，state=INIT
用户：我头也有点晕        → dialogue_state=INIT → 判 new_symptom → 清空历史
```

**问题**：头晕可能是腹泻的伴随症状（脱水），清空历史导致 LLM 看不到腹泻上下文，建议不准确。

**根本原因**：`is_sufficient()` 是规则判断，分诊完成后重置 `SymptomInfo`，LLM 的 `ctx.messages` 历史和结构化 `SymptomInfo` 生命周期不一致。

**修复方向**：把充分性判断改为 LLM 驱动（Think 阶段额外一次 LLM 调用），传入完整对话历史判断"当前信息是否足够出建议"。成本：每轮多一次 gpt-4o-mini 调用，延迟+约 300ms。

**当前决策**：暂不修复，Phase 3 考虑。保守策略（分诊完成后新输入判 new_symptom）的失败代价比错误判 symptom 小——前者最多多问一次，后者会把不相关症状混入建议。

### 5.2 OPQRST 关键词提取覆盖不全

O/P/Q/S/T 字段的提取依赖关键词列表，口语化表达（"好像是前天""越走越疼"）可能无法命中。R 字段有 LLM 推断兜底，其他字段暂无。

**当前决策**：缺失字段不影响 `is_sufficient()` 判断（只要 R + T/O 有），由追问机制补充，由最终的 GPT-4o 建议通过 `ctx.messages` 获取原始描述，不依赖结构化字段完整。

### 5.3 科室提取改进历程

Phase 2 初版 `EpisodicMemory.save()` 通过解析建议文本提取科室名，依赖 LLM 输出格式（列表 vs 行内），格式不固定导致频繁返回"待确认"。

修复：直接传入 `candidates[0].department`（向量检索的结构化结果），不解析文本。

**教训**：能用结构化数据的地方不要解析非结构化文本，LLM 输出格式永远不稳定。

---

## 六、LLM 调用汇总

Phase 2 中一次完整分诊（含追问）涉及的 LLM 调用：

| 调用点 | 模型 | max_tokens | 触发条件 |
|--------|------|-----------|----------|
| `decompose_input()` | gpt-4o-mini | 150 | 每轮必调（检测混合输入） |
| `classify_intent()` | gpt-4o-mini | 15 | 每个子问题各调一次 |
| `_enrich_region()` | gpt-4o-mini | 10 | NER 未提取到部位时 |
| `build_follow_up_question()` | gpt-4o-mini | 80 | 信息不足需追问时 |
| `_act_search()` Tool Use | gpt-4o-mini | 100 | 信息充分后，LLM 决定调用 search_symptom_kb |
| `_respond()` | gpt-4o | 1000 | 工具返回结果后生成建议 |
| `MedicationAgent.handle()` | gpt-4o | 600 | medication 意图时 |
| `handle_followup()` | gpt-4o-mini | 300 | followup 意图时 |

**模型路由原则**：轻量判断用 gpt-4o-mini（快、便宜），深度推理用 gpt-4o（准）。

---

## 七、开发过程中遇到的问题与反思

### 7.1 typer 入口方式反复

**问题**：Phase 1 把入口指向 `medi.cli:chat` 函数，Phase 2 加入 `OrchestratorAgent` 后需要在 chat 函数外初始化更多对象，导致反复在 `medi.cli:app` 和 `medi.cli:chat` 之间切换，引发 typer `OptionInfo` 类型错误。

**根本原因**：typer 的 `@app.command()` 装饰器和直接把函数作为入口的行为不同，混用导致参数解析错误。

**解决**：统一用 `medi.cli:app`，配合 `invoke_without_command=True`，`medi --user-id jeremy` 直接触发默认命令。

### 7.2 每轮 bus 重建 vs agent 复用

**问题**：第一版实现中每轮调用结束后 `bus.close()`，下轮开始时 `bus` 已关闭，新事件无法被消费。

**解决**：每轮重建 `bus = AsyncStreamBus()`，通过 `agent._bus = bus` 和 `orchestrator._bus = bus` 替换引用，agent 本身不重建（保留 NER 模型缓存、symptom_info 积累）。

**设计原则**：总线是无状态的通道，应该随每轮通信重建；Agent 是有状态的处理器，应该跨轮复用。

### 7.3 NER 模型对症状词提取部位失败

**问题**："腹泻"是症状词，NER 识别为症状实体而不是部位实体，`region` 始终为空，导致系统不断追问"哪里不舒服"——用户说"腹泻"当然是肚子，追问体验很差。

**方案一（拒绝）**：维护症状词→部位的映射表（"腹泻→腹部"）。问题：硬编码，维护成本随词汇增长线性增加。

**方案二（采用）**：NER 提取不到时，调用 gpt-4o-mini 推断部位，覆盖任意症状词，维护成本固定。

**教训**：同 Phase 1 的关键词 vs NER 决策，语言理解类任务不要用规则对抗语言多样性。

### 7.4 意图分类中 followup 误判

**问题**：急救场景（ESCALATION）后，`ctx.messages` 最后一条是急救提示。用户追问"消化内科在几楼"时，分类器看到急救历史，把"科室在哪"判断为 symptom 而不是 followup。

**根本原因**：初版 followup 描述过于具体（"追问刚才给出的分诊建议"），与急救场景的上下文不匹配。

**修复**：把描述改为"追问对话中任何已有内容，包括科室位置、就诊流程、紧急程度含义"，同时注入 `dialogue_state` 让分类器有更多上下文。

**教训**：意图描述是 prompt 工程，要覆盖所有合理的用户行为，不要假设固定的对话前置状态。

### 7.5 EpisodicMemory 科室解析失败

**问题**：通过解析 LLM 建议文本提取科室名，LLM 输出格式不稳定（有时"消化内科，普通外科"行内，有时"• 消化内科"列表），正则匹配频繁失败，记录显示"待确认"。

**修复**：直接使用向量检索的结构化结果 `candidates[0].department`，不依赖 LLM 输出格式。

**教训**：设计数据存储时，能从结构化来源拿的数据就不要从非结构化文本解析，LLM 输出格式不可信赖。

### 7.6 MedicationAgent 回答不稳定

**现象**：同一个药物冲突问题，两次回答详细程度不同——第二次回答提到了"扑尔敏"等具体成分，第一次没有。

**原因**：
1. temperature 默认 1.0，生成有随机性
2. 第二次调用时 `ctx.messages` 里有上一次的对话历史，LLM 看到更多上下文，倾向于更详细的回答

**根本问题**：GPT-4o 直接回答的输出质量依赖上下文和随机性，不可预测。接药物数据库 API 后，结构化字段固定，回答稳定性大幅提升。

**当前决策**：不设 temperature=0（适度随机性让回答更自然），接受这个局限，等接 API 后从根本上解决。

---

## 八、Phase 3 已完成升级

### 8.1 Tool Use — TriageAgent 从流水线升级为 ReAct Agent

**升级前（Phase 1/2）**

```python
# _act_search() 直接调 Python 方法，LLM 不参与决策
query = self._symptom_info.to_query_text()
candidates = await self._router.route(query)
```

流水线模式：代码硬编码"一定要检索"，LLM 只负责最后生成建议。

**升级后（Phase 3）**

```python
# LLM 收到症状摘要，自主决定调用工具
response = await client.chat.completions.create(
    tools=[SEARCH_SYMPTOM_KB_SCHEMA],
    tool_choice="auto",
    messages=act_messages,
)
# ToolRuntime 执行工具，把结果返回给 LLM
tool_result = await self._tool_runtime.call("search_symptom_kb", query=...)
```

ReAct 模式：LLM 自主决策 → ToolRuntime 执行 → LLM 拿结果生成建议。

**新增文件：`medi/agents/triage/tools.py`**

```python
SEARCH_SYMPTOM_KB_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_symptom_kb",
        "description": "在症状-科室知识库中检索，根据症状描述返回最匹配的科室候选列表",
        "parameters": {"query": "症状描述文本"},
    },
}

def make_search_tool(router: DepartmentRouter) -> ToolDefinition:
    """把 DepartmentRouter.route() 包装为 ToolDefinition"""
    async def search_symptom_kb(query: str) -> dict:
        candidates = await router.route(query)
        return {"candidates": [...]}  # 序列化为 dict 供 LLM 读取
    return ToolDefinition(name="search_symptom_kb", priority=STANDARD, fn=search_symptom_kb)
```

**ToolRuntime 在 Phase 1 已实现，Phase 3 正式接入**

Phase 1 的 ToolRuntime 包含：权限检查（`ctx.has_tool()`）、分级超时、重试、审计日志。Phase 3 通过 `make_search_tool()` 注册工具后，这些能力自动生效。

**为什么不强制调用工具（`tool_choice="required"`）**

强制调用退化成流水线——和直接调 `DepartmentRouter` 没有区别。`tool_choice="auto"` 保留 LLM 自主判断空间，为未来多工具场景（"要不要同时查药物冲突？"）打基础。

**扩展多工具无需改 Agent 主流程**

未来加 `check_drug_interaction`、`get_lab_indicators` 等工具，只需：
1. 在 `tools.py` 定义新工具
2. `self._tool_runtime.register(make_new_tool())` 注册
3. 把新 schema 加入 `tools` 列表

TriageAgent 的 `_act_search()` 主循环不需要修改。

### 8.2 就诊记录展示优化

- **科室名清洗**：`_clean_department()` 去掉数据集携带的序号（"1. 神经科" → "神经科"）
- **多行展示**：`chief_complaint` 从截断单行改为按 OPQRST 字段分行展示

---

## 九、Phase 3 待办

| 模块 | 内容 | 优先级 |
|------|------|--------|
| MedicationAgent 接 API | 替换 GPT-4o，接 NMPA/RxNorm/OpenFDA 药物数据库 | 高 |
| MedicationAgent 多模态 | 拍药盒/说明书图片，视觉模型提取药物名后查库 | 中 |
| 多工具扩展 | 注册 check_drug_interaction、get_lab_indicators 等工具 | 中 |
| Think 阶段 LLM 充分性判断 | 替代规则 `is_sufficient()`，解决分诊完成后伴随症状误判问题 | 中 |
| 全量向量索引 | build_index.py 跑全量 17.7 万条 | 中 |
| MemoryDistiller | 设计有缺陷，等历史数据积累后重新设计 | 低 |
| API 层 | FastAPI 接口，供前端或移动端接入 | 低 |
