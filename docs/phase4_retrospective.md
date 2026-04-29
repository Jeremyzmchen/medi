# Medi Phase 4 — 开发回顾文档

## 一、Phase 4 目标与完成情况

### 计划目标

| 任务 | 状态 |
|------|------|
| FastAPI HTTP 层（POST /chat、GET /chat/stream SSE）| 完成 |
| PDF 体检报告上传解析（POST /upload/report）| 完成 |
| 多 Agent 协同 Pipeline（体检报告解读 → 膳食 → 日程）| 完成 |
| `medi serve` CLI 子命令 | 完成 |

**未完成项（顺延或放弃）**：
- 用药提醒（定时调度 + 通知渠道）— 顺延，后续结合站内消息和邮件一起做
- MedicationAgent 接真实药物数据库 — 顺延
- MemoryDistiller — 放弃（测试发现历史洞察注入会干扰当前分诊判断）
- 全量向量索引（17.7 万条）— 顺延（跑脚本即可，不影响架构）
- Continual Learning — 规划中，复杂度高

---

## 二、系统架构

### 2.1 整体分层（Phase 4 新增部分用 ★ 标注）

```
用户层
  CLI（typer + rich）
    medi chat          — 对话
    medi observe       — 可观测性查询
    ★ medi serve       — 启动 FastAPI HTTP 服务

  ★ HTTP API（FastAPI + uvicorn）
    POST /chat             — 单轮对话（JSON 响应）
    GET  /chat/stream      — SSE 流式对话
    POST /upload/report    — PDF 体检报告上传
    GET  /observe          — 最近会话摘要
    GET  /observe/{id}     — 单会话完整链路
    GET  /health           — 健康检查

Agent 层
  OrchestratorAgent（意图分类 + 路由）
    ├── symptom       → TriageAgent
    ├── new_symptom   → 清空历史 → TriageAgent
    ├── medication    → MedicationAgent
    ├── followup      → OrchestratorAgent 直接回答
    ★ health_report  → HealthReportAgent
    └── out_of_scope  → 边界提示

  TriageAgent（TAOR 主流程，无变化）
  MedicationAgent（用药咨询，无变化）
  ★ HealthReportAgent（体检报告 Pipeline 入口）
    ├── _analyze()         — 解读异常指标 → ReportAnalysis
    ├── DietAgent          — 膳食方案 → DietPlan
    └── ScheduleAgent      — 日程规划 → SchedulePlan

基础设施层
  ★ Session 会话管理（api/session_store.py）
  LLMProvider 抽象层（无变化）
  call_with_fallback()（无变化）
  ObservabilityStore（无变化）
  UnifiedContext / AsyncStreamBus / ToolRuntime（无变化）

记忆层
  HealthProfile（SQLite，无变化）
  EpisodicMemory（SQLite，无变化）

知识层
  ChromaDB（无变化）
```

### 2.2 多 Agent Pipeline 数据流

```
用户输入（文字 或 PDF 提取文字）
    ↓
OrchestratorAgent.classify_intent() → HEALTH_REPORT
    ↓
HealthReportAgent._analyze()
    输出：ReportAnalysis
      - summary: str
      - abnormal_indicators: list[AbnormalIndicator]
        └── name / value / reference / interpretation / severity
      - recommendations: list[str]
    ↓
DietAgent.handle(analysis: ReportAnalysis)
    输出：DietPlan
      - summary: str
      - suggestions: list[DietSuggestion]
        └── category / items / reason
      - daily_schedule: list[str]
    ↓
ScheduleAgent.handle(diet: DietPlan, analysis: ReportAnalysis)
    输出：SchedulePlan
      - summary: str
      - weekly_plan: list[str]
      - reminders: list[str]
    ↓
HealthReportAgent._format_final_report() → Markdown 报告
    ↓
RESULT 事件 → CLI 展示 / API 响应
```

---

## 三、核心模块详解

### 3.1 FastAPI 层（medi/api/）

**文件结构**

```
medi/api/
├── app.py            # FastAPI 应用入口，CORS，lifespan
├── schemas.py        # Pydantic 请求/响应模型
├── session_store.py  # 内存级 Session 状态管理
└── routes/
    ├── chat.py       # POST /chat + GET /chat/stream
    ├── observe.py    # GET /observe + GET /observe/{id}
    └── upload.py     # POST /upload/report
```

**Session 管理**

每个 session 持有 `UnifiedContext`（对话历史、状态机、健康档案）和所有已初始化的 Agent，跨轮复用，避免重复初始化。

```python
@dataclass
class Session:
    session_id: str
    ctx: UnifiedContext
    agent: TriageAgent
    orchestrator: OrchestratorAgent
    medication_agent: MedicationAgent
    health_report_agent: HealthReportAgent
    obs: ObservabilityStore
```

每轮请求开始时新建 `AsyncStreamBus` 并通过 `rebind_bus()` 注入所有 Agent，旧 Bus 已关闭后重建，保证事件流互不干扰。

**为什么每轮重建 Bus 而不是复用**

`AsyncStreamBus.close()` 发送 sentinel 关闭所有订阅者，关闭后不可再发事件。多轮对话必须每轮新建 Bus，Agent 引用通过 `rebind_bus()` 统一更新，避免遗漏。

**SSE 流式响应**

```
GET /chat/stream?message=...&user_id=...&session_id=...

每个 StreamEvent → 一条 SSE data 行：
  data: {"event_type": "follow_up", "content": "...", "session_id": "..."}

结束标记：
  data: {"event_type": "done", "content": "", "session_id": "..."}
```

前端用原生 `EventSource` 消费，无需 WebSocket，单向推送足够。

### 3.2 PDF 解析（medi/api/routes/upload.py）

**技术选型**

使用 `pypdf` 提取电子版 PDF 文字层，不依赖 OCR 或视觉模型。

**为什么不做 Vision 降级**

医院出具的电子体检报告基本都有文字层（非扫描件）。Vision 降级（PDF 转图片 → GPT-4o）token 消耗高（每张图约 1000+ token），实际需求低，复杂度不值得。

**提取失败处理**

```python
if len(text.strip()) < 50:
    raise HTTPException(422, "无法提取文字，请手动输入关键指标")
```

扫描件场景返回明确错误，引导用户降级到文字输入，不做静默失败。

**流程**

```
POST /upload/report（multipart/form-data）
    ↓
pypdf 提取文字（失败 → 422）
    ↓
文字 < 50 字符 → 422（可能是扫描件）
    ↓
"以下是我的体检报告内容：\n\n{text}"
    ↓
HealthReportAgent.handle()（与文字输入路径完全一致）
```

### 3.3 多 Agent 协同设计

**Agent 间通信：结构化数据而非自由文本**

三个 Agent 之间通过 `ReportAnalysis` / `DietPlan` / `SchedulePlan` dataclass 传递，而非把上游输出作为字符串传给下游。

好处：
- 下游 Agent 能精确访问需要的字段（如 `analysis.abnormal_indicators`），不受上游表达方式影响
- 字段类型明确，减少 LLM 解析歧义
- 单个 Agent 的 prompt 可以更聚焦

**进度事件推送**

Pipeline 内部每个阶段开始时 emit 一条进度 RESULT 事件：

```python
await self._bus.emit(StreamEvent(
    type=EventType.RESULT,
    data={"content": "正在解读体检报告，请稍候..."},
    ...
))
```

用户（CLI 或 SSE 消费方）可以实时看到当前阶段，避免长时间等待无反馈。

**LLM call_type 新增**

| call_type | 调用位置 |
|-----------|----------|
| `report_analyze` | HealthReportAgent._analyze() |
| `diet_plan` | DietAgent.handle() |
| `schedule_plan` | ScheduleAgent.handle() |

---

## 四、LLM 调用汇总（Phase 4 更新）

| 调用点 | call_type | 模型链 | max_tokens | 说明 |
|--------|-----------|--------|-----------|------|
| `classify_intent()` | intent_classify | fast_chain | 15 | 每轮必调 |
| `_enrich_region()` | enrich_region | fast_chain | 10 | NER 失败时 |
| `build_follow_up_question()` | — | fast_chain | 80 | 未纳入可观测性 |
| `_act_search()` | act_search | fast_chain | 100 | Tool Use 决策 |
| `_respond()` | respond | smart_chain | 1000 | 最终分诊建议 |
| `MedicationAgent.handle()` | medication | smart_chain | 600 | 用药咨询 |
| `handle_followup()` | followup_answer | fast_chain | 300 | 追问已有内容 |
| `HealthReportAgent._analyze()` | report_analyze | smart_chain | 1000 | 体检报告解读 |
| `DietAgent.handle()` | diet_plan | smart_chain | 800 | 膳食方案 |
| `ScheduleAgent.handle()` | schedule_plan | smart_chain | 1000 | 周健康日程 |

**Phase 4 新增**：report_analyze / diet_plan / schedule_plan（体检报告 Pipeline）

---

## 五、开发过程中遇到的问题与反思

### 5.1 MemoryDistiller 放弃的决策

**初始设想**：会话结束后用 LLM 提炼对话洞察，存入用户档案，下次对话时注入 context，让 Agent 记住用户历史模式。

**测试发现**：历史洞察注入后，LLM 倾向于用历史模式预设当前症状。用户上次因胃病就诊，本次描述头痛时，LLM 仍会过度关联消化系统。医疗分诊场景要求每次独立评估当前主诉，历史偏见反而降低准确性。

**决策**：放弃 MemoryDistiller。EpisodicMemory 只展示历史记录（供用户参考），不注入分诊 context。

**面试表述**：
> "记忆注入并非越多越好。在分诊场景，历史就诊记录会引入锚定偏差，让模型预设诊断方向。我们最终选择展示历史记录但不注入推理 context，把判断权留给当前症状描述。"

### 5.2 多 Agent Pipeline 中的 JSON 解析容错

三个 Agent 都要求 LLM 输出严格 JSON，但 LLM 经常：
- 用 markdown 代码块包裹（` ```json ... ``` `）
- 输出 `json` 前缀
- 键名大小写不一致

每个 Agent 都做了统一容错：

```python
if raw.startswith("```"):
    raw = raw.split("```")[1]
    if raw.startswith("json"):
        raw = raw[4:]
data = json.loads(raw)
```

解析失败时降级为纯文本摘要（`DietPlan(summary=raw)`），不报错，保证 Pipeline 不中断。

**更优方案**（未实现）：使用 OpenAI `response_format={"type": "json_object"}` 强制 JSON 输出，从根本上消除格式问题。当前容错方案够用，留作后续优化。

### 5.3 Bus 重绑定的必要性

**问题**：Session 跨轮复用，但 AsyncStreamBus 每轮必须新建（旧 Bus 已 close）。Agent 持有 Bus 引用，若不更新，会向已关闭的 Bus 发事件，导致事件丢失。

**解决**：`rebind_bus()` 统一更新所有 Agent 的 `_bus` 引用，包括子 Agent（`health_report_agent._diet_agent._bus`）。

**经验**：Agent 持有可变引用时，需要明确的生命周期管理接口。这里选择显式 `rebind_bus()` 而非在 Agent 内部订阅 Bus 工厂，保持简单直接。

---

## 六、面试题覆盖分析（Phase 4 更新）

| 面试题 | 覆盖情况 |
|--------|----------|
| 智能分诊 Agent 设计 | 完整（TriageAgent + TAOR + OPQRST + 状态机）|
| Tool Calling 可靠性、超时与异常 | 完整（ToolRuntime + call_with_fallback + ObservabilityStore）|
| Agent 记忆机制 | 完整（HealthProfile + EpisodicMemory + ChromaDB）|
| 模型抽象与供应商切换 | 完整（LLMProvider + 跨供应商降级链）|
| 多 Agent 协同 | ★ Phase 4 完成（HealthReportAgent 编排 DietAgent + ScheduleAgent，结构化数据传递）|
| Agent 对外暴露服务 | ★ Phase 4 完成（FastAPI + SSE 流式返回 + PDF 上传）|
| 智能用药依从性管理 Agent | 部分（MedicationAgent 框架，提醒调度规划中）|
| Continual Learning | 规划中 |

---

## 七、Phase 5 规划

| 模块 | 内容 | 优先级 |
|------|------|--------|
| 用药提醒 | APScheduler 定时调度 + 站内消息 + 邮件通知 | 高 |
| MedicationAgent 接 API | 接 NMPA/RxNorm/OpenFDA 真实药物数据库 | 中 |
| response_format JSON | 替换当前手动容错解析，用 OpenAI 强制 JSON 输出 | 中 |
| 全量向量索引 | build_index.py 跑全量 17.7 万条 | 低 |
| MedicationAgent 多模态 | 拍药盒图片，视觉模型提取药物名 | 低 |
