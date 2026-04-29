# Medi Phase 3 — 开发回顾文档

## 一、Phase 3 目标与完成情况

### 计划目标

| 任务 | 状态 |
|------|------|
| 跨供应商模型降级策略（LLMProvider 抽象层）| 完成 |
| 可观测性采集（LLM 调用、工具调用、阶段耗时）| 完成 |
| `medi observe` CLI 查询命令 | 完成 |
| decompose_input 设计反思与重构 | 完成（删除，改为单一意图路由）|

**未完成项（顺延 Phase 4）**：
- MedicationAgent 接药物数据库 API（NMPA/RxNorm/OpenFDA）
- MedicationAgent 多模态输入（拍药盒图片）
- 全量向量索引（17.7 万条，当前仍为 2000 条测试量）
- MemoryDistiller（等历史数据积累后设计）
- FastAPI 层
- 用药提醒调度

---

## 二、系统架构

### 2.1 整体分层（Phase 3 新增部分用 ★ 标注）

```
用户层
  CLI（typer + rich）
    ★ medi observe          — 可观测性查询子命令
    ★ medi chat             — 对话子命令（原 medi --user-id）

Agent 层
  OrchestratorAgent（意图分类 + 路由）
    ├── classify_intent()   — 整句话单一意图路由（★ 删除 decompose_input）
    ├── symptom     → TriageAgent
    ├── new_symptom → 清空历史 → TriageAgent
    ├── medication  → MedicationAgent
    ├── followup    → OrchestratorAgent 直接回答
    └── out_of_scope → 边界提示

  MedicationAgent（用药咨询，无变化）
  TriageAgent（TAOR 主流程，无变化）

基础设施层
  ★ LLMProvider 抽象层（core/providers.py）
    ├── OpenAIProvider      — OpenAI 官方 API
    ├── QwenProvider        — 阿里云通义千问（OpenAI 兼容格式）
    └── LocalProvider       — 本地 Ollama（离线兜底）
  ★ call_with_fallback()   — 跨供应商降级调用（core/llm_client.py）
  ★ ObservabilityStore     — 可观测性数据采集与存储（core/observability.py）
  UnifiedContext
    ★ observability         — 新增字段，注入 ObservabilityStore
    ★ model_config          — smart_chain/fast_chain 替代 fast/smart 字符串
  AsyncStreamBus
  ToolRuntime
    ★ ToolTrace 采集         — 工具调用耗时和成功/失败

记忆层
  HealthProfile（SQLite，无变化）
  EpisodicMemory（SQLite，无变化）

知识层
  ChromaDB（无变化）
  bge-large-zh-v1.5（无变化）
  bert-NER（无变化）
```

### 2.2 模型降级链

```
smart 模型降级链（按环境变量动态构建）：
  OpenAI gpt-4o
    ↓ RateLimitError / APIStatusError / APITimeoutError
  通义千问 qwen-max（需 DASHSCOPE_API_KEY）
    ↓ 失败
  本地 Ollama qwen2.5:7b（无需 API Key，始终可尝试）

fast 模型降级链：
  OpenAI gpt-4o-mini
    ↓ 失败
  通义千问 qwen-turbo（需 DASHSCOPE_API_KEY）
    ↓ 失败
  本地 Ollama qwen2.5:7b
```

**动态构建**：`build_smart_chain()` / `build_fast_chain()` 在 `ModelConfig.__post_init__()` 时根据环境变量决定链的组成，没有对应 Key 的供应商自动跳过。

---

## 三、核心模块详解

### 3.1 LLMProvider 抽象层（core/providers.py）

**设计动机**

Phase 2 所有 LLM 调用直接用 `AsyncOpenAI`，供应商和调用逻辑耦合。当需要降级到通义千问或本地模型时，每个调用点都要修改。

**统一接口**

```python
class LLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def create(self, messages, max_tokens, **kwargs): ...
```

调用方只依赖 `LLMProvider` 接口，不感知具体供应商。

**三个适配器**

| 适配器 | 供应商 | 特殊处理 |
|--------|--------|----------|
| `OpenAIProvider` | OpenAI 官方 | 完整支持 tool use / function calling |
| `QwenProvider` | 阿里云通义千问 | 兼容 OpenAI 格式，过滤 tools/tool_choice 参数 |
| `LocalProvider` | 本地 Ollama | 无需 API Key，过滤 tools/tool_choice 参数 |

**为什么 Qwen/Local 要过滤 tools 参数**

通义千问和 Ollama 对 OpenAI tool use 协议的支持不完整，传入 tools 参数会报错。降级场景下放弃 tool use，LLM 直接生成建议（不经向量检索），是可接受的降级代价。

### 3.2 call_with_fallback()（core/llm_client.py）

```python
async def call_with_fallback(
    chain: list[LLMProvider],
    bus: AsyncStreamBus,
    session_id: str,
    messages: list[dict],
    max_tokens: int,
    obs: ObservabilityStore | None = None,
    call_type: str = "unknown",
    **kwargs,
):
```

**降级流程**

```
for i, provider in enumerate(chain):
    try:
        response = await provider.create(messages, max_tokens, **kwargs)
        # 成功 → 记录 LLMTrace(success=True) → 返回
    except (RateLimitError, APIStatusError, APITimeoutError):
        # 失败 → 记录 LLMTrace(success=False, error_type=...) → 向 bus 发 ERROR 事件 → 继续
# 全部失败 → raise 最后一个异常
```

**每次调用（成功或失败）都写入 ObservabilityStore**，记录供应商、call_type、token 消耗、耗时、是否降级。

**`call_type` 字段**

标注每次调用的用途，供 `medi observe` 展示：

| call_type | 调用位置 |
|-----------|----------|
| `intent_classify` | OrchestratorAgent.classify_intent() |
| `followup_answer` | OrchestratorAgent.handle_followup() |
| `enrich_region` | TriageAgent._enrich_region() |
| `act_search` | TriageAgent._act_search() |
| `respond` | TriageAgent._respond() |
| `medication` | MedicationAgent.handle() |

### 3.3 ObservabilityStore（core/observability.py）

**三类 Trace**

```python
@dataclass
class LLMTrace:
    session_id, timestamp, provider, call_type
    is_fallback, prompt_tokens, completion_tokens
    latency_ms, success, error_type

@dataclass
class ToolTrace:
    session_id, timestamp, tool_name
    latency_ms, success, error_msg

@dataclass
class StageTrace:
    session_id, timestamp, stage   # think / act / respond
    latency_ms
```

**采集点**

| 数据 | 采集位置 | 采集方式 |
|------|----------|----------|
| LLM 调用 | `call_with_fallback()` | 成功/失败都记录 |
| 工具调用 | `ToolRuntime.call()` | 成功/失败都记录 |
| TAOR 阶段 | `TriageAgent._think/_act_search/_respond` | `_record_stage()` 辅助方法 |
| 追问生成 | 未记录（轻量调用，token 数拿不到，只记耗时意义有限）| — |

**持久化**

每轮对话结束后 `obs.flush()` 写入 SQLite（`data/observability.db`），三张表：`llm_traces`、`tool_traces`、`stage_traces`。

**查询**

```bash
medi observe                        # 最近 10 个会话汇总
medi observe --session <session_id> # 单个会话完整链路
```

### 3.4 decompose_input 的删除

Phase 2 引入 `decompose_input()` 将混合输入拆分为子问题，Phase 3 测试中发现根本性缺陷，决定删除。

**问题复现**

```
输入："我右上腹痛了两天，吃东西后会加重"

Phase 2 错误拆分：
  ['我右上腹痛了两天', '吃东西后会加重']
  → 两次独立分诊 → 两条重复建议
```

**修复 prompt 后仍存在的问题**

```
输入："我发烧38度，布洛芬和阿司匹林能一起吃吗"

拆分后：['我发烧38度', '布洛芬和阿司匹林能一起吃吗']
执行：
  Medi: 请问您感觉不舒服的部位是哪里？  ← 症状追问
  Medi: 布洛芬和阿司匹林不建议同时服用   ← 用药回答
  # 追问发出后不等用户回答，用药回答紧接输出，体验割裂
```

**根本矛盾**

追问需要等用户下一轮回答才能继续，但 for 循环无法在中途暂停等待用户输入。两个子问题在同一轮内全部执行，追问失去意义。

**两个完整问题并存时无解**

```
输入："我昨天胸部右上方有些抽痛该去哪里就诊？我能吃家里有的xxx药吗？"
```

两个问题信息都完整，无论哪个作为主意图，另一个都面临被丢弃或乱序输出的问题。CLI 的线性输出无法承载并行回答块。

**最终决策：方案 A — 单一意图路由 + 引导分步提问**

整句话走 `classify_intent`，识别主意图处理，次要问题引导用户下一轮单独提问。

测试验证：
```
输入："我右上腹痛了两天，看什么科室？家里的喇叭丸能先吃一下吗？"
→ classify_intent: SYMPTOM
→ TriageAgent 分诊，建议里自然带入"喇叭丸未明确诊断前不建议使用"
→ 一条回答覆盖两个问题，体验流畅
```

LLM 在生成建议时能自然吸收次要问题，大多数混合输入场景不需要显式拆分。

详细分析见：`docs/decompose_input_retrospective.md`

---

## 四、可观测性实测数据

以下为一次完整分诊（两轮对话）的 `medi observe --session` 输出：

**TAOR 阶段耗时**

| 阶段 | 耗时 |
|------|------|
| think（第一轮）| 833ms |
| think（第二轮）| 22ms（NER 已缓存）|
| act | 4283ms（含向量检索 2811ms）|
| respond | 1859ms |

**LLM 调用记录**

| 调用类型 | 供应商 | Prompt Token | Completion Token | 耗时 |
|----------|--------|-------------|-----------------|------|
| intent_classify | gpt-4o-mini | 253 | 2 | 812ms |
| intent_classify | gpt-4o-mini | 303 | 2 | 442ms |
| act_search | gpt-4o-mini | 367 | 35 | 1467ms |
| respond | gpt-4o | 554 | 100 | 1856ms |

**工具调用记录**

| 工具名 | 耗时 | 成功 |
|--------|------|------|
| search_symptom_kb | 2811ms | Y |

**观察**：
- 第一次 think 耗时 833ms（NER 模型冷启动），第二次只有 22ms（模型已缓存）
- `act_search` 里 2811ms 是向量检索（ChromaDB + bge-large-zh 计算），非 LLM 耗时
- 全程无降级，OpenAI 正常

---

## 五、LLM 调用汇总（Phase 3 更新）

| 调用点 | call_type | 模型 | max_tokens | 说明 |
|--------|-----------|------|-----------|------|
| `classify_intent()` | intent_classify | gpt-4o-mini | 15 | 每轮必调 |
| `_enrich_region()` | enrich_region | gpt-4o-mini | 10 | NER 失败时 |
| `build_follow_up_question()` | — | gpt-4o-mini | 80 | 未纳入可观测性（轻量）|
| `_act_search()` | act_search | gpt-4o-mini | 100 | Tool Use 决策 |
| `_respond()` | respond | gpt-4o | 1000 | 最终建议 |
| `MedicationAgent.handle()` | medication | gpt-4o | 600 | 用药咨询 |
| `handle_followup()` | followup_answer | gpt-4o-mini | 300 | 追问已有内容 |

**Phase 3 移除**：`decompose_input()` 调用（删除该功能）

---

## 六、开发过程中遇到的问题与反思

### 6.1 同供应商降级无意义

**初版设计**：`ModelConfig` 加 `smart_fallbacks: tuple = ("gpt-4o-mini",)`，gpt-4o 失败时降级到 gpt-4o-mini。

**问题**：OpenAI 整体限流或宕机时，同供应商的所有模型一样不可用，降级没有任何效果。

**修正**：改为跨供应商降级链（OpenAI → 通义千问 → 本地 Ollama），确保供应商多样性是核心，不是模型版本多样性。

**面试表述**：
> "模型降级的核心是供应商多样性，不是模型版本降级。同一供应商限流时所有模型都受影响，真正有效的降级链必须跨供应商。"

### 6.2 LLM 输出格式不可信任（再次验证）

Phase 2 已遇到 EpisodicMemory 科室解析失败问题，Phase 3 的 decompose_input 再次验证：LLM 返回 JSON 时，markdown 代码块包裹、`json` 前缀、键名大小写都可能不一致，需要多层容错处理。

删除 decompose_input 后，这个问题连同容错代码一起消失，是"删功能比修 bug 更优雅"的典型案例。

### 6.3 ObservabilityStore 采集点的取舍

**build_follow_up_question 不纳入可观测性**

`build_follow_up_question()` 是独立纯函数，直接使用 `AsyncOpenAI`，没有 bus/obs 参数。给它加参数会破坏单一职责，且 token 数在纯函数里拿不到。

**决策**：追问生成是轻量调用，耗时不是关键指标，不纳入可观测性。`_act_search` 里的 tool use 和 `_respond` 的最终建议才是需要监控的核心调用。

### 6.4 decompose_input 的过度设计

**过度设计的识别特征**：
- 引入时感觉合理（混合输入确实存在）
- 实现后产生了比解决的更多的问题（拆分错误、追问乱序、体验割裂）
- 删除后系统反而更简单、更稳定

**教训**：边缘场景（混合输入频率低）不值得引入复杂机制。在医疗场景下，宁可让用户分步提问，也不能让系统输出混乱的回答。

---

## 七、京东健康面试题覆盖分析（Phase 3 更新）

| 面试题 | 覆盖情况 |
|--------|----------|
| 智能分诊 Agent 设计 | 完整（TriageAgent + TAOR + OPQRST + 状态机）|
| Tool Calling 可靠性、超时与异常 | 完整（ToolRuntime 分级 + call_with_fallback 降级 + ObservabilityStore 监控）|
| Agent 记忆机制 | 完整（HealthProfile + EpisodicMemory + ChromaDB）|
| 模型抽象与供应商切换 | ★ Phase 3 完成（LLMProvider + 跨供应商降级链）|
| 智能用药依从性管理 Agent | 部分（MedicationAgent 框架，提醒调度未做）|
| 多 Agent 协同 | 部分（Orchestrator 路由，无真正协作）|
| Continual Learning | 未做 |

---

## 八、Phase 4 待办

| 模块 | 内容 | 优先级 |
|------|------|--------|
| FastAPI 层 | HTTP API，SSE 流式返回，供前端/移动端接入 | 高 |
| MedicationAgent 接 API | 替换 GPT-4o，接真实药物数据库 | 高 |
| 用药提醒 | 定时调度 + 漏服处理，依赖 FastAPI 后台常驻 | 中 |
| MedicationAgent 多模态 | 拍药盒图片，视觉模型提取药物名后查库 | 中 |
| 全量向量索引 | build_index.py 跑全量 17.7 万条 | 中 |
| 多 Agent 协同 | 体检报告解读 → 膳食建议 → 日程协调，Orchestrator 编排 | 中 |
| MemoryDistiller | 等历史数据积累后设计 | 低 |
