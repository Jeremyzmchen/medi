# Medi Phase 1 — 开发回顾文档

## 一、Phase 1 目标与完成情况

### 计划目标

| 任务 | 状态 |
|------|------|
| 搭建项目骨架（core 层） | 完成 |
| 下载处理症状-科室数据集 | 完成（换用 Huatuo26M-Lite） |
| 构建症状-科室向量知识库 | 完成（2000 条测试量，ChromaDB） |
| 实现 SymptomCollector（追问逻辑） | 完成 |
| 实现 UrgencyEvaluator（规则层） | 完成 |
| 实现 DepartmentRouter（向量检索） | 完成 |
| 实现 TriageAgent TAOR 主流程 | 完成 |
| CLI 接口可用 | 完成 |

**未完成项（顺延 Phase 2）**：
- 向量库全量索引（当前 2000 条，全量 17.7 万条）
- HealthProfile 持久化（用户性别/过敏史等，依赖 Phase 2 记忆系统）

---

## 二、系统架构

### 2.1 整体分层

```
用户层
  CLI（typer + rich）

Agent 层
  TriageAgent
    ├── Safety 层（UrgencyEvaluator 规则扫描）
    ├── Think（NER 提取症状实体）
    ├── Act（DepartmentRouter 向量检索 / 追问）
    ├── Observe（评估检索结果）
    └── Respond（GPT-4o 生成建议）

基础设施层
  UnifiedContext     — 共享上下文 + 对话状态机
  AsyncStreamBus     — 异步事件总线
  ToolRuntime        — 统一工具执行（优先级 + 审计）

知识层
  ChromaDB           — 症状-科室向量索引
  bge-large-zh-v1.5  — 症状文本 embedding
  bert-NER           — 症状实体识别
```

### 2.2 对话状态机

```
INIT
  |
  ├── 检测到红旗症状 ──→ ESCALATING ──→ 输出急救提示 ──→ INIT（重置）
  |
  └── 正常症状
        |
      COLLECTING（Think 阶段，NER 提取实体）
        |
        ├── 信息不足 & 追问次数 < 3 ──→ FOLLOW_UP ──→ 等待用户输入
        |
        └── 信息足够 or 追问次数耗尽
              |
            SUFFICIENT
              |
            SEARCHING（Act 阶段，向量检索）
              |
            RESPONDING（Respond 阶段，LLM 生成建议）
              |
            INIT（重置，等待下一轮）
```

---

## 三、技术栈与依赖

| 依赖 | 版本 | 用途 |
|------|------|------|
| `openai` | 2.x | GPT-4o 生成分诊建议 |
| `chromadb` | 1.5.x | 本地向量数据库，存储症状-科室索引 |
| `sentence-transformers` | 5.x | 运行 bge-large-zh-v1.5，生成症状文本 embedding |
| `transformers` | 5.x | 运行 bert-NER，提取症状实体 |
| `datasets` | 4.x | 加载 Huatuo26M-Lite 数据集构建索引 |
| `typer` | 0.12.x | CLI 框架 |
| `rich` | 13.x | 终端美化输出 |
| `pydantic` | 2.x | 数据模型验证 |
| `anthropic` | - | 计划用于模型路由，当前换用 openai |

**离线模型**（缓存于 `C:\Users\{user}\.cache\huggingface\hub\`）：
- `BAAI/bge-large-zh-v1.5`（约 1.3GB）— 向量检索
- `Adapting/bert-base-chinese-finetuned-NER-biomedical`（约 400MB）— 实体识别

**向量索引**（存储于项目 `data/chroma/`）：
- collection: `symptom_kb`，2000 条，余弦距离空间

---

## 四、数据流转全流程

### 4.1 知识库构建（离线，一次性）

```
FreedomIntelligence/Huatuo26M-Lite（HuggingFace）
    ↓ load_dataset()
    ↓ 过滤空字段、去重（MD5）
17.7 万条 {question, label} 记录（测试用 2000 条）
    ↓ bge-large-zh-v1.5 encode()（加 BGE 检索前缀）
    ↓ batch_size=512，normalize_embeddings=True
向量矩阵 (N, 1024)
    ↓ chromadb.PersistentClient.create_collection(hnsw:space=cosine)
    ↓ collection.add(ids, embeddings, documents, metadatas)
data/chroma/ 持久化索引
```

### 4.2 用户对话流转（在线）

```
用户输入："我肚子右边下面这里疼了两天了，昨晚开始发烧"
    │
    ▼
[Safety 层] evaluate_urgency_by_rules(text)
    ├── 命中红旗关键词（胸痛/昏迷/大量出血等）
    │       → emit(ESCALATION) → emit(RESULT: 急救提示) → 重置
    └── 未命中 → 进入 TAOR
    │
    ▼
[Think] _extract_symptom_info(text)
    ├── NER 模型识别实体
    │     "肚" → 解剖部位 → symptom_info.location = "肚"
    │     "疼" → 症状 → symptom_info.accompanying = ["疼"]
    └── 关键词兜底时间
          "两天" → symptom_info.duration = text
    │
    ├── is_sufficient()=False & assistant_count 未达上限 → emit(FOLLOW_UP: 追问)
    └── is_sufficient()=True → SUFFICIENT
    │
    ▼
[Act] DepartmentRouter.route(query_text)
    query = "我肚子右边下面这里疼了两天了 肚 两天 疼"
    │
    ├── bge encode(BGE_PREFIX + query) → embedding (1024,)
    ├── chroma.query(top_k=15, include=distances+metadatas+docs)
    ├── 余弦距离 → 相似度：similarity = 1 - dist/2
    ├── 按科室聚合，取最高相似度
    └── 返回 top_3 DepartmentCandidate [{department, confidence, reason}]
    │
    ▼
[Observe] 直接透传 candidates（Phase 1 不做额外过滤）
    │
    ▼
[Respond] GPT-4o chat.completions.create()
    system: "你是分诊助手...{health_profile 硬约束}"
    user:   "症状描述: {query_text}\n知识库结果: {dept_list}"
    max_tokens: 1000
    │
    └── emit(RESULT: "1.建议科室 2.紧急程度 3.就医建议")
    │
    ▼
重置状态机 → INIT，等待下一轮
```

### 4.3 事件流（AsyncStreamBus）

```
Agent emit()          CLI consume()
─────────────         ──────────────
STAGE_START           （静默，不展示）
FOLLOW_UP      ──→    打印追问文字
ESCALATION     ──→    打印红色警告
RESULT         ──→    打印 Markdown 建议
TOOL_CALL/RESULT      （静默，供调试用）
```

---

## 五、核心模块详解

### 5.1 UnifiedContext — 共享上下文

所有模块持有同一个 `ctx` 对象，不逐层传参。

```python
@dataclass
class UnifiedContext:
    user_id: str
    session_id: str
    dialogue_state: DialogueState   # 状态机当前状态
    health_profile: HealthProfile | None  # 硬约束（Phase 2 接入）
    model_config: ModelConfig       # 模型名称配置
    enabled_tools: set[str]         # 工具权限白名单
    messages: list[dict]            # 会话内对话历史，用 assistant 消息数判断追问轮次
```

与 Weave 的区别：增加了 `DialogueState` 状态机和 `HealthProfile` 硬约束字段。

### 5.2 AsyncStreamBus — 异步事件总线

```python
# Agent 发布
await bus.emit(StreamEvent(type=EventType.RESULT, data={...}))

# CLI 消费（独立队列，互不干扰）
async for event in bus.stream():
    ...
```

每轮对话重建 `bus`（保证消费者能收到新事件），`agent` 复用（保留症状积累状态），通过 `agent._bus = bus` 替换引用。

**并发模型**：`consume()` 和 `produce()` 用 `asyncio.gather` 真正并发执行，避免 `await` 顺序执行导致的死锁。

### 5.3 ToolRuntime — 工具执行层

三级优先级：

| 优先级 | 超时 | 重试 | 失败行为 | 审计 |
|--------|------|------|----------|------|
| CRITICAL | 10s | 不重试 | 抛出异常，告知用户 | 强制记录 |
| STANDARD | 5s | 3 次 | 抛出异常 | 不记录 |
| OPTIONAL | 3s | 不重试 | 静默跳过 | 不记录 |

Phase 1 中 ToolRuntime 已实现但 TriageAgent 暂未调用（直接调用 DepartmentRouter），Phase 2 用药管理接入后会走 ToolRuntime。

**Phase 3 正式接入（更新）**：Phase 3 将 `_act_search()` 改为 LLM Tool Use 模式，通过 `ToolRuntime.call()` 执行 `search_symptom_kb` 工具，TriageAgent 从流水线升级为 ReAct Agent。Phase 1 设计的权限检查、分级超时、重试、审计日志能力在 Phase 3 正式生效。详见 phase2_retrospective.md § 8.1。

### 5.4 DepartmentRouter — 向量检索

检索策略：检索 `top_k * 5` 条原始结果，按科室聚合取最高相似度，避免单一科室因多条命中而稀释得分。

余弦距离转相似度：`similarity = 1 - distance / 2`（ChromaDB 余弦距离范围 0~2）。

### 5.5 NER 实体提取

模型：`Adapting/bert-base-chinese-finetuned-NER-biomedical`

实体类型映射：
- `解剖部位` → `symptom_info.location`
- `症状` → `symptom_info.accompanying`
- 时间关键词兜底 → `symptom_info.duration`

懒加载：首次调用时初始化，避免 CLI 启动慢。

---

## 六、设计决策记录

### 6.1 为什么用 TAOR 而不是普通 ReAct

ReAct 的 Thought-Action-Observation 是通用的，但分诊任务的结构是已知的：信息收集 → 检索 → 推理 → 建议。TAOR 把四个阶段显式分开，每个阶段可以单独限制 token 用量，Think 阶段对用户不可见，符合医疗场景的信息安全要求。

### 6.2 为什么 Safety 层在 TAOR 之前

LLM 有一定的延迟，在用户描述"胸痛昏迷"时等 LLM 判断再响应是不可接受的。规则层的关键词匹配是毫秒级的，保证了紧急情况的响应速度，也避免了 LLM 幻觉导致的漏判。

### 6.3 为什么用显式状态机而不是让 LLM 决定流程

分诊是多轮引导对话，流程状态（追问几次了、信息是否足够）需要跨轮次保留。让 LLM 每轮自己决定"下一步做什么"会导致：行为不稳定、无法测试、无法审计。状态机让流程可预测、可调试。

### 6.4 为什么健康档案是硬约束而不是软约束

Weave 的记忆（洞察）是软约束，LLM 可以参考也可以忽略。过敏史不能被忽略——如果用户对青霉素过敏，建议用药时必须排除，这是强制约束，体现在 system prompt 的措辞上（"必须严格遵守，不得忽略"）。

---

## 七、开发过程中遇到的问题与反思

### 7.1 数据集选型失误

**问题**：最初选择 `shibing624/medical`，花了大量时间处理版本冲突（新版 datasets 不支持 loading script），最终发现这个数据集根本没有科室标签。

**反思**：选数据集应该先验证字段结构，再写代码。正确流程：
1. 找候选数据集
2. `load_dataset(..., split='train[:3]')` 看字段
3. 确认有所需标签再动手

**教训**：`shibing624/medical` 的 README 里写的"含科室标签"是指数据集整体的用途描述，不代表每条记录都有标签字段。不要只看描述，要看实际数据。

### 7.2 依赖版本冲突处理

**问题**：`datasets==2.14`（支持 loading script）与主环境的 `sentence-transformers 5.x` + `torch 2.11` 形成三方版本死锁：
- `datasets 2.14` 需要 `pyarrow<14`（用了已移除的 `pa.PyExtensionType`）
- `pyarrow<14` 需要 `numpy<2`
- `torch 2.11` 需要 `numpy>=2`

**解决过程**：先尝试独立 venv，再发现 `sentence-transformers` 也需要进 venv，导致 torch 版本冲突。最终方案是把数据下载和向量化拆成两步，但后来发现数据集本身没有科室标签，整个折腾都是无用功。

**反思**：遇到依赖冲突，正确做法是先搞清楚每个依赖的用途，判断是否能拆分职责，再决定隔离方案。不要陷入"先解决冲突再看数据"的顺序陷阱。

### 7.3 异步并发死锁

**问题**：CLI 中 `consumer task` 和 `agent.handle()` 顺序执行，`consumer` 在等事件，`agent` 还没 emit，导致卡死。

**原因**：`asyncio.create_task()` 创建了 task 但没有真正并发，`await agent.handle()` 之后再 `await consumer` 是串行的。

**解决**：用 `asyncio.gather(consume(), produce())` 让两者真正并发。

**反思**：`asyncio.create_task()` 只是把协程放入事件循环等待调度，不等于立即并发执行。真正的并发需要 `gather` 或在同一个 await 链上交替执行。

### 7.4 关键词匹配 vs NER 模型

**问题**：最初用关键词列表匹配症状部位，"肚子"这种口语化表达无法匹配。

**错误思路**：不断扩充关键词列表，试图覆盖所有口语表达——这是无止境的，医疗场景的口语表达极其多样。

**正确做法**：用 NER 模型，让模型自己学会识别部位实体，不依赖穷举。代码量反而更少，鲁棒性更高。

**教训**：规则方案的维护成本会随覆盖范围线性增长，模型方案的维护成本是固定的（换更好的模型）。在语言理解类任务上，不要用规则对抗语言的多样性。

### 7.5 CLI 入口方式

**问题**：`python -m medi.cli chat` 报错，`chat` 不被识别为子命令。

**原因**：typer 在只有一个子命令时会把它当默认命令处理，用 `app` 作为入口时 `chat` 参数被当成多余参数报错。

**解决**：`pyproject.toml` 的入口改为直接指向 `chat` 函数（`medi.cli:chat`），而不是 `app`，`medi` 直接启动对话。

**反思**：Phase 1 只有一个功能，不需要子命令设计。`medi chat` 这种形式是为未来扩展预留的，过早设计反而带来问题。YAGNI（You Aren't Gonna Need It）原则。

---

## 八、Phase 2 待办

| 模块 | 内容 |
|------|------|
| HealthProfile | SQLite 持久化用户档案（年龄、性别、过敏史、慢性病、用药记录） |
| EpisodicMemory | 向量存储历史症状描述 |
| MemoryDistiller | 会话结束后异步提炼洞察 |
| MedicationAgent | 用药管理、副作用咨询、药物交互检查 |
| 全量向量索引 | build_index.py 跑全量 17.7 万条 |
| OrchestratorAgent | 意图识别，路由到 TriageAgent / MedicationAgent |
