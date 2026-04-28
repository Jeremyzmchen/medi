# Medi — 智能健康 Agent 系统设计文档

## 一、项目定位

Medi 是一个面向中文用户的智能健康 Agent 系统，核心目标：

- **学习向**：系统覆盖 Agent 开发的关键技术点（记忆、Tool Calling、Multi-Agent 协同、模型路由）
- **面试向**：对标京东健康等大厂健康 AI 岗位，可作为项目经历完整展示

**Phase 1 目标**：实现智能分诊 Agent（TriageAgent），用户描述症状后引导问询，输出科室建议和紧急程度评估。

---

## 二、整体架构

### 2.1 系统分层

```
用户层
  CLI / API（FastAPI）

Orchestrator 层
  OrchestratorAgent — 意图识别 + 子 Agent 路由

Agent 层
  TriageAgent          # Phase 1：症状分析、科室路由
  MedicationAgent      # Phase 2：用药管理、副作用咨询
  ReportAgent          # Phase 2：检验报告解读
  NutritionAgent       # Phase 3：膳食建议
  ScheduleAgent        # Phase 3：预约挂号

基础设施层
  UnifiedContext       # 共享上下文（用户档案、会话信息）
  AsyncStreamBus       # 异步事件总线（流式输出）
  ToolRuntime          # 统一工具执行层（超时、重试、审计）
  MemorySystem         # 短期 + 长期记忆
  ModelRouter          # 模型分级路由
```

### 2.2 Multi-Agent 协同机制

- **OrchestratorAgent** 只做意图识别和路由，不处理业务逻辑
- 子 Agent 之间通过 `UnifiedContext` 共享关键数据，不直接互相调用
- 复杂场景（如分诊后需要预约）由 Orchestrator 串联多个子 Agent

---

## 三、Phase 1：TriageAgent 详细设计

### 3.1 核心流程

采用 TAOR 四阶段（与 Weave 一致）：

```
Think   — 分析用户已描述的症状，判断信息是否充分
Act     — 检索症状-科室知识库，必要时调用外部工具
Observe — 评估检索结果，决定是否需要追问
Respond — 生成科室建议 + 紧急程度 + 就医指导
```

Token Budget：
```python
TRIAGE_TOKEN_BUDGET = {
    "think":   200,   # 症状完整性判断
    "act":     100,   # 检索指令生成
    "observe": 400,   # 结果评估
    "respond": 1000,  # 最终建议
}
```

### 3.2 关键模块

**症状收集器（SymptomCollector）**
- 维护当次对话的症状列表
- 判断信息是否足够：症状部位、持续时间、伴随症状、既往史
- 不足时生成追问，最多追问 3 轮（避免用户体验差）

**紧急程度评估器（UrgencyEvaluator）**
- 规则层前置（不走 LLM）：胸痛、呼吸困难、意识丧失 -> 立即拨打 120
- LLM 层：根据症状组合评估紧急程度（紧急/较急/普通/可观察）

**科室路由器（DepartmentRouter）**
- 检索症状-科室知识库，输出候选科室列表（含置信度）
- 支持多科室：同一症状可能对应内科/外科，需给出优先级排序

### 3.3 TAOR 流程示例

```
用户：我最近头很痛

Think：症状信息不足，缺少部位（额头/后脑）、持续时间、伴随症状
Act：生成追问

用户：右侧太阳穴，持续3天，有点恶心

Think：信息基本充分，检索知识库
Act：query("偏头痛 恶心 太阳穴")
Observe：匹配度高，神经内科 0.85 / 头颈外科 0.3
         紧急程度规则：无红旗症状（无发热、意识清醒）

Respond：
  建议科室：神经内科（优先）
  紧急程度：普通（建议近期就诊）
  注意事项：如出现发热、视力模糊请立即就医
```

---

## 四、数据方案

### 4.1 症状-科室知识库

**数据来源**（开源，可直接使用）：
- `shibing624/medical`（Hugging Face）— 195 万条医疗对话，含科室标签
- `LCMDC` — 43 万条粗粒度分诊样本，14 个科室分类，199K 条细粒度诊断
- `cMedQA2` — 中文医疗问答，含症状-回答对

**使用策略**：
- 不做模型微调（成本高），用数据构建向量知识库
- 用 `shibing624/medical` + `LCMDC` 构建症状-科室向量索引
- 向量模型：`BAAI/bge-large-zh-v1.5`（中文 embedding 效果最好）

### 4.2 向量数据库选型

| 方案 | 优点 | 缺点 |
|------|------|------|
| ChromaDB | 轻量，本地无需部署 | 生产性能有限 |
| Qdrant | 高性能，支持过滤 | 需要单独启动服务 |
| FAISS | 纯本地，极快 | 无持久化 |

**Phase 1 选择**：ChromaDB（本地开发简单，不依赖外部服务）

### 4.3 药品数据（Phase 2 备用）

- NMPA 数据库（国家药品监督管理局）：官方 API，药品注册信息
- YaozKnowledge（药智数据）：药品说明书、适应症
- Phase 1 暂时 mock，Phase 2 接入

---

## 五、记忆系统

### 5.1 分层设计

```
短期记忆（当次会话）
  存储位置：LLM context window
  内容：完整问询对话、症状收集过程
  生命周期：会话结束即丢弃

长期记忆（跨会话）
  医疗事实（结构化）-> SQLite
    - 用户基本档案：年龄、性别、慢性病史、过敏史
    - 用药记录：药品名、用量、周期
    - 就诊记录：时间、科室、结论
  行为习惯（语义化）-> ChromaDB（向量）
    - 历史症状描述
    - 问询偏好（喜欢详细解释 or 简短结论）

记忆蒸馏（MemoryDistiller）
  会话结束后异步执行
  提炼关键事件：本次症状 + 建议科室 + 紧急程度
  存洞察不存原文
```

### 5.2 用户健康档案（HealthProfile）

```python
class HealthProfile:
    user_id: str
    age: int
    gender: str
    chronic_conditions: list[str]   # 慢性病：高血压、糖尿病等
    allergies: list[str]            # 过敏史
    current_medications: list[str]  # 当前用药
    visit_history: list[VisitRecord]  # 就诊记录摘要
```

这个档案会作为每次对话的前缀上下文注入，让 Agent 的建议更个性化。

---

## 六、工具层（Tool Calling）

### 6.1 工具分级

```python
class ToolPriority(Enum):
    CRITICAL = "critical"   # 药品信息查询 — 失败必须告知用户
    STANDARD = "standard"   # 科室信息检索 — 可重试 3 次
    OPTIONAL = "optional"   # 健康资讯推送 — 失败静默跳过
```

### 6.2 Phase 1 工具清单

| 工具名 | 优先级 | 描述 |
|--------|--------|------|
| `search_symptom_kb` | STANDARD | 检索症状-科室知识库 |
| `evaluate_urgency` | CRITICAL | 紧急程度规则评估 |
| `get_department_info` | OPTIONAL | 获取科室介绍 |
| `get_user_profile` | STANDARD | 读取用户健康档案 |

### 6.3 ToolRuntime 设计

继承 Weave 的 ToolRuntime 模式，增加健康场景特有的：
- **审计日志**：所有 CRITICAL 工具调用必须记录（时间、入参、出参）
- **超时分级**：CRITICAL 工具超时 10s，STANDARD 5s，OPTIONAL 3s
- **降级策略**：OPTIONAL 工具失败不中断主流程

---

## 七、模型路由

```python
MODEL_ROUTING = {
    "intent_classification": "claude-haiku-4-5",   # 意图识别，高频低延迟
    "symptom_analysis":      "claude-sonnet-4-6",  # 症状分析，准确性优先
    "urgency_evaluation":    "rule_based",          # 紧急评估，规则层优先
    "response_generation":   "claude-sonnet-4-6",  # 最终建议生成
    "memory_distill":        "claude-haiku-4-5",   # 记忆蒸馏，成本敏感
}
```

紧急程度评估走规则层，不依赖 LLM，避免误判风险。

---

## 八、目录结构

```
Medi/
├── DESIGN.md               # 本文档
├── medi/
│   ├── core/
│   │   ├── context.py      # UnifiedContext（用户档案 + 会话信息）
│   │   ├── stream_bus.py   # AsyncStreamBus
│   │   └── tool_runtime.py # ToolRuntime（含分级、审计）
│   ├── agents/
│   │   ├── orchestrator.py # OrchestratorAgent
│   │   └── triage/
│   │       ├── agent.py         # TriageAgent 主逻辑
│   │       ├── symptom_collector.py  # 症状收集 + 追问
│   │       ├── urgency_evaluator.py  # 紧急程度评估
│   │       └── department_router.py  # 科室路由
│   ├── memory/
│   │   ├── health_profile.py   # 用户健康档案（SQLite）
│   │   ├── episodic.py         # 历史记忆（向量）
│   │   └── distiller.py        # 记忆蒸馏
│   ├── tools/
│   │   ├── symptom_kb.py       # 症状知识库检索
│   │   ├── urgency_rules.py    # 紧急程度规则
│   │   └── department_info.py  # 科室信息
│   ├── knowledge/
│   │   └── build_index.py      # 构建向量知识库（离线脚本）
│   └── api/
│       └── main.py             # FastAPI 入口
├── data/
│   └── raw/                    # 原始数据集（不入 git）
├── tests/
└── pyproject.toml
```

---

## 九、开发路线

### Phase 1（当前）：TriageAgent
- [ ] 搭建项目骨架（core 层复用 Weave 设计）
- [ ] 下载并处理 `shibing624/medical` 数据集
- [ ] 构建症状-科室向量知识库（ChromaDB + bge-large-zh）
- [ ] 实现 SymptomCollector（追问逻辑）
- [ ] 实现 UrgencyEvaluator（规则层）
- [ ] 实现 DepartmentRouter（向量检索）
- [ ] 实现 TriageAgent TAOR 主流程
- [ ] CLI 接口可用

### Phase 2：用药管理 + 记忆系统
- [ ] 接入 NMPA 药品数据
- [ ] 实现 MedicationAgent
- [ ] 实现长期记忆（SQLite + 向量双存储）
- [ ] 实现 MemoryDistiller

### Phase 3：Multi-Agent 完整系统
- [ ] 实现 OrchestratorAgent（意图识别 + 路由）
- [ ] 接入 ReportAgent、NutritionAgent
- [ ] FastAPI 接口
- [ ] 完整的多 Agent 协同场景演示

---

## 十、与 Weave 的关系

### 10.1 共享的架构理念

Medi 和 Weave 共享相同的基础设施设计：

| 维度 | Weave | Medi |
|------|-------|------|
| 场景 | 行研报告生成 | 健康咨询与分诊 |
| Agent 模式 | 单 Agent + LangGraph | Multi-Agent + Orchestrator |
| 记忆重点 | 领域洞察蒸馏 | 用户健康档案 |
| 工具重点 | 搜索、爬虫 | 知识库检索、医疗 API |
| 安全约束 | 低 | 高（医疗风险，规则兜底） |

core 层（UnifiedContext、AsyncStreamBus、ToolRuntime）代码可直接参考 Weave 实现，按 Medi 需求裁剪。

### 10.2 Medi 的差异化设计

以下四点是 Medi 独有的，有明确的必要性理由，不是为差异化而差异化。

---

**差异一：Safety-First 架构（规则层 + LLM 双保险）**

Weave 是纯 LLM 驱动的，行研答错了最多报告质量差。医疗场景中 LLM 幻觉可能造成真实伤害，因此必须有规则层前置。

```
用户输入
    |
规则层扫描（红旗症状检测）  <-- Weave 没有这一层
    |               |
普通症状        胸痛/呼吸困难/意识丧失
    |               |
LLM 分析      直接输出"请立即拨打120"，不等 LLM
```

设计原则：**紧急情况的判断权不能交给 LLM**，规则层是最后一道安全网。

---

**差异二：显式对话状态机**

Weave 的议题处理是线性的（一个议题跑完再跑下一个）。分诊是多轮引导对话，状态流转复杂，需要显式的状态机管理：

```
INIT
  |
  v
COLLECTING（追问中，信息不足）
  |         |
  |    达到3轮追问上限，强制进入下一步
  |
SUFFICIENT（信息足够）
  |
  v
SEARCHING（检索知识库）
  |         |
  |    发现红旗症状
  |         |
  v         v
RESPONDING  ESCALATING（立即升级，跳过检索）
```

状态机让流程可测试、可调试，而不是靠 LLM 自己决定"下一步做什么"。

---

**差异三：健康档案作为硬约束注入**

Weave 的记忆是"洞察"，是软性参考，LLM 可以选择忽略。

Medi 的健康档案是硬性事实，必须作为强约束注入每次 LLM 调用的 system prompt：

```python
# Weave 的记忆注入方式（软约束，供参考）
system_prompt += f"\n历史洞察：{memory_insights}"

# Medi 的档案注入方式（硬约束，必须遵守）
system_prompt += f"""
[用户健康约束 - 必须严格遵守]
- 过敏史：{profile.allergies}  # 涉及药物时必须过滤
- 慢性病：{profile.chronic_conditions}  # 影响科室优先级判断
- 当前用药：{profile.current_medications}  # 避免药物冲突建议
"""
```

"参考"和"约束"在 prompt 工程上是完全不同的设计语义。

---

**差异四：工具调用审计链路**

Weave 的工具调用只需正确执行，失败了重试即可。

医疗工具调用需要可追溯，这是合规要求：

```python
@dataclass
class AuditRecord:
    session_id: str
    timestamp: datetime
    tool_name: str
    priority: ToolPriority
    input_params: dict
    output_result: dict
    latency_ms: int
    success: bool
    error_msg: str | None

# CRITICAL 级别工具调用自动写入审计表，不可关闭
# 审计记录用于事后溯源：哪次问诊查了什么药、返回了什么结果
```

这一层在 Weave 中完全不存在，是 Medi 独有的基础设施需求。

---

### 10.3 差异总结

| 差异点 | Weave | Medi | 必要性 |
|--------|-------|------|--------|
| 安全层 | 无 | 规则层前置扫描 | LLM 不可信赖于紧急判断 |
| 对话管理 | 线性流程 | 显式状态机 | 多轮引导的复杂状态流转 |
| 记忆语义 | 软约束（洞察） | 硬约束（档案） | 过敏史不能只"参考" |
| 工具审计 | 无 | 完整审计链路 | 医疗合规可追溯要求 |
