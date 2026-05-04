# Medi 数据结构设计记录

本文档用于记录 Medi 项目数据模型的重新设计过程。目标不是一次性重构全部代码，而是先把核心概念、生命周期、数据关系和后续修改方案逐步确定下来，再按确认后的方案执行代码调整。

## 1. HealthProfile

### 1.1 定位

`HealthProfile` 是用户级、长期存在的健康档案。

它描述的是一个用户跨多次咨询都可能长期有效的背景信息，而不是某一次预诊会话中的临时症状记录。

它应该回答的问题是：

- 这个用户是谁？
- 这个用户有哪些长期健康背景？
- 这些背景信息会如何影响本次分诊、用药咨询或健康报告分析？

它不应该回答的问题是：

- 这一次用户哪里不舒服？
- 这一次症状从什么时候开始？
- 这一次分诊问到了哪些事实？
- 这一次最终推荐了什么科室？

这些本次会话内产生的信息应该属于 `Encounter`、`ClinicalFacts`、`PreconsultationRecord`、`ClinicalAssessment` 或最终输出，而不是直接放进 `HealthProfile`。

### 1.2 生命周期

`HealthProfile` 的生命周期是跨 session 的。

```text
Patient/User
  -> HealthProfile
  -> Encounter 1
  -> Encounter 2
  -> Encounter 3
```

一个用户通常只有一份长期 `HealthProfile`，但可以有多次 `Encounter`。

例如：

- 第一次：用户咨询头痛分诊
- 第二次：用户咨询腹痛分诊
- 第三次：用户上传体检报告
- 第四次：用户咨询某种药物能不能吃

这些都是不同的本次会话或本次医疗事件，但它们都可以引用同一个长期用户档案。

### 1.3 主要字段

当前 `HealthProfile` 可以保留为长期档案模型，主要包含：

- `user_id`
- `age`
- `gender`
- `chronic_conditions`
- `allergies`
- `current_medications`
- `visit_history`

这些字段大致可以分为两类。

第一类是用户基础信息：

- 年龄
- 性别

第二类是长期医疗背景：

- 慢性病史
- 长期用药
- 过敏史
- 历史就诊记录

这些信息对分诊有重要影响。例如年龄、慢病、长期用药和过敏史都可能影响紧急程度判断、风险提示和用药安全。

### 1.4 设计原则

`HealthProfile` 不作为 LangGraph 内部的主状态流转。

也就是说，它不应该在每个节点之间被不断修改、传递、合并。它更像一张长期持久化主表，而不是一次问诊流程里的工作草稿。

建议原则如下：

- 开始一次预诊时读取 `HealthProfile`
- 预诊过程中把它作为只读背景参考
- 不在普通 intake 节点中直接修改 `HealthProfile`
- 本次预诊结束后，由专门的更新逻辑选择性回写
- 不把本次短期症状直接写入长期档案

### 1.5 推荐数据流

开始预诊时：

```text
user_id
  -> load_profile(user_id)
  -> HealthProfile
  -> ProfileSnapshot
  -> 当前 Encounter 只读参考
```

预诊过程中：

```text
用户输入
  -> Messages
  -> ClinicalFacts
  -> PreconsultationRecord
  -> TaskBoard
  -> ClinicalAssessment
  -> Output
```

预诊结束后：

```text
PreconsultationRecord / Output
  -> ProfileUpdateCandidate
  -> 人工规则或确认逻辑
  -> save_profile(profile)
```

### 1.6 ProfileSnapshot

建议后续新增 `ProfileSnapshot` 概念。

`HealthProfile` 是数据库中的长期模型，负责持久化。

`ProfileSnapshot` 是一次预诊开始时从 `HealthProfile` 读取出来的只读快照，进入当前运行时上下文。

两者区别如下：

| 概念 | 生命周期 | 是否持久化 | 是否在图中修改 | 作用 |
| --- | --- | --- | --- | --- |
| `HealthProfile` | 跨 session | 是 | 否 | 用户长期健康档案 |
| `ProfileSnapshot` | 单次 Encounter | 否，或随 Encounter 保存快照 | 否 | 本次预诊的只读背景 |

`ProfileSnapshot` 可以被注入到分诊 prompt、风险评估、输出生成中，但节点不应该直接改它。

### 1.7 与本次预诊数据的关系

`HealthProfile` 与本次预诊数据不是父子包含关系，而是引用关系。

推荐关系：

```text
Patient/User
  -> HealthProfile
  -> Encounter[]
       -> Messages
       -> ClinicalFacts
       -> PreconsultationRecord
       -> TaskBoard
       -> ClinicalAssessment
       -> Output
```

也就是说，`HealthProfile` 不是 `preconsultation_record` 的父字段，也不是本次症状摘要的父字段。

更准确地说：

- `HealthProfile` 提供长期背景
- `Encounter` 表示一次医疗咨询事件
- `ClinicalFacts` 记录本次对话中抽取到的事实
- `PreconsultationRecord` 把本次 facts 投影成 T1/T2/T3/T4 结构
- `ClinicalAssessment` 结合本次记录和长期背景做科室、风险、紧急程度判断
- `TriageOutput` 生成患者端建议和医生端预诊报告

### 1.8 什么时候读取 HealthProfile

建议在创建一次运行时对象时读取，而不是在每个节点中反复读取。

当前可以理解为：

```text
handle_turn(user_input)
  -> get_or_create_session(session_id)
  -> load_profile(user_id)
  -> 构造 ctx / graph runner / agents
```

生产级优化后可以调整为：

```text
request(session_id, user_input)
  -> 加载 SessionSnapshot
  -> 加载 HealthProfile
  -> 构造 Runtime
  -> 执行本轮
  -> 保存 SessionSnapshot
```

在图内部，节点应该拿到的是 `ProfileSnapshot` 或由 `UnifiedContext.build_constraint_prompt()` 生成的约束文本，而不是直接操作数据库中的 `HealthProfile`。

### 1.9 什么时候更新 HealthProfile

建议只在预诊完成后更新，且只更新长期、稳定、已确认的信息。

当前代码状态：

```text
已经实现：
  分诊结束后写入 visit_records 历史就诊记录

尚未实现：
  从本次预诊中提取过敏史、慢性病、长期用药等信息并回写 profiles 表
```

也就是说，当前如果用户在预诊中说“我青霉素过敏”，这条信息可以进入本次 `clinical_facts`、`preconsultation_record` 和医生报告，但不会自动写入：

```python
HealthProfile.allergies
```

可以回写：

- 用户明确确认的年龄
- 用户明确确认的性别
- 用户明确确认的慢性病史
- 用户明确确认的长期用药
- 用户明确确认的药物或食物过敏
- 本次分诊形成的一条 `visit_record`

不建议回写：

- 本次短期症状，例如“今天头痛两小时”
- 本次症状严重程度，例如“疼痛 6 分”
- 本次伴随症状，例如“这次有恶心”
- 本次病情进展，例如“今天下午加重”
- 临时处理经过，例如“刚吃了一片布洛芬”

这些应属于本次 `Encounter` 的病历记录，而不是长期健康档案。

### 1.10 更新策略

后续计划设计一个独立的 `ProfileUpdater` 或 `ProfileUpdateService`。

它的职责是：

1. 从本次 `PreconsultationRecord` 中识别可能属于长期档案的信息。
2. 生成 `ProfileUpdateCandidate`，而不是直接写入 `HealthProfile`。
3. 判断这些信息是否足够明确、稳定、可回写。
4. 必要时要求用户确认。
5. 将确认后的信息写入 `HealthProfile`。

示例：

```text
用户说：“我有高血压，长期吃氨氯地平。”

本次 Facts:
  ph.disease_history = 高血压
  safety.current_medications = 氨氯地平

预诊结束后:
  ProfileUpdater 识别为长期信息并生成候选项
  -> ProfileUpdateCandidate(field="chronic_conditions", value="高血压")
  -> ProfileUpdateCandidate(field="current_medications", value="氨氯地平")
  -> 用户确认或规则校验通过
  -> chronic_conditions += 高血压
  -> current_medications += 氨氯地平
```

但如果用户说：

```text
我今天头痛后吃了一片布洛芬。
```

这更像本次治疗经过，不应自动写入长期 `current_medications`。

### 1.11 与 T1/T2/T3/T4 的关系

`HealthProfile` 不属于 T1/T2/T3/T4 中的任何一个任务组。

T1/T2/T3/T4 是本次预诊的结构。

`HealthProfile` 是这些任务的背景参考。

关系可以理解为：

```text
HealthProfile
  -> 影响 T1 分诊风险判断
  -> 补充 T3 既往史与安全信息的参考
  -> 约束 Output 生成
```

具体影响：

- T1：年龄、慢病、免疫抑制、妊娠等可能影响紧急程度
- T2：通常不直接填充现病史，因为现病史必须来自本次症状
- T3：可作为既往史、用药史、过敏史的候选参考，但最好由用户确认
- T4：通常不直接参与主诉生成，除非长期疾病与本次主诉高度相关

### 1.12 当前确认的设计决策

目前确认以下原则：

1. `HealthProfile` 是用户长期档案，生命周期跨多次预诊。
2. `HealthProfile` 不在 LangGraph 内部作为可变状态流转。
3. 一次预诊开始时读取 `HealthProfile`，生成只读 `ProfileSnapshot`。
4. 预诊过程中，`ProfileSnapshot` 只作为背景、约束和风险参考。
5. 本次预诊产生的数据进入 `Encounter`、`ClinicalFacts`、`PreconsultationRecord`、`ClinicalAssessment` 和 `Output`。
6. 预诊结束后，由独立更新逻辑选择性回写长期档案。
7. 不把本次短期症状和临时病情变化直接写入长期 `HealthProfile`。

## 2. ProfileSnapshot

### 2.1 定位

`ProfileSnapshot` 是一次医疗咨询开始时，从长期 `HealthProfile` 截取出来的只读背景副本。

它位于长期用户档案和本次 `Encounter` 之间，作用是把“本次咨询开始时系统已知的用户背景”固定下来。

可以理解为：

```text
HealthProfile  = 数据库里的长期用户档案
ProfileSnapshot = 本次 Encounter 创建时截取的只读档案快照
```

`ProfileSnapshot` 不等于 `HealthProfile` 本身。

`HealthProfile` 会随着用户长期信息更新而变化；`ProfileSnapshot` 一旦进入本次 `Encounter`，就应该保持不变。

### 2.2 为什么需要 ProfileSnapshot

直接在图里使用 `HealthProfile` 会带来几个问题。

第一，语义不清。

如果每个节点都拿到 `HealthProfile`，代码阅读者会误以为节点可以直接修改长期档案。这会让长期数据和本次运行时状态混在一起。

第二，历史不可复现。

一次预诊判断应该基于当时系统知道的信息。如果后来用户补充了新的慢病或用药，不能反过来影响历史预诊记录的解释。

例如：

```text
5 月 1 日：用户咨询头痛，当时档案中没有高血压记录
5 月 10 日：用户补录高血压
```

回看 5 月 1 日那次预诊时，系统应该能够知道当时并没有使用“高血压”这个背景信息。

第三，生产级存储更清晰。

每次 `Encounter` 应该保存当时使用过的用户背景快照，而不是永远引用一个会变化的当前 `HealthProfile`。

### 2.3 生命周期

`ProfileSnapshot` 的生命周期属于单次 `Encounter`。

```text
load_profile(user_id)
  -> HealthProfile
  -> build_profile_snapshot(profile)
  -> ProfileSnapshot
  -> Encounter
```

它在一次咨询开始时生成，在本次咨询过程中保持只读。

如果未来做生产级持久化，`ProfileSnapshot` 可以随 `Encounter` 一起保存，作为本次咨询使用过的背景证据。

### 2.4 推荐字段

建议字段如下：

```python
class ProfileSnapshot(TypedDict, total=False):
    user_id: str
    age: int | None
    gender: str | None
    chronic_conditions: list[str]
    allergies: list[str]
    current_medications: list[str]
    recent_visits: list[VisitSnapshot]
    source_updated_at: str | None
    captured_at: str
```

历史就诊记录可以简化为：

```python
class VisitSnapshot(TypedDict, total=False):
    visit_date: str
    department: str
    chief_complaint: str
    conclusion: str
```

字段说明：

- `user_id`：用户 ID
- `age`：本次咨询开始时已知年龄
- `gender`：本次咨询开始时已知性别
- `chronic_conditions`：本次咨询开始时已知慢性病或长期疾病背景
- `allergies`：本次咨询开始时已知过敏史
- `current_medications`：本次咨询开始时已知长期或当前用药背景
- `recent_visits`：最近若干次历史咨询记录，只作参考
- `source_updated_at`：长期档案最后更新时间
- `captured_at`：本次快照生成时间

### 2.5 与 HealthProfile 的关系

`HealthProfile` 是长期主档案。

`ProfileSnapshot` 是从 `HealthProfile` 派生出来的只读副本。

关系如下：

```text
HealthProfile
  -> ProfileSnapshot
       -> Encounter
```

`ProfileSnapshot` 不回写 `HealthProfile`。

如果本次咨询产生了可以更新长期档案的信息，应该通过独立的 `ProfileUpdater` 或 `ProfileUpdateService` 完成，而不是让 `ProfileSnapshot` 自己承担更新职责。

### 2.6 与 Encounter 的关系

`ProfileSnapshot` 属于本次 `Encounter` 的背景部分。

推荐关系：

```text
Encounter
  -> profile_snapshot
  -> messages
  -> facts
  -> preconsult_record
  -> task_board
  -> assessment
  -> output
```

也就是说，`ProfileSnapshot` 可以进入 graph state，但它是只读参考，不是本次事实源。

### 2.7 与 ClinicalFacts 的关系

`ProfileSnapshot` 不承担事实采集职责。

如果用户在本次对话中说：

```text
我有高血压，长期吃氨氯地平。
```

这些信息首先应该进入本次 `ClinicalFacts`：

```text
ph.disease_history = 高血压
safety.current_medications = 氨氯地平
```

而不是直接修改 `ProfileSnapshot`。

原因是：本次用户新说出来的信息，是本次 `Encounter` 的证据。它是否应该成为长期档案，需要在预诊结束后由更新逻辑判断。

### 2.8 与 T1/T2/T3/T4 的关系

`ProfileSnapshot` 不属于 T1/T2/T3/T4 中的任何一个任务组。

它是本次预诊任务的背景参考。

具体影响如下：

- T1：影响分诊风险和紧急程度判断。例如年龄、慢病、免疫抑制、妊娠等。
- T2：通常不直接填充现病史。现病史应来自本次症状和本次对话。
- T3：作为既往史、长期用药、过敏史的参考，但最好由用户确认。
- T4：通常不直接生成主诉，只在长期疾病与本次主诉高度相关时作为背景。

### 2.9 与 Prompt 和风险判断的关系

`ProfileSnapshot` 可以用于构造 prompt 约束。

例如：

```text
[用户长期背景]
年龄：68 岁
性别：男
慢性病：高血压、糖尿病
长期用药：二甲双胍、氨氯地平
过敏史：青霉素
```

这些信息可以影响：

- ClinicalNode 的科室判断
- UrgencyEvaluator 的紧急程度判断
- OutputNode 的安全提醒
- MedicationAgent 的用药风险判断
- HealthReportAgent 的报告解读建议

但 prompt 中应明确说明：

```text
这些是用户长期档案中的背景信息，不等同于本次症状。
如与本次对话冲突，以本次用户明确陈述为准，并避免编造未确认信息。
```

### 2.10 是否需要用户确认

`ProfileSnapshot` 里的信息可以作为参考，但不应该在所有场景里直接当成本次已确认事实。

例如档案显示：

```text
过敏史：青霉素
```

系统可以在用药安全场景里引用：

```text
档案显示您有青霉素过敏史。除了青霉素，您还有其他药物或食物过敏吗？
```

这比直接假设“本次已经完成过敏史采集”更稳妥。

也就是说：

```text
ProfileSnapshot = 背景参考
ClinicalFacts = 本次确认事实
```

### 2.11 Graph State 中的使用方式

当前代码先不把 `ProfileSnapshot` 放入 `TriageGraphState`。

原因是现在的 `TriageGraphState` 仍然承载分诊图内部的本次工作状态，而 `ProfileSnapshot` 更像本次运行时的外部只读依赖。

当前使用方式是：

```python
class UnifiedContext:
    profile_snapshot: ProfileSnapshot
```

然后在构建分诊图时，把它作为只读依赖绑定给节点：

```python
build_triage_graph(
    ...,
    profile_snapshot=ctx.profile_snapshot,
)
```

约束是：

- 节点可以读取 `profile_snapshot`
- 节点不应该修改 `profile_snapshot`
- 节点不能把 `profile_snapshot` 当成本次事实源
- 如果需要把其中某个信息纳入本次病历，应通过提问确认或明确标记来源为 profile

未来做 `EncounterState` 持久化时，可以把 `profile_snapshot.to_dict()` 保存到本次 `Encounter` 中，用于历史复现和审计。

### 2.12 当前确认的设计决策

目前确认以下原则：

1. `ProfileSnapshot` 是从长期 `HealthProfile` 截取的只读快照。
2. `ProfileSnapshot` 生命周期属于单次 `Encounter`。
3. `ProfileSnapshot` 可以进入 graph state，但不被节点修改。
4. `ProfileSnapshot` 不是本次事实源，不能替代 `ClinicalFacts`。
5. `ProfileSnapshot` 可以用于 T1 风险判断、T3 背景参考和最终输出约束。
6. `ProfileSnapshot` 应随未来的 `Encounter` 一起保存，保证历史咨询可复现。
7. 本次新获得的长期信息应先进入 `ClinicalFacts`，预诊结束后再由独立更新逻辑选择性回写 `HealthProfile`。

### 2.13 当前代码落点

当前代码已经完成运行时边界清理：

```text
HealthProfile
  只属于长期档案持久化层

ProfileSnapshot
  只属于本次运行时上下文和本次分诊图依赖
```

新增代码：

- `medi/memory/profile_snapshot.py`
  - 定义 `ProfileSnapshot`
  - 定义 `VisitSnapshot`
  - 提供 `build_profile_snapshot(profile)` 从长期档案生成只读快照

接入方式：

- API session 创建时：
  - `load_profile(user_id)`
  - `build_profile_snapshot(profile)`
  - 只把 `profile_snapshot` 放入 `UnifiedContext`
- CLI 会话创建时执行同样逻辑
- `UnifiedContext.build_constraint_prompt()` 只读取 `profile_snapshot`
- `TriageGraphRunner` 构图时只把 `profile_snapshot` 作为只读背景传给分诊图
- Graph builder 和各节点参数统一命名为 `profile_snapshot`
- `resolve_intake_plan()`、`evaluate_risk_factors()`、`OutputNode` 均读取 `profile_snapshot`

当前明确不做的事：

- 不在 `UnifiedContext` 中保留 `health_profile`
- 不把长期可变档案对象传入 LangGraph 节点
- 不让节点直接更新长期用户档案

后续阶段再考虑：

- 将 `profile_snapshot.to_dict()` 放入未来的 `EncounterState`
- 将长期档案回写逻辑收敛到独立的 `ProfileUpdater`

## 3. dataclass 在当前数据模型中的使用

### 3.1 dataclass 是什么

`dataclass` 是 Python 标准库提供的数据类工具，适合定义“主要用于承载数据”的对象。

它可以自动生成：

- `__init__`
- `__repr__`
- `__eq__`
- 字段默认值处理

例如：

```python
@dataclass
class HealthProfile:
    user_id: str
    age: int | None = None
    chronic_conditions: list[str] = field(default_factory=list)
```

这等价于手写一个初始化方法，但更短、更清楚。

### 3.2 为什么 HealthProfile 适合 dataclass

`HealthProfile` 是长期档案模型，主要职责是承载数据库读写后的结构化数据。

它包含：

- `user_id`
- `age`
- `gender`
- `chronic_conditions`
- `allergies`
- `current_medications`
- `visit_history`
- `updated_at`

这些字段本身没有复杂行为，使用 `dataclass` 可以让它保持轻量。

需要注意的是，`HealthProfile` 是长期可变对象，所以它没有设置 `frozen=True`。

例如 CLI 首次收集档案时会修改：

```python
profile.age = age
profile.chronic_conditions = [...]
await save_profile(profile)
```

这符合长期档案模型的定位。

### 3.3 为什么 ProfileSnapshot 使用 frozen dataclass

`ProfileSnapshot` 的设计目标是“本次 Encounter 开始时的只读快照”。

所以它使用：

```python
@dataclass(frozen=True)
class ProfileSnapshot:
    ...
```

`frozen=True` 的含义是：对象创建后不允许再修改字段。

这样可以在代码层面表达设计意图：

```text
ProfileSnapshot 一旦创建，就代表本次咨询开始时系统已知的长期背景。
后续节点只能读取，不能修改。
```

### 3.4 为什么 ProfileSnapshot 使用 tuple 而不是 list

`HealthProfile` 中的列表字段是可变的：

```python
chronic_conditions: list[str]
allergies: list[str]
current_medications: list[str]
```

但 `ProfileSnapshot` 中使用 tuple：

```python
chronic_conditions: tuple[str, ...]
allergies: tuple[str, ...]
current_medications: tuple[str, ...]
```

原因是 `frozen=True` 只能防止字段被重新赋值，不能阻止字段内部的 list 被修改。

如果 frozen dataclass 里放的是 list，仍然可能出现：

```python
snapshot.chronic_conditions.append("高血压")
```

为了让快照真正接近只读，快照内部使用 tuple。

### 3.5 default_factory 的作用

不要这样写可变默认值：

```python
chronic_conditions: list[str] = []
```

因为这个 list 会被所有实例共享，容易导致数据串到别的用户身上。

正确写法是：

```python
chronic_conditions: list[str] = field(default_factory=list)
```

这样每次创建对象都会得到一个新的 list。

同理，tuple 虽然不可变，也可以用：

```python
recent_visits: tuple[VisitSnapshot, ...] = field(default_factory=tuple)
```

### 3.6 dataclass 与 TypedDict 的边界

当前项目里同时有 `dataclass` 和 `TypedDict`。

推荐边界：

- `dataclass`：适合 Python 运行时对象，例如 `HealthProfile`、`ProfileSnapshot`
- `TypedDict`：适合 LangGraph state、JSON-like dict、LLM 结构化输出

原因是 LangGraph checkpoint、API 响应、LLM JSON 输出更天然地使用 dict。

而长期档案、快照、服务层对象更适合 dataclass，因为它们有明确字段和少量行为方法。

### 3.7 当前决策

当前确认：

1. `HealthProfile` 继续使用普通 dataclass，因为它是长期可变档案模型。
2. `ProfileSnapshot` 使用 `frozen=True` dataclass，因为它是本次会话只读快照。
3. `ProfileSnapshot` 内部列表型字段使用 tuple，避免创建后被节点修改。
4. `ProfileSnapshot.to_dict()` 用于未来保存到 `Encounter` 或返回 JSON。
5. `TriageGraphState` 继续使用 `TypedDict`，因为它是 LangGraph 的 JSON-like 状态容器。

### 3.8 UnifiedContext 为什么也是 dataclass

`UnifiedContext` 也是 `dataclass`，但它和 `HealthProfile`、`ProfileSnapshot` 的角色不同。

`UnifiedContext` 是一次会话运行时的共享上下文，里面放的是 Agent 执行时需要共同访问的运行时资源和轻量状态。

当前它包含：

- `user_id`
- `session_id`
- `dialogue_state`
- `profile_snapshot`
- `model_config`
- `enabled_tools`
- `observability`
- `messages`

它适合用 dataclass 的原因是：

- 字段明确
- 初始化频繁
- 需要默认值
- 没有复杂继承关系
- 不需要数据库 ORM 能力

但 `UnifiedContext` 不应该成为“什么都往里塞”的全局对象。

设计边界是：

```text
UnifiedContext
  放本次运行时共享状态和依赖

TriageGraphState
  放 LangGraph 节点之间流转的本次分诊状态

HealthProfile
  放长期用户档案，不进入 UnifiedContext

ProfileSnapshot
  放本次会话开始时截取的长期背景，只读进入 UnifiedContext
```

所以这次清理后，`UnifiedContext` 不再持有 `HealthProfile`。

这是一个重要边界：

```text
长期可变数据不进入运行时共享上下文；
运行时上下文只拿本次固定下来的只读快照。
```

## 4. Encounter

### 4.1 定位

`Encounter` 表示一次医疗咨询事件。

它不是技术会话，也不是用户长期档案，而是一次具体医疗问题的业务容器。

可以理解为：

```text
Session = 技术会话
Encounter = 医疗事件
```

例如用户打开网页开始聊天，这是一个 `Session`。

在同一个 `Session` 中，用户可能先说：

```text
我头痛两天了
```

这会形成一个分诊类 `Encounter`。

后面用户又说：

```text
我这个体检报告血糖偏高是什么意思？
```

这应该形成另一个健康报告类 `Encounter`。

所以 `Encounter` 是比 `Session` 更贴近医疗业务的数据单位。

### 4.2 为什么需要 Encounter

如果只有 `session_id`，很多数据会混在一起。

例如：

```text
同一个 session:
  第 1 轮：头痛分诊
  第 2 轮：继续补充分诊信息
  第 3 轮：问药物副作用
  第 4 轮：上传体检报告
```

这些内容都属于同一个技术会话，但不应该属于同一个医疗事件。

分诊事实、健康报告指标、用药咨询结果，它们的数据结构完全不同。

`Encounter` 的作用就是把它们分开：

```text
Session
  -> Encounter(triage)
  -> Encounter(medication)
  -> Encounter(health_report)
```

这样后续做持久化、审计、复盘、长期档案回写时，边界才清楚。

### 4.3 与 Session 的区别

| 概念 | 关注点 | 生命周期 | 示例 |
| --- | --- | --- | --- |
| `Session` | 技术连接和对话上下文 | 用户一次聊天过程 | 浏览器聊天窗口、CLI 一次运行 |
| `Encounter` | 一次医疗咨询事件 | 一个具体医疗问题从开始到结束 | 头痛分诊、用药咨询、体检报告解读 |

`Session` 可以包含多个 `Encounter`。

一个 `Encounter` 通常属于一个 `Session`。

未来如果做跨设备续聊，一个 `Encounter` 甚至可以跨多个技术 session 恢复，但初期不需要做这么复杂。

### 4.4 与 HealthProfile / ProfileSnapshot 的关系

`HealthProfile` 不属于 `Encounter`。

它是用户长期档案，生命周期跨多次医疗咨询。

`ProfileSnapshot` 属于 `Encounter`。

它表示这个 `Encounter` 创建时，系统已知的长期健康背景。

推荐关系：

```text
Patient/User
  -> HealthProfile
  -> Encounter[]
       -> ProfileSnapshot
```

这意味着：

- `HealthProfile` 是长期、可变、可回写的
- `ProfileSnapshot` 是单次 Encounter 的只读背景
- 每个 Encounter 都可以保存自己的 ProfileSnapshot
- 后续 HealthProfile 变化，不影响历史 Encounter 的解释

### 4.5 Encounter 下应该挂什么

一个分诊类 `Encounter` 未来应该包含：

```text
Encounter
  -> profile_snapshot
  -> messages
  -> clinical_facts
  -> preconsultation_record
  -> task_board
  -> clinical_assessment
  -> output
```

各部分含义：

- `profile_snapshot`：本次医疗事件开始时的只读长期背景
- `messages`：本次医疗事件相关的对话消息
- `clinical_facts`：从本次对话抽取出的事实源
- `preconsultation_record`：按 T1/T2/T3/T4 投影出的本次预诊病历
- `task_board`：T1/T2/T3/T4 的任务进度
- `clinical_assessment`：科室、紧急程度、风险因子、鉴别诊断
- `triage_output`：患者端建议和医生端预诊报告

不同类型的 Encounter 可以有不同子结构。

例如：

```text
TriageEncounter
  -> clinical_facts
  -> preconsultation_record
  -> task_board
  -> clinical_assessment
  -> triage_output

MedicationEncounter
  -> medication_question
  -> medication_answer
  -> safety_flags

HealthReportEncounter
  -> report_input
  -> report_analysis
  -> diet_plan
  -> schedule_plan
```

### 4.6 推荐的最小字段

初版 `EncounterState` 可以只定义最小骨架：

```python
@dataclass
class EncounterState:
    encounter_id: str
    session_id: str
    user_id: str
    intent: str
    status: str
    profile_snapshot: ProfileSnapshot | None
    created_at: str
    updated_at: str | None = None
```

字段解释：

- `encounter_id`：本次医疗事件 ID
- `session_id`：所属技术会话 ID
- `user_id`：所属用户 ID
- `intent`：医疗事件类型，如 `triage`、`medication`、`health_report`
- `status`：事件状态，如 `active`、`waiting`、`completed`、`cancelled`
- `profile_snapshot`：本次事件开始时截取的长期背景快照
- `created_at`：创建时间
- `updated_at`：最后更新时间

后续可以再增加：

```python
facts: list[dict]
record: PreconsultationRecord | None
task_board: TaskBoard | None
assessment: ClinicalAssessment | None
triage_output: TriageOutput | None
```

但第一阶段不建议一次性塞满。

先把“本次医疗事件”这个容器定下来，再逐步把现有 state 字段归位。

### 4.7 Encounter 状态

初版状态建议：

```text
active
waiting
completed
cancelled
```

含义：

- `active`：正在处理本次医疗事件
- `waiting`：系统已经追问，等待用户回答
- `completed`：本次医疗事件已经生成最终结果
- `cancelled`：用户中断或切换到了另一个医疗事件

这和 `DialogueState` 不同。

`DialogueState` 是当前对话路由状态。

`Encounter.status` 是业务事件生命周期状态。

### 4.8 与 DialogueState 的区别

`DialogueState` 当前主要服务于路由：

```text
INIT
TRIAGE_GRAPH_RUNNING
INTAKE_WAITING
ESCALATING
```

它回答的问题是：

```text
下一轮用户输入应该如何被理解？
```

而 `Encounter.status` 回答的问题是：

```text
这一次医疗事件是否还在进行？
```

所以两者不要混用。

例如：

```text
Encounter.status = waiting
DialogueState = INTAKE_WAITING
```

二者可能同时出现，但语义不同。

### 4.9 与当前代码的关系

当前代码已经有显式 `EncounterState` 骨架。

代码落点：

- `medi/core/encounter.py`
  - `EncounterIntent`
  - `EncounterStatus`
  - `EncounterState`
  - `create_encounter()`
  - `mark_active()`
  - `mark_waiting()`
  - `mark_completed()`
  - `mark_cancelled()`
- `medi/core/context.py`
  - `UnifiedContext.active_encounter_id`
- `medi/api/session_store.py`
  - `Session.encounters`
  - `ensure_active_encounter()`
  - `active_encounter()`
  - `mark_active_encounter_waiting()`
  - `mark_active_encounter_completed()`

当前设计是：

```text
UnifiedContext
  -> active_encounter_id

Session
  -> encounters: dict[str, EncounterState]
```

也就是说，`UnifiedContext` 只知道当前正在处理哪个医疗事件，不直接持有完整 `EncounterState`。

完整的 Encounter 集合暂时放在内存级 `Session` 中。

API 和 CLI 在意图路由时创建或复用 Encounter：

```text
NEW_SYMPTOM / SYMPTOM -> EncounterIntent.TRIAGE
MEDICATION             -> EncounterIntent.MEDICATION
HEALTH_REPORT          -> EncounterIntent.HEALTH_REPORT
FOLLOWUP               -> 复用 active_encounter_id
OUT_OF_SCOPE           -> 不创建 Encounter
```

事件状态同步：

```text
FOLLOW_UP  -> EncounterStatus.WAITING
RESULT     -> EncounterStatus.COMPLETED
ESCALATION -> EncounterStatus.COMPLETED
```

当前仍未迁移的字段：

- `clinical_facts`
- `preconsultation_record`
- `task_board`
- `clinical_assessment`
- `triage_output`

这些仍在 `TriageGraphState` 中，下一阶段再逐步归位到 Encounter 下。

### 4.10 当前确认的设计决策

目前确认：

1. `Encounter` 表示一次医疗咨询事件。
2. `Session` 是技术会话，不等于 `Encounter`。
3. 一个 `Session` 可以包含多个 `Encounter`。
4. `HealthProfile` 不属于 `Encounter`。
5. `ProfileSnapshot` 属于 `Encounter`。
6. 分诊类 Encounter 下应逐步承载 `ClinicalFacts`、`PreconsultationRecord`、`TaskBoard`、`ClinicalAssessment` 和 `Output`。
7. 当前初版只实现最小 `EncounterState` 骨架，不一次性塞入全部业务表。
8. `UnifiedContext` 只保存 `active_encounter_id`，不保存完整 Encounter。
9. `Session.encounters` 目前是内存级存储，未来可替换为持久化 EncounterStore。

## 5. ClinicalFacts

### 5.1 定位

`ClinicalFacts` 是本次 `Encounter` 中的事实源。

它表示从本次对话中抽取出来的结构化临床事实，回答的是：

```text
用户在本次医疗事件里到底说过什么？
系统把这些话抽取成了哪些事实？
每条事实来自哪一句证据？
```

它不是最终病历，不是症状摘要，不是任务进度，也不是科室判断。

推荐关系：

```text
Messages
  -> ClinicalFacts
       -> PreconsultationRecord
       -> TaskBoard
       -> ClinicalAssessment
       -> Output
```

### 5.2 三个命名层级

当前确认三个概念：

```text
ClinicalFact
  = 单条事实对象

clinical_facts
  = 多条事实的可序列化列表，放在 state / 未来数据库里

ClinicalFactStore
  = 对这些 facts 做合并、查询、投影的运行时管理器
```

这三个不是同一层东西。

### 5.3 ClinicalFact

`ClinicalFact` 是最小事实单元。

示例：

```python
ClinicalFact(
    slot="hpi.onset",
    status="present",
    value="昨晚开始",
    evidence="我昨晚开始头痛",
    confidence=0.92,
    source_turn=1,
)
```

它表达的是：

- 哪个槽位：`slot`
- 用户是否确认：`status`
- 具体值是什么：`value`
- 证据来自哪句话：`evidence`
- 抽取置信度：`confidence`
- 来自第几轮：`source_turn`

### 5.4 ClinicalFact 如何从 Message 抽取

当前 `ClinicalFact` 不是由 `Message` 对象自己生成的。

它是在 `IntakeNode` 中由“LLM 结构化抽取 + 确定性规则补抽”共同生成的。

当前流程：

```text
state["messages"]
  -> intake_node
  -> resolve_intake_plan(...)
  -> ClinicalFactStore.from_state(state["clinical_facts"])
  -> _call_fact_extractor(...)
  -> extract_deterministic_facts(...)
  -> ClinicalFactStore.merge_items(...)
  -> store.to_state()
  -> state["clinical_facts"]
```

也就是说：

```text
Message 是原始对话记录
ClinicalFact 是从 Message 中抽取出的结构化临床事实
ClinicalFactStore 是负责合并这些事实的运行时管理器
```

#### 5.4.1 LLM 抽取

`IntakeNode` 会把以下信息交给 LLM：

- 完整医患对话：`messages`
- 当前采集计划：`ResolvedIntakePlan`
- 当前允许抽取的 slot 列表
- 已经抽取过的事实：`ClinicalFactStore.prompt_context()`

LLM 的职责非常窄：

```text
只从对话中抽取 facts，严格输出 JSON。
不负责追问。
不负责诊断。
不负责判断流程是否完成。
```

输出形态：

```json
{
  "facts": [
    {
      "slot": "hpi.onset",
      "status": "present",
      "value": "昨晚开始",
      "evidence": "我昨晚开始头痛",
      "confidence": 0.92
    }
  ]
}
```

这些 JSON item 随后会被 `ClinicalFact.from_dict(...)` 转成单条 `ClinicalFact`。

#### 5.4.2 为什么用完整 messages，而不是只看最后一句

很多用户回答必须结合上一句问题才能理解。

例如：

```text
Medi: 您有没有吃过什么药？
User: 布洛芬。
```

如果只看最后一句 `布洛芬`，系统知道它是药名，但不知道它回答的是“当前用药”还是“过敏药物”。

所以当前抽取器读取的是完整对话窗口，并且系统 prompt 会明确要求：

- 不能把护士问题当成患者回答
- 必须只记录患者明确说过的内容
- 遇到否认表达时用 `absent`
- 遇到“不知道”时用 `unknown`

#### 5.4.3 确定性规则补抽

LLM 抽取之后，还会调用：

```python
extract_deterministic_facts(messages, protocol_id=...)
```

这一层用于补一些更稳定、可规则化的事实。

比如某些明确的数值、否认表达、当前 protocol 下非常确定的槽位信息，适合用规则补充，减少 LLM 漏抽导致的重复追问。

所以当前事实来源有两类：

```text
LLM extractor        -> 语义抽取，覆盖复杂自然语言
deterministic rules  -> 稳定补抽，覆盖确定模式
```

#### 5.4.4 Store 合并

抽取结果不会直接散落在 state 里。

它会先进入：

```python
ClinicalFactStore.merge_items(...)
```

合并后再写回：

```python
state["clinical_facts"] = store.to_state()
```

这样做的原因是：

- 同一个 slot 可能被多轮回答重复提到
- 新回答可能比旧回答更准确
- 旧事实不能被低置信度的新事实随便覆盖
- 节点不应该各自实现一套事实合并规则

因此责任边界是：

```text
LLM / rules 只负责产生候选事实
ClinicalFactStore 负责决定事实如何合并
state["clinical_facts"] 只负责保存合并后的事实列表
```

### 5.5 clinical_facts

`clinical_facts` 是放在 `TriageGraphState` 里的可序列化数据字段。

它通常是：

```python
clinical_facts: list[dict]
```

示例：

```python
state["clinical_facts"] = [
    {
        "slot": "hpi.onset",
        "status": "present",
        "value": "昨晚开始",
        "evidence": "我昨晚开始头痛",
        "confidence": 0.92,
        "source_turn": 1,
    }
]
```

它的职责是：

- 能进入 LangGraph state
- 能被 checkpoint 保存和恢复
- 未来能保存到 Encounter 数据库表或 JSON 字段
- 只表达事实数据，不包含复杂查询逻辑

### 5.6 ClinicalFactStore

`ClinicalFactStore` 是运行时管理器。

它负责把 `clinical_facts` 变成方便使用的 Python 对象，并提供合并、查询和投影能力。

当前流程：

```text
state["clinical_facts"]
  -> ClinicalFactStore.from_state(...)
  -> merge_items(...)
  -> is_answered(...)
  -> value(...)
  -> prompt_context(...)
  -> prompt_context(...)
  -> store.to_state()
  -> state["clinical_facts"]
```

为什么需要它？

因为多轮对话会重复提到同一个 slot。

例如：

```text
第一轮：昨天开始头痛
第二轮：准确说是昨晚 10 点左右开始的
```

这两句话都对应：

```text
hpi.onset
```

如果只用 `list[dict]`，每个节点都要自己决定该用哪条事实。

`ClinicalFactStore` 统一负责：

- 空字段补充
- 高置信度事实覆盖低置信度事实
- 判断 slot 是否已回答
- 判断 slot 是否已采集
- 投影成症状视图或 prompt context

### 5.7 为什么不用 FactStore 这个名字

`FactStore` 这个名字太泛。

项目中未来可能会出现很多事实：

- 临床事实
- 报告指标事实
- 用药事实
- 用户偏好事实

所以这里明确叫：

```text
ClinicalFactStore
```

它说明这个 store 管的是临床问诊事实，不是所有事实。

### 5.8 为什么不用 intake_facts 这个名字

`intake_facts` 这个名字偏“护士采集阶段”。

但这些事实不只服务于 intake。

它们还会被以下模块使用：

- `PreconsultationRecord`
- `TaskBoard`
- `ClinicalAssessment`
- `Output`

所以更准确的名字是：

```text
clinical_facts
```

这表示它属于本次医疗事件的临床事实源，而不是某个节点的临时产物。

### 5.9 与 ProfileSnapshot 的关系

`ProfileSnapshot` 是本次医疗事件开始时已有的长期背景。

`ClinicalFacts` 是本次对话中新采集到的事实。

两者不能混。

例如 snapshot 中有：

```text
过敏史：青霉素
```

这不应该自动变成：

```python
ClinicalFact(slot="safety.allergies", value="青霉素")
```

除非用户在本次对话中确认：

```text
是的，我青霉素过敏。
```

所以：

```text
ProfileSnapshot = 背景参考
ClinicalFacts   = 本次确认事实
```

### 5.10 与 T1/T2/T3/T4 的关系

`ClinicalFacts` 是底层原料。

T1/T2/T3/T4 是上层预诊结构。

示例映射：

```text
ClinicalFact(slot="hpi.onset", value="昨晚开始")
  -> T2_HPI.onset

ClinicalFact(slot="safety.current_medications", value="氨氯地平")
  -> T3_BACKGROUND.current_medications

ClinicalFact(slot="hpi.chief_complaint", value="头痛")
  -> T4_CHIEF_COMPLAINT.source
```

这就是：

```text
clinical_facts -> preconsultation_record projection
```

### 5.11 当前代码落点

当前代码已经完成命名统一：

- `medi/agents/triage/clinical_facts.py`
  - `ClinicalFact`
  - `ClinicalFactStore`
  - `required_slots_for_plan()`
- `TriageGraphState.clinical_facts`
- 各 intake / monitor / controller / prompter / clinical / output 节点均读取 `clinical_facts`

当前仍未完成：

- `clinical_facts` 还在 `TriageGraphState` 中
- 尚未挂到 `EncounterState`
- 尚未持久化到 Encounter store

这是后续 Encounter 数据归位的一部分。

## 6. PreconsultationRecord

### 6.1 定位

`PreconsultationRecord` 是本次 `Encounter` 的预诊档案视图。

它不是正式电子病历，也不是长期健康档案。

它回答的问题是：

```text
本次预诊已经为医生整理出了哪些结构化材料？
这些材料分别服务于 T1/T2/T3/T4 哪个预诊任务？
每个字段是否能追溯到本次对话中的 ClinicalFact？
```

推荐数据流：

```text
Messages
  -> ClinicalFacts
  -> PreconsultationRecord
  -> TaskBoard
  -> ClinicalAssessment
  -> Output
```

所以它和 `clinical_facts` 的关系是：

```text
clinical_facts         = 本次对话抽取出的事实源
PreconsultationRecord  = 基于 facts 投影出的 T1/T2/T3/T4 预诊档案
```

`ClinicalFacts` 是底层事实表。

`PreconsultationRecord` 是面向医生交接和任务评估的结构化视图。

### 6.2 为什么不用 MedicalRecord

之前代码里叫：

```text
medical_record
MedicalRecord
```

这个名字容易让人误解为正式病历。

但当前系统生成的只是预诊阶段的结构化草稿，它还没有经过医生确认，也不应该被当作正式病历。

所以统一改成：

```text
preconsultation_record
PreconsultationRecord
```

这个名字表达得更准确：

- 它属于预诊流程
- 它属于本次 Encounter
- 它是医生接诊前的结构化材料
- 它仍然需要保留证据和置信度

### 6.3 当前确认结构

当前结构直接对应 T1/T2/T3/T4：

```text
PreconsultationRecord
  -> meta
  -> t1_triage
  -> t2_hpi
  -> t3_background
  -> t4_chief_complaint
```

### 6.4 meta

`meta` 记录预诊档案本身的元信息。

当前字段：

```python
meta = {
    "schema_version": "preconsultation_record.v1",
    "source": "clinical_facts",
}
```

它说明：

- 当前结构版本是什么
- 这个 record 是从 `clinical_facts` 投影来的

后续可以再加：

```text
encounter_id
session_id
updated_at
projection_version
```

### 6.5 t1_triage

`t1_triage` 对应 T1：分诊方向。

它承载“应该去哪里看、是否有初步科室方向”的内容。

当前字段：

```text
protocol_id
protocol_label
overlay_ids
primary_department
secondary_department
```

示例：

```python
t1_triage = {
    "protocol_id": "fever",
    "protocol_label": "发热",
    "overlay_ids": ["pediatric"],
    "primary_department": {
        "department": "内科",
        "confidence": 0.55,
        "reason": "根据当前预问诊主题初步判断为内科",
    },
    "secondary_department": {
        "department": "发热门诊",
        "confidence": 0.55,
        "reason": "根据当前预问诊主题初步判断为发热门诊",
    },
}
```

注意：

`t1_triage` 里的科室可以先由 intake protocol 给出初步提示，后续 `ClinicalAssessment` 可以生成更完整、更高可信度的科室判断。

### 6.6 t2_hpi

`t2_hpi` 对应 T2：现病史。

它是医生端预诊报告中 HPI 部分的主要材料来源。

当前字段：

```text
chief_complaint
onset
exposure_event
exposure_symptoms
location
character
duration
severity
timing
progression
aggravating_alleviating
radiation
associated_symptoms
diagnostic_history
therapeutic_history
relevant_history
general_condition
specific
```

其中 `general_condition` 放一般情况：

```text
mental_status
sleep
appetite
bowel
urination
weight_change
```

`specific` 放协议特异字段：

```text
max_temperature
frequency
mental_status
intake_urination
...
```

### 6.7 t3_background

`t3_background` 对应 T3：本次预诊确认过的背景史。

当前字段：

```text
disease_history
immunization_history
surgical_history
trauma_history
blood_transfusion_history
allergy_history
current_medications
```

这里要和 `ProfileSnapshot` 区分：

```text
ProfileSnapshot.t3_background?  不存在
PreconsultationRecord.t3_background = 本次对话确认过的背景史
```

如果长期档案里有“青霉素过敏”，但本次对话没有确认，那么它只能作为背景提示进入 prompt，不能自动写入：

```text
t3_background.allergy_history
```

除非用户本次明确确认。

### 6.8 t4_chief_complaint

`t4_chief_complaint` 对应 T4：主诉生成。

当前字段：

```text
source
draft
generated
```

含义：

- `source`：来自 `ClinicalFact(slot="hpi.chief_complaint")` 的证据化字段
- `draft`：从用户主诉直接整理出的草稿
- `generated`：系统生成的规范主诉

示例：

```text
source: 用户说“我昨晚开始头痛，还有点恶心”
draft: 头痛，还有点恶心
generated: 头痛伴恶心约1天
```

### 6.9 字段值为什么保留证据结构

`PreconsultationRecord` 里的多数临床字段不是裸字符串，而是证据化结构。

示例：

```python
{
    "slot": "hpi.onset",
    "status": "present",
    "value": "昨晚开始",
    "evidence": "我昨晚开始头痛",
    "confidence": 0.92,
    "source_turn": 1,
}
```

这样做的原因：

- 医生端输出可以追溯证据
- TaskBoard 可以判断字段是否已完成
- 后续 UI 可以展示“这条信息来自哪句话”
- 后续持久化时不会丢失事实来源

### 6.10 与 TaskBoard 的关系

`TaskBoard` 读取 `PreconsultationRecord` 来判断 T1/T2/T3/T4 的完成度。

当前代码已经把任务状态收进：

```text
TriageGraphState.task_board
```

内部 progress 读取新的路径：

```text
t1_triage.primary_department
t2_hpi.onset
t2_hpi.general_condition.mental_status
t3_background.current_medications
t4_chief_complaint.generated
```

这意味着：

```text
ClinicalFacts 负责“用户说过什么”
PreconsultationRecord 负责“预诊档案现在长什么样”
TaskBoard 负责“这些档案是否足够完成任务”
```

### 6.11 当前代码落点

当前代码已经完成：

- `medi/agents/triage/preconsultation_record.py`
  - `update_preconsultation_record()`
  - `PRECONSULTATION_RECORD_SCHEMA_VERSION`
  - `t1_triage / t2_hpi / t3_background / t4_chief_complaint` 投影
- `TriageGraphState.preconsultation_record`
- `empty_preconsultation_record()`
- `IntakeNode` 写入 `preconsultation_record`
- `IntakeMonitorNode` 用 `preconsultation_record` 评估任务进度
- `IntakePrompterNode` 把 `preconsultation_record` 放入追问上下文
- `OutputNode` 优先读取 `preconsultation_record` 生成医生端预诊报告
- `task_definitions.py` 的 `record_paths` 已迁移到 T1/T2/T3/T4 路径
- `task_progress.py` 的入口参数已改为 `preconsultation_record`
- `TriageGraphState.task_board`

当前仍未完成：

- `preconsultation_record` 仍在 `TriageGraphState` 中
- 尚未挂到 `EncounterState`
- 尚未持久化到 Encounter store

## 7. TaskBoard

### 7.1 定位

`TaskBoard` 是本次预诊的任务看板。

它不保存用户原话，也不保存临床事实本身。

它回答的问题是：

```text
为了完成这次预诊，T1/T2/T3/T4 哪些任务已经完成？
哪些任务还缺？
下一轮最应该推进哪个任务？
某个任务已经被追问过几次？
```

推荐数据流：

```text
ClinicalFacts
  -> PreconsultationRecord
  -> TaskBoard
  -> Controller
  -> Prompter
  -> Inquirer
```

### 7.2 与 PreconsultationRecord 的关系

`PreconsultationRecord` 是预诊档案。

`TaskBoard` 是围绕这份档案的任务完成度和调度状态。

两者区别：

```text
PreconsultationRecord
  = 已经整理出了哪些预诊材料

TaskBoard
  = 这些材料是否足够完成 T1/T2/T3/T4 任务
```

示例：

```text
t2_hpi.onset = 昨晚开始
t2_hpi.location = 缺失
t2_hpi.exposure_event = 缺失
```

则 `TaskBoard.progress["T2_ONSET"]` 可能是：

```python
{
    "task_id": "T2_ONSET",
    "score": 0.33,
    "status": "partial",
    "completed_requirements": ["onset_time"],
    "missing_requirements": ["onset_location", "onset_trigger"],
}
```

### 7.3 当前结构

当前 `TaskBoard` 放在 `TriageGraphState.task_board` 中。

结构：

```text
TaskBoard
  -> monitor
  -> controller
  -> tree
  -> progress
  -> pending_tasks
  -> current_task
  -> task_rounds
```

### 7.4 monitor

`monitor` 是 Monitor 节点输出的质量评估结果。

它来自：

```text
IntakeMonitorNode
```

主要字段：

```text
score
red_flags_checked
safety_slots_covered
doctor_summary_ready
required_slots_covered
high_value_missing_slots
reason
```

它是整体质量门，不是单个任务的 progress。

### 7.5 progress

`progress` 是每个 T1/T2/T3/T4 子任务的完成度。

它来自：

```text
evaluate_task_progress(preconsultation_record=...)
```

当前判定是确定性规则，不由 LLM 直接决定。

判定逻辑：

```text
completed_requirements / total_requirements = score
```

状态规则：

```text
score >= 0.85 -> complete
score > 0     -> partial
score == 0    -> pending
```

所以：

```text
pending  = 一个 requirement 都没有满足
partial  = 满足了一部分 requirement
complete = 达到任务完成阈值
```

### 7.6 tree

`tree` 是 T1/T2/T3/T4 的层级任务视图。

它来自：

```text
build_intake_task_tree(...)
```

它适合用于 UI 展示、调试和解释：

```text
T1 Triage
T2 HPI
T3 Background
T4 Chief Complaint
```

当前它仍由 `ClinicalFactStore` 和 intake plan 生成。

### 7.7 pending_tasks

`pending_tasks` 是当前未达到完成阈值的任务 ID 列表。

Controller 会读取它来选择下一轮任务。

注意：

`T4_CHIEF_COMPLAINT_GENERATION` 目前不会在追问阶段优先调度，因为它通常可以由已有材料生成，不应该为了生成主诉单独追问患者。

### 7.8 controller

`controller` 是 Controller 节点输出的调度决策。

主要字段：

```text
can_finish_intake
next_best_task
task_priority_score
task_instruction
next_best_question
reason
```

其中：

- `next_best_task`：下一轮要推进的任务
- `task_instruction`：给 Prompter 的任务说明
- `next_best_question`：Prompter 生成后的自然语言追问

### 7.9 current_task

`current_task` 是本轮 Controller 选中的任务。

它用于：

- Prompter 生成问题
- Inquirer 记录本轮追问属于哪个任务
- 后续 `task_rounds` 计数

### 7.10 task_rounds

`task_rounds` 记录每个任务被追问过几次。

例如：

```python
task_rounds = {
    "T2_ONSET": 1,
    "T3_CURRENT_MEDICATIONS": 2,
}
```

Controller 会用它做重复追问惩罚，避免一直追着同一个任务问。

### 7.11 当前代码落点

当前代码已经完成：

- `TriageGraphState.task_board`
- `empty_task_board()`
- `IntakeMonitorNode`
  - 写入 `task_board.monitor`
  - 写入 `task_board.tree`
  - 写入 `task_board.progress`
  - 写入 `task_board.pending_tasks`
- `IntakeControllerNode`
  - 读取 `task_board.monitor/progress/pending_tasks/task_rounds`
  - 写入 `task_board.controller/current_task`
- `IntakePrompterNode`
  - 读取 `task_board.controller/current_task/progress`
  - 写入 `task_board.controller.next_best_question`
- `IntakeInquirerNode`
  - 读取 `task_board.controller/current_task`
  - 写入 `task_board.task_rounds`
- `TriageGraphRunner`
  - 初始化 `task_board=empty_task_board()`

当前仍未完成：

- `TaskBoard` 仍在 `TriageGraphState` 中
- 尚未挂到 `EncounterState`
- 尚未持久化到 Encounter store
- `task_progress.py` 和 `task_tree.py` 仍作为底层 evaluator/builder 模块保留

## 8. ClinicalAssessment

### 8.1 定位

`ClinicalAssessment` 是 ClinicalNode 生成的临床判断视图。

它不负责采集信息，也不负责生成最终患者建议。

它回答的问题是：

```text
基于当前预诊档案和用户长期背景，临床上初步怎么判断？
应该推荐哪些科室？
当前急不急？
有哪些患者特异性风险因子？
有哪些可能的鉴别诊断？
如果信息不足，明确还缺哪些字段？
```

推荐数据流：

```text
ClinicalFacts
  -> PreconsultationRecord
  -> TaskBoard
  -> ClinicalNode
  -> ClinicalAssessment
  -> Output
```

所以：

```text
ClinicalNode 是 ClinicalAssessment 的生产者
OutputNode 是 ClinicalAssessment 的消费者
```

### 8.2 与 PreconsultationRecord 的关系

`PreconsultationRecord` 是预诊档案。

`ClinicalAssessment` 是基于这份档案做出的临床判断。

两者区别：

```text
PreconsultationRecord
  = 已知信息是什么

ClinicalAssessment
  = 基于已知信息，临床上怎么初步判断
```

示例：

```text
t2_hpi.chief_complaint = 胸闷
t2_hpi.onset = 今天上午
t3_background.disease_history = 高血压
```

这些是档案。

ClinicalNode 可能进一步生成：

```text
urgency.level = urgent
risk_factors.summary = 患者有高血压史，胸闷需优先排除心源性原因
department_candidates = 心血管内科 / 急诊内科
differential_diagnoses = 心绞痛、焦虑相关胸闷、呼吸系统疾病等
```

这才是 `ClinicalAssessment`。

### 8.3 当前结构

当前结构：

```text
ClinicalAssessment
  -> department_candidates
  -> urgency
  -> differential_diagnoses
  -> risk_factors
  -> missing_slots
  -> status
```

### 8.4 department_candidates

`department_candidates` 是科室候选列表。

它来自：

```text
DepartmentRouter.route(...)
```

结构示例：

```python
[
    {
        "department": "心血管内科",
        "confidence": 0.82,
        "reason": "胸闷伴高血压史，需优先评估心血管原因",
    }
]
```

它不是最终诊断，只是就诊方向建议。

### 8.5 urgency

`urgency` 是紧急程度判断。

当前字段：

```text
level
reason
```

示例：

```python
urgency = {
    "level": "urgent",
    "reason": "胸闷合并基础疾病风险，建议尽快就医评估。",
}
```

当前 `level` 可取：

```text
emergency
urgent
normal
watchful
```

规则层红旗症状仍在 `TriageGraphRunner` 前置处理。

ClinicalNode 中的 urgency 是语义层评估，并会结合 `ProfileSnapshot` 风险因子做必要升级。

### 8.6 risk_factors

`risk_factors` 是患者特异性风险因子分析。

它来自：

```text
evaluate_risk_factors(symptom_summary, profile_snapshot)
```

当前字段：

```text
items
summary
elevated_urgency
```

示例：

```python
risk_factors = {
    "items": ["高血压史（心血管症状需排除相关急症）"],
    "summary": "患者存在以下风险因子，建议医生重点关注：高血压史...",
    "elevated_urgency": True,
}
```

这里会读取 `ProfileSnapshot`，但不会修改 `HealthProfile`。

### 8.7 differential_diagnoses

`differential_diagnoses` 是初步鉴别诊断列表。

它来自 ClinicalNode 中的 LLM 结构化调用。

字段：

```text
condition
likelihood
reasoning
supporting_symptoms
risk_factors
```

它不是最终诊断。

它用于医生端预诊报告和患者端就医方向解释。

### 8.8 missing_slots

`missing_slots` 是 ClinicalNode 认为“临床判断前仍值得明确回补”的字段。

例如：

```text
hpi.location
hpi.radiation
specific.dyspnea_sweating
```

当满足以下条件时，图会 back-loop 回到 Intake：

```text
没有高可能性鉴别诊断
且 missing_slots 非空
且 workflow_control.graph_iteration < 2
```

然后 IntakeMonitor 会读取：

```text
clinical_assessment.missing_slots
```

把这些缺口纳入下一轮任务评估。

### 8.9 status

`status` 表示本次临床判断状态。

当前值：

```text
pending
needs_more_info
complete
```

含义：

- `pending`：ClinicalNode 尚未运行
- `needs_more_info`：ClinicalNode 已运行，但需要回到 Intake 补问
- `complete`：ClinicalNode 已生成可供 Output 使用的判断

后续如需表达急症升级，可增加：

```text
escalated
```

### 8.10 当前代码落点

当前代码已经完成：

- `TriageGraphState.clinical_assessment`
- `empty_clinical_assessment()`
- `ClinicalNode`
  - 写入 `clinical_assessment.department_candidates`
  - 写入 `clinical_assessment.urgency`
  - 写入 `clinical_assessment.risk_factors`
  - 写入 `clinical_assessment.differential_diagnoses`
  - 写入 `clinical_assessment.missing_slots`
  - 写入 `clinical_assessment.status`
- `IntakeMonitorNode`
  - 读取 `clinical_assessment.missing_slots`
- `OutputNode`
  - 读取 `clinical_assessment` 生成 `triage_output.patient` 和 `triage_output.doctor_report`
- `TriageGraphRunner`
  - 初始化 `clinical_assessment=empty_clinical_assessment()`

当前仍未完成：

- `ClinicalAssessment` 仍在 `TriageGraphState` 中
- 尚未挂到 `EncounterState`
- 尚未持久化到 Encounter store
- `risk_factors.items` 尚未单独在 UI 或输出中展开展示

## 9. TriageOutput

### 9.1 定位

`TriageOutput` 是分诊流程的最终输出对象。

它不是采集过程中的工作草稿，也不是 ClinicalNode 的中间判断，而是 OutputNode 在预诊结束时生成的对外结果。

它回答的是：

```text
这次分诊最后要给患者看什么？
这次分诊最后要给医生交接什么？
这份输出是什么版本、什么时候生成、是否发生过降级？
```

当前确认统一使用一个顶层字段：

```text
triage_output
```

不再保留旧的顶层 `patient_output`、`doctor_hpi`，也不再在事件 data 里平铺 `patient`、`doctor_report`。

### 9.2 当前结构

```text
TriageOutput
  -> meta
  -> patient
  -> doctor_report
```

对应代码：

```python
class TriageOutput(TypedDict):
    meta: TriageOutputMeta
    patient: TriagePatientOutput
    doctor_report: DoctorReport
```

`patient` 是患者侧结果。

`doctor_report` 是医生侧预诊报告。

`meta` 是输出本身的元信息。

### 9.3 meta

`meta` 不承载临床内容，只承载输出对象的技术元信息。

当前字段：

```python
class TriageOutputMeta(TypedDict, total=False):
    schema_version: str
    generated_at: str
    source: str
    fallback_used: bool
    fallback_reason: str | None
```

含义：

- `schema_version`：输出结构版本，当前为 `triage_output.v1`
- `generated_at`：OutputNode 生成输出的时间
- `source`：生成来源，当前为 `output_node`
- `fallback_used`：是否因为 LLM 输出失败或 JSON 不合法而使用降级生成
- `fallback_reason`：降级原因

### 9.4 patient

`patient` 是面向患者展示的分诊结果。

当前结构：

```python
class TriagePatientOutput(TypedDict):
    primary_department: DepartmentResult
    alternative_departments: list[DepartmentResult]
    urgency_level: str
    urgency_reason: str
    patient_advice: str
    red_flags_to_watch: list[str]
```

它应该保持简洁、温和、可执行。

它不承担完整临床交接职责，也不展示过多鉴别诊断细节，重点是：

- 首选科室
- 备选科室
- 紧急程度
- 就医建议
- 需要立即就医的危险信号

### 9.5 doctor_report

`doctor_report` 是面向接诊医生的预诊报告，不只是 HPI。

HPI 仍然是它的核心部分，但医生接诊前需要看的信息通常还包括：

- 患者基础信息
- 主诉
- HPI 叙述
- OLDCARTS/OPQRST 关键字段
- 伴随症状和相关阴性
- 检查/诊断经过
- 治疗经过
- 一般情况
- 既往史、用药、过敏
- 鉴别诊断
- 建议检查
- 分诊摘要
- 仍未采集到的关键项

所以命名使用 `DoctorReport`，而不是 `DoctorHPI`。

### 9.6 数据来源

`TriageOutput` 由 OutputNode 生成，读取的数据来源包括：

```text
clinical_facts
preconsultation_record
clinical_assessment
profile_snapshot
episodic memory
conversation messages
```

推荐数据流：

```text
Messages
  -> ClinicalFacts
  -> PreconsultationRecord
  -> TaskBoard
  -> ClinicalAssessment
  -> TriageOutput
       -> patient
       -> doctor_report
```

其中：

- `patient` 主要消费 `ClinicalAssessment.department_candidates`、`ClinicalAssessment.urgency` 和安全提醒
- `doctor_report` 主要消费 `PreconsultationRecord`，并补充 `ClinicalAssessment`、`ProfileSnapshot` 和对话上下文

### 9.7 为什么不兼容旧字段

这次输出层选择一次性改干净。

不再兼容：

```text
patient_output
doctor_hpi
event.data["patient"]
event.data["doctor_report"]
```

原因是输出层属于最终对外合同，如果同时保留多套字段，会产生几个问题：

- API、CLI、前端会不知道该读哪一个
- 后续持久化 Encounter 时会出现重复字段
- 文档和代码会继续混用旧命名
- 医生侧报告会被误解为只有 HPI

统一成 `triage_output.patient` 与 `triage_output.doctor_report` 后，结构更稳定：

```text
event.data["triage_output"]["patient"]
event.data["triage_output"]["doctor_report"]
```

### 9.8 当前代码落点

当前代码已经完成：

- `TriageGraphState.triage_output`
- `TriageOutput`
- `TriageOutputMeta`
- `TriagePatientOutput`
- `DoctorReport`
- `OutputNode`
  - LLM JSON schema 要求输出 `triage_output`
  - RESULT 事件只透传 `triage_output`
  - 降级输出也收口为 `triage_output`
- `TriageGraphRunner`
  - 通过 `result["triage_output"]` 判断分诊图完成
- API
  - metadata 只透传 `triage_output`
- CLI
  - 从 `event.data["triage_output"]` 展示患者端结果和医生端报告

当前仍未完成：

- `triage_output` 仍在 `TriageGraphState` 中
- 尚未挂到 `EncounterState`
- 尚未持久化到生产级 Encounter store

## 10. 症状摘要文本

### 10.1 为什么删除 state.symptom_data

`symptom_data` 曾经是从对话里抽取出来的扁平症状结构。

但现在项目已经明确：

```text
ClinicalFacts = 本次事实源
PreconsultationRecord = T1/T2/T3/T4 预诊档案视图
```

继续把 `symptom_data` 放在 `TriageGraphState` 里，会让它看起来像第三份事实数据，和 `clinical_facts`、`preconsultation_record` 形成重复。

所以当前决定是：

```text
删除 TriageGraphState.symptom_data
不再保留 SymptomData
不再通过 ClinicalFactStore.to_symptom_data() 生成中间结构
```

### 10.2 保留的能力

删除 `symptom_data` 不等于删除症状摘要能力。

现在改为纯函数现场生成：

```python
from medi.agents.triage.clinical_summary import build_symptom_summary_from_record

build_symptom_summary_from_record(
    preconsultation_record,
    clinical_facts,
    messages,
)
```

它返回的是一段文本，例如：

```text
主诉：发热伴咽痛1天
起病时间：昨晚
部位：咽部
最高体温：39度
伴随症状：咳嗽、乏力
用药：布洛芬
过敏：青霉素过敏
```

这段文本不进入 state，不持久化，不作为事实源。

它只是给 LLM prompt、科室检索、风险判断和意图识别使用的运行时摘要。

### 10.3 数据来源顺序

摘要函数的读取顺序是：

```text
PreconsultationRecord 优先
ClinicalFacts 补充
messages 兜底
```

含义：

- `PreconsultationRecord` 已经把事实投影成 T1/T2/T3/T4，更适合生成可读摘要
- `ClinicalFacts` 保留槽位和值，可在 record 缺失时补充
- `messages` 只在结构化数据还很少时兜底，避免首轮无法检索或分类

### 10.4 当前代码落点

当前代码已经完成：

- 删除 `TriageGraphState.symptom_data`
- 删除 `SymptomData`
- 删除 `empty_symptom_data()`
- 删除 `ClinicalFactStore.to_symptom_data()`
- 将运行时摘要构建能力收口到 `medi/agents/triage/clinical_summary.py`
- `IntakeNode` 不再返回 `symptom_data`
- `ClinicalNode`
  - 用 `build_symptom_summary_from_record(...)` 生成 urgency/differential prompt 文本
  - 用 `build_department_query_from_record(...)` 生成科室检索文本
  - back-loop 缺口判断直接读取 `ClinicalFacts` 和 `PreconsultationRecord`
- `OutputNode`
  - 用 `build_symptom_summary_from_record(...)` 生成输出 prompt 的症状摘要
- `TriageGraphRunner`
  - 不再缓存 `_cached_symptom_data`
  - 改为缓存 `_cached_symptom_summary` 文本，给 Orchestrator 意图识别参考

### 10.5 当前设计决策

当前确认：

1. 症状摘要是运行时派生文本，不是数据层。
2. 本次事实源只认 `ClinicalFacts`。
3. 本次预诊档案视图只认 `PreconsultationRecord`。
4. LLM prompt 或检索需要文本时，从 record/facts/messages 现场生成。
5. 不新增 `PresentationView` 或其他持久化中间对象。

## 11. IntakePlan 与 WorkflowControl

### 11.1 为什么继续收敛

在删除 `symptom_data` 后，`TriageGraphState` 里还残留了一批“看起来像数据，但其实只是流程标记”的字段。

这会造成两个问题：

1. 临床数据、采集协议、图路由信号混在同一层。
2. 后续维护时很难判断某个字段是否应该持久化、是否能给医生看、是否只是 LangGraph 内部控制。

因此本轮把它们继续收敛成两个明确对象：

```text
intake_plan       = 本次预问诊采用什么采集协议
workflow_control  = LangGraph 下一步怎么走
```

### 11.2 IntakePlanState

`IntakePlanState` 表示当前 Encounter 的采集协议选择结果。

它回答的是：

```text
这次预问诊应该按什么主诉协议采集？
是否叠加儿科、妊娠等 overlay？
```

当前字段：

```python
class IntakePlanState(TypedDict, total=False):
    protocol_id: str
    protocol_label: str
    overlay_ids: list[str]
```

它取代了旧字段：

```text
intake_protocol_id
intake_overlays
```

这样协议相关信息都归在一个对象下，而不是散落在 state 顶层。

### 11.3 WorkflowControl

`WorkflowControl` 表示 LangGraph 内部运行控制信号。

它回答的是：

```text
图下一步应该进入哪个节点？
本次预问诊是否可以进入 ClinicalNode？
当前已经经历了几轮 intake graph 迭代？
```

当前字段：

```python
class WorkflowControl(TypedDict, total=False):
    next_node: str
    intake_complete: bool
    graph_iteration: int
```

它取代了旧顶层字段：

```text
next_node
intake_complete
graph_iteration
```

关键点是：`WorkflowControl` 不是临床数据，也不是患者档案。它只服务于图的路由和循环防护。

### 11.4 删除的旧字段

本轮确认删除：

```text
requested_slots
triage_done
error
collection_status
```

删除理由：

- `requested_slots`：旧槽位追问系统的痕迹，当前由 `TaskBoard.task_rounds` 记录任务级追问次数。
- `triage_done`：与 `triage_output is not None`、`workflow_control.next_node == "done"` 表达重复。
- `error`：错误是运行时事件，应该通过 `StreamEvent(ERROR)` 发出，不放进临床 state。
- `collection_status`：由 facts 推导出的旧聚合状态，和 `ClinicalFacts` / `TaskBoard.progress` 重复。

### 11.5 collection_status 为什么删

以前的结构大概是：

```text
ClinicalFacts -> collection_status -> Controller / Clinical
```

现在改为：

```text
ClinicalFacts -> PreconsultationRecord -> TaskBoard.progress -> Controller
ClinicalFacts + PreconsultationRecord -> ClinicalAssessment
```

也就是说：

- “事实有没有问过”由 `ClinicalFactStore.is_collected()` 判断。
- “事实是否有效回答”由 `ClinicalFactStore.is_answered()` 判断。
- “T1/T2/T3/T4 任务完成度”由 `TaskBoard.progress` 判断。
- “ClinicalNode 是否需要回补”由 `ClinicalAssessment.missing_slots` 表达。

因此 `collection_status` 不再承担独立职责。

### 11.6 当前 TriageGraphState 结构

当前预诊图的核心 state 已收敛为：

```text
TriageGraphState
  session_id
  user_id
  messages
  intake_plan
  workflow_control
  clinical_facts
  preconsultation_record
  task_board
  clinical_assessment
  triage_output
```

其中：

```text
messages                 = 对话历史
intake_plan              = 采集协议
workflow_control         = 图运行控制
clinical_facts           = 本次事实源
preconsultation_record   = T1/T2/T3/T4 预诊档案视图
task_board               = 任务完成度与调度看板
clinical_assessment      = 临床判断视图
triage_output            = 最终患者侧与医生侧输出
```

这也是后续迁移到生产级持久化时推荐的层级边界。

## 12. 后续 Data 优化路线

### 12.1 当前阶段结论

分诊链路的核心数据已经基本收敛：

```text
HealthProfile
ProfileSnapshot
EncounterState
ClinicalFacts
PreconsultationRecord
TaskBoard
ClinicalAssessment
TriageOutput
IntakePlanState
WorkflowControl
```

后续不建议继续在同一层随意新增字段。

下一阶段应该围绕“生产级必要性”逐个设计，优先处理所有意图都会复用的底层数据，再处理分诊之外的业务数据。

### 12.2 Message 数据模型

当前 `messages` 仍然是：

```python
list[dict]
```

这在原型阶段够用，但生产级会遇到几个问题：

- 缺少稳定 `message_id`
- 缺少 `created_at`
- 缺少 `encounter_id`
- 缺少附件结构
- 缺少消息来源，如 user / assistant / tool / system / upload
- 不利于审计、回放、分页、跨端同步

后续建议建立明确的消息模型，例如：

```python
@dataclass
class ConversationMessage:
    message_id: str
    session_id: str
    encounter_id: str | None
    role: str
    content: str
    created_at: str
    source: str = "chat"
    attachments: tuple[dict, ...] = ()
```

它的定位是：

```text
ConversationMessage = 对话与输入事件的最小持久化单元
```

它不是临床事实，也不是输出结果。

推荐优先级：高。

原因是分诊、用药咨询、健康报告分析都会依赖消息模型。

### 12.3 Encounter 持久化结构

当前 `EncounterState` 还只是业务事件骨架。

分诊图内部的这些数据仍然主要留在 `TriageGraphState`：

```text
clinical_facts
preconsultation_record
task_board
clinical_assessment
triage_output
```

后续需要明确：

```text
哪些数据是运行时 state？
哪些数据要归档到 Encounter？
哪些数据只在图执行中临时存在？
```

推荐方向：

```text
Encounter
  -> profile_snapshot
  -> messages
  -> clinical_facts
  -> preconsultation_record
  -> clinical_assessment
  -> triage_output
```

`TaskBoard` 是否持久化需要单独讨论。

它有调试和审计价值，但未必是医生或患者需要直接看到的业务结果。

推荐优先级：高。

原因是生产级需要恢复会话、审计输出来源、支持多进程和水平扩容。

### 12.4 Medication 数据结构

目前用药咨询还没有像分诊一样形成明确的 record / assessment / output 分层。

后续建议建立：

```text
MedicationConsultationRecord
MedicationAssessment
MedicationOutput
```

可能的数据边界：

```text
MedicationConsultationRecord
  = 用户本次问药相关的事实记录

MedicationAssessment
  = 药物适应症、禁忌、相互作用、风险提醒等判断

MedicationOutput
  = 面向用户的用药建议、安全提醒和就医建议
```

它不应该复用 `ClinicalAssessment`。

原因是用药咨询关注的是药物安全和用药决策，和分诊的科室、紧急程度、鉴别诊断不是同一类输出。

推荐优先级：中高。

### 12.5 HealthReport 数据结构

健康报告分析也需要自己的数据边界。

后续建议建立：

```text
HealthReportRecord
HealthReportAssessment
HealthReportOutput
```

可能的数据边界：

```text
HealthReportRecord
  = 上传报告、指标、参考范围、异常标记、报告时间

HealthReportAssessment
  = 指标解释、风险分层、关联因素、需复查项目

HealthReportOutput
  = 面向用户的解释、饮食建议、复查建议、就医建议
```

它也不应该直接塞进 `ClinicalFacts`。

报告指标不是用户口述症状事实，应该有自己的结构。

推荐优先级：中高。

### 12.6 ProfileUpdateCandidate

当前已经明确：

```text
HealthProfile   = 长期用户档案
ProfileSnapshot = 本次只读快照
```

但还缺少一层：

```text
本次 Encounter 结束后，哪些信息值得回写到长期 HealthProfile？
```

当前代码尚未实现这层逻辑。

现在分诊结束后只会把本次咨询摘要保存为 `visit_records` 历史就诊记录，不会自动更新：

```text
profiles.chronic_conditions
profiles.allergies
profiles.current_medications
```

后续计划建立：

```python
@dataclass
class ProfileUpdateCandidate:
    field: str
    value: str
    source_encounter_id: str
    evidence: str
    confidence: float
    requires_confirmation: bool = True
```

它的定位是：

```text
ProfileUpdateCandidate = 长期档案回写候选，不是直接写入结果
```

推荐流程：

```text
ClinicalFacts / PreconsultationRecord
  -> ProfileUpdater
  -> ProfileUpdateCandidate[]
  -> 用户确认或规则校验
  -> save_profile(profile)
```

例如：

- 新确认的药物过敏
- 新确认的慢性病
- 新确认的长期用药
- 重要手术史
- 重要家族史

原则：

```text
不能因为用户一句话就自动污染长期 HealthProfile。
需要确认、置信度和来源证据。
```

推荐优先级：中。

### 12.7 SlotCatalog

当前 `ClinicalFact.slot` 是字符串：

```text
hpi.onset
specific.max_temperature
safety.current_medications
```

这种方式灵活，但生产级会有风险：

- 拼写错误不容易发现
- 同义槽位可能重复出现
- slot 的含义容易漂移
- 很难知道某个 slot 属于 T1/T2/T3/T4 哪一层
- 很难统一管理问题模板、优先级和映射关系

后续可以建立 `SlotCatalog`：

```text
SlotCatalog
  -> slot_id
  -> label
  -> group
  -> t_layer
  -> default_question
  -> priority
  -> aliases
```

它不是本次事实数据，而是事实槽位的元数据目录。

推荐优先级：中。

原因是当前 slot 数量还可控，可以先继续用字符串；等新增更多主诉协议和报告指标后，再统一 catalog 会更稳。

### 12.8 推荐执行顺序

推荐后续顺序：

```text
1. Message / ConversationMessage
2. Encounter 持久化结构
3. MedicationConsultationRecord / MedicationAssessment / MedicationOutput
4. HealthReportRecord / HealthReportAssessment / HealthReportOutput
5. ProfileUpdateCandidate
6. SlotCatalog
```

理由：

```text
Message 是所有意图的底座
Encounter 是生产级恢复和审计的业务容器
Medication / HealthReport 是分诊之外的业务结构补齐
ProfileUpdateCandidate 是长期档案回写安全层
SlotCatalog 是规模扩大后的规范化层
```

下一步最推荐先讨论 `ConversationMessage`。

## 13. Prompt 归类

### 13.1 当前原则

Prompt 不属于数据模型。

但大块 LLM prompt 会直接影响数据抽取、任务追问、临床判断和最终输出，所以需要从节点流程代码中抽离出来，集中管理。

当前原则：

```text
大块 LLM prompt 抽到 prompts 模块
短小 fallback 文案暂时留在原业务文件
数据对象自己的 prompt_context() 暂时保留在对象附近
```

这样可以避免两个极端：

- prompt 全散在节点里，后续很难审查和迭代
- 所有文案都抽走，导致读流程时到处跳转

### 13.2 当前 triage prompt 落点

分诊主链路已建立：

```text
medi/agents/triage/prompts/
  __init__.py
  fact_extraction.py
  intake_prompter.py
  differential.py
  triage_output.py
```

对应职责：

```text
fact_extraction.py
  -> 事实抽取 system prompt
  -> build_fact_extract_prompt(...)

intake_prompter.py
  -> 追问生成 system prompt
  -> build_prompter_input(...)

differential.py
  -> 鉴别诊断 prompt
  -> build_differential_prompt(...)

triage_output.py
  -> 最终患者端/医生端输出 system prompt
  -> build_triage_output_user_context(...)
```

节点现在只负责：

```text
读取 state
组织运行时参数
调用 prompt builder
调用 LLM
解析结果
写回 state
```

### 13.3 暂时不抽的内容

以下内容暂时不强行抽离：

```text
fallback question
fallback patient advice
StreamEvent 展示文案
slot question
task description
```

原因：

- 它们有些属于 UI/event 文案，不是 LLM prompt。
- 有些和 slot/task 定义强绑定，留在 `clinical_facts.py`、`task_definitions.py` 附近更容易维护。
- 过早统一成一个文案目录，会让业务阅读成本上升。

### 13.4 后续计划

后续如果 prompt 继续变多，可以再做：

```text
Prompt version
Prompt registry
Prompt evaluation cases
A/B prompt experiment
```

当前阶段不引入复杂模板引擎。

Python 常量 + prompt builder 函数已经足够。
