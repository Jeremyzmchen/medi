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

也就是说，`HealthProfile` 不是 `medical_record` 的父字段，也不是 `symptom_data` 的父字段。

更准确地说：

- `HealthProfile` 提供长期背景
- `Encounter` 表示一次医疗咨询事件
- `ClinicalFacts` 记录本次对话中抽取到的事实
- `PreconsultationRecord` 把本次 facts 投影成 T1/T2/T3/T4 结构
- `ClinicalAssessment` 结合本次记录和长期背景做科室、风险、紧急程度判断
- `Output` 生成患者端建议和医生端 HPI

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

后续可以设计一个独立的 `ProfileUpdater` 或 `ProfileUpdateService`。

它的职责是：

1. 从本次 `PreconsultationRecord` 中识别可能属于长期档案的信息。
2. 判断这些信息是否足够明确、稳定、可回写。
3. 必要时要求用户确认。
4. 将确认后的信息写入 `HealthProfile`。

示例：

```text
用户说：“我有高血压，长期吃氨氯地平。”

本次 Facts:
  ph.disease_history = 高血压
  safety.current_medications = 氨氯地平

预诊结束后:
  ProfileUpdater 识别为长期信息
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

后续可以在 `TriageGraphState` 中增加：

```python
class TriageGraphState(TypedDict, total=False):
    profile_snapshot: ProfileSnapshot
    ...
```

但约束是：

- 节点可以读取 `profile_snapshot`
- 节点不应该修改 `profile_snapshot`
- 节点不能把 `profile_snapshot` 当成本次事实源
- 如果需要把其中某个信息纳入本次病历，应通过提问确认或明确标记来源为 profile

### 2.12 当前确认的设计决策

目前确认以下原则：

1. `ProfileSnapshot` 是从长期 `HealthProfile` 截取的只读快照。
2. `ProfileSnapshot` 生命周期属于单次 `Encounter`。
3. `ProfileSnapshot` 可以进入 graph state，但不被节点修改。
4. `ProfileSnapshot` 不是本次事实源，不能替代 `ClinicalFacts`。
5. `ProfileSnapshot` 可以用于 T1 风险判断、T3 背景参考和最终输出约束。
6. `ProfileSnapshot` 应随未来的 `Encounter` 一起保存，保证历史咨询可复现。
7. 本次新获得的长期信息应先进入 `ClinicalFacts`，预诊结束后再由独立更新逻辑选择性回写 `HealthProfile`。
