# 持久化分诊会话恢复开发方案

## 背景

当前分诊图已经使用 LangGraph `interrupt/resume` 实现追问暂停和恢复，但 checkpoint 仍主要依赖进程内存。只要服务重启、API 进程被回收、用户刷新页面后服务状态丢失，系统就可能不知道当前 `session_id` 原本处于哪一步。

这个机制不应该只服务于 `interrupt` 阶段。更完整的目标是：只要用户没有明确结束本次分诊，系统就能根据 `session_id` 恢复未完成的问诊状态，包括已经采集的事实、任务进度、上次追问、是否已经进入临床推理，以及是否已经完成输出。

这和客服系统里的“确认关闭此次会话”类似：

- 页面关闭、网络中断、服务重启，不等于会话结束。
- 用户明确点击结束、重新开始，才表示关闭或放弃当前会话。
- 未结束的会话再次进入时，应提示用户继续还是重新开始。

## 目标

1. 将 LangGraph checkpoint 从 `MemorySaver` 升级为可持久化 checkpointer。
2. 用 `session_id` 作为 LangGraph `thread_id`，跨进程恢复同一分诊图状态。
3. 支持任何未完成阶段恢复，而不仅是 interrupt 等待回答阶段。
4. 提供用户可理解的恢复上下文，而不是只展示一句“上次问到的问题”。
5. 提供明确的继续、重新开始、结束本次分诊动作。
6. 为后续审计、回放、time travel 调试打基础。

## 非目标

第一阶段不做完整生产级多实例部署；本地和单机 API 优先使用 SQLite。后续如果要部署到多进程、多容器或云服务，再切换 Postgres。

第一阶段不把所有历史病历和隐私信息完整暴露给前端恢复卡片，只展示恢复决策所需的最小上下文。

## 当前问题

当前 Runner 内部仍有运行时内存标记：

```python
self._is_first_turn
self._pending_interrupt
self._pending_interrupt_payload
```

这些字段在服务重启后会丢失。即使 LangGraph checkpoint 已经写入数据库，如果 Runner 仍只依赖这些内存字段，也无法正确判断下一条用户输入应该：

- 作为 `Command(resume=...)` 交回 interrupt；
- 作为增量 `messages` 继续旧图；
- 还是新建一轮 `initial_state`。

因此持久化方案必须同时改造 checkpointer、Runner 状态检查、API 会话恢复提示和显式关闭/重开行为。

## 目标状态模型

后端需要能识别以下状态：

| 状态 | 含义 | 用户下一步 |
| --- | --- | --- |
| `empty` | 没有 checkpoint 或 checkpoint 已清理 | 开始新分诊 |
| `interrupted` | 图停在 LangGraph interrupt，等待用户回答 | 回答上次问题，或重新开始/结束 |
| `active` | 有未完成图状态，但当前没有 pending interrupt | 继续补充症状或进入下一步 |
| `completed` | 已生成分诊结果 | 默认不恢复旧图，可开启新分诊 |
| `cancelled` | 用户明确结束 | 不恢复 |
| `expired` | 超过可恢复时间窗口 | 建议重新开始 |

## 技术实现方案

### 1. 增加持久化 checkpointer 依赖

开发和本地运行优先使用 SQLite：

```toml
"langgraph-checkpoint-sqlite",
```

生产环境后续可增加：

```toml
"langgraph-checkpoint-postgres",
```

SQLite 路径建议：

```text
data/langgraph_checkpoints.sqlite
```

### 2. 新增 CheckpointerProvider

新增模块：

```text
medi/core/langgraph_checkpoint.py
```

职责：

- 根据环境变量选择 backend：`memory` / `sqlite` / `postgres`。
- 在 API lifespan 启动时打开连接。
- 在应用关闭时释放连接。
- 向 `TriageGraphRunner` 提供同一个 checkpointer 实例。

建议环境变量：

```text
MEDI_CHECKPOINTER=sqlite
MEDI_CHECKPOINT_DB=data/langgraph_checkpoints.sqlite
MEDI_CHECKPOINT_POSTGRES_URL=postgresql://...
```

### 3. Runner 注入 checkpointer

`TriageGraphRunner` 不再直接固定创建 `MemorySaver`：

```python
def __init__(..., checkpointer=None):
    self._checkpointer = checkpointer or MemorySaver()
```

`session_store.py` 创建 Runner 时注入 provider：

```python
agent = TriageGraphRunner(
    ctx=ctx,
    bus=bus,
    router=_router,
    checkpointer=checkpoint_provider.get(),
)
```

### 4. Runner 检查 thread 当前状态

新增：

```python
async def inspect_thread_state(self) -> GraphThreadStatus:
    graph = self._build_graph(history_prompt="")
    config = {"configurable": {"thread_id": self._ctx.session_id}}
    snapshot = await graph.aget_state(config)

    if not snapshot or not snapshot.values:
        return GraphThreadStatus.EMPTY

    if snapshot.interrupts:
        return GraphThreadStatus.INTERRUPTED

    values = snapshot.values
    if values.get("triage_output") is not None:
        return GraphThreadStatus.COMPLETED

    return GraphThreadStatus.ACTIVE
```

`handle()` 输入选择逻辑：

```python
status = await self.inspect_thread_state()

if status == GraphThreadStatus.INTERRUPTED:
    input_data = Command(resume=user_input)
elif status == GraphThreadStatus.ACTIVE:
    input_data = {"messages": [{"role": "user", "content": user_input}]}
else:
    input_data = initial_state
```

这使系统可以恢复任意未完成阶段，而不只恢复 interrupt。

### 5. 分诊完成、取消、重新开始时清理 checkpoint

当前 `reset_graph_state()` 只重置内存字段。持久化后要支持删除当前 thread：

```python
async def reset_graph_state(self, *, clear_checkpoint: bool = True) -> None:
    self._is_first_turn = True
    self._pending_interrupt = False
    self._pending_interrupt_payload = None
    self._cached_symptom_summary = ""

    if clear_checkpoint and hasattr(self._checkpointer, "adelete_thread"):
        await self._checkpointer.adelete_thread(self._ctx.session_id)
```

调用点：

- emergency 红旗拦截后；
- output 完成后；
- 用户点击重新开始；
- 用户点击结束本次分诊；
- session 过期清理任务。

## 用户恢复上下文设计

只展示一句“上次问题”是不够的。用户可能隔了很久回来，不一定记得为什么要回答，也不一定知道自己上一轮说了什么。

建议引入 `ResumeContext`，分为默认短卡片和可展开详情。

### 默认短卡片

恢复时默认展示：

```text
检测到你有一个未完成的分诊。

上次问到：
这个症状是什么时候开始的？

已了解：
你提到主要不适是头痛，伴有恶心，目前还缺起病时间。

继续回答 / 查看详情 / 重新开始 / 结束本次分诊
```

默认卡片包括：

- pending question：上次等待回答的问题。
- one-line clinical summary：已采集核心摘要。
- why asking：为什么还需要这个信息。
- elapsed time：距离上次会话多久。
- actions：继续回答、查看详情、重新开始、结束。

### 展开详情

用户点击“查看详情”后展示：

```text
本次分诊进度
- 主诉：头痛
- 起病时间：待补充
- 伴随症状：恶心
- 危险信号：尚未完全确认
- 用药/过敏：待补充

最近对话
患者：我头痛，还有点恶心
护士：这个症状是什么时候开始的？
```

展开详情包括：

- 最近 3-5 轮对话；
- 已采集事实摘要；
- 当前缺失任务；
- 当前分诊阶段；
- 过期提示；
- 隐私提示。

### ResumeContext 建议结构

```python
class ResumeContext(TypedDict, total=False):
    status: str
    session_id: str
    pending_question: str | None
    pending_task: str | None
    why_needed: str | None
    collected_summary: str
    missing_summary: list[str]
    recent_messages: list[dict]
    last_updated_at: str | None
    expires_at: str | None
    recommended_action: str
    actions: list[str]
```

示例响应：

```json
{
  "status": "interrupted",
  "pending_question": "这个症状是什么时候开始的？",
  "pending_task": "T2_ONSET",
  "why_needed": "起病时间会帮助医生判断急性程度和可能病因。",
  "collected_summary": "已了解：主要不适为头痛，伴恶心。",
  "missing_summary": ["起病时间", "严重程度", "用药/过敏"],
  "recent_messages": [
    {"role": "user", "content": "我头痛，还有点恶心"},
    {"role": "assistant", "content": "这个症状是什么时候开始的？"}
  ],
  "recommended_action": "continue",
  "actions": ["continue", "details", "restart", "close"]
}
```

## API 设计

### 查询恢复状态

```text
GET /chat/session/{session_id}/resume-state
```

返回 `ResumeContext`。

前端进入页面时先调用该接口。如果状态是 `interrupted` 或 `active`，展示恢复卡片。

### 继续会话

```text
POST /chat
```

如果用户选择继续，后续消息仍走现有 chat 接口。后端根据 checkpoint 状态自动决定是 `Command(resume=...)` 还是增量 messages。

### 重新开始

```text
POST /chat/session/{session_id}/restart
```

行为：

- 删除当前 LangGraph thread checkpoint。
- 当前 encounter 标记 `cancelled` 或 `restarted`。
- 清理 Runner runtime flags。
- 后续第一条症状重新创建 initial_state。

### 结束本次分诊

```text
POST /chat/session/{session_id}/close
```

行为：

- encounter 标记 `cancelled` 或 `closed`。
- 删除 checkpoint。
- 返回确认结果。

## 过期策略

医疗分诊不能无限恢复，因为症状会变化。

建议策略：

| 间隔 | 行为 |
| --- | --- |
| 2 小时内 | 默认展示“继续上次分诊” |
| 2-24 小时 | 展示恢复卡片，同时提醒症状可能变化 |
| 超过 24 小时 | 建议重新开始，仍允许查看旧摘要 |
| 超过 7 天 | 自动归档或清理 checkpoint |

过期提示：

```text
上次分诊已经过去较久，症状可能发生变化。为了判断更准确，建议重新开始一次分诊。
```

## 前端/CLI 交互建议

### 前端

打开聊天页：

1. 如果没有 `session_id`，正常开始。
2. 如果有 `session_id`，先请求 `/resume-state`。
3. 如果返回 `empty/completed/cancelled`，正常展示新会话。
4. 如果返回 `interrupted/active`，展示恢复卡片。

恢复卡片按钮：

- 继续回答：输入框聚焦，提示“请回答上次问题”。
- 查看详情：展开最近对话和已采集摘要。
- 重新开始：调用 restart。
- 结束本次分诊：调用 close。

### CLI

启动时如果检测到未完成分诊：

```text
检测到未完成分诊。
上次问到：这个症状是什么时候开始的？
已了解：主要不适为头痛，伴恶心。

输入回答继续，输入 /details 查看详情，输入 /restart 重新开始，输入 /close 结束。
```

## 测试计划

### 单元测试

1. `inspect_thread_state()` 能识别 empty/interrupted/active/completed。
2. interrupted 状态下 `handle()` 使用 `Command(resume=...)`。
3. active 状态下 `handle()` 使用增量 messages。
4. completed/cancelled 状态不会继续旧图。
5. `reset_graph_state(clear_checkpoint=True)` 删除 thread checkpoint。
6. `build_resume_context()` 返回 pending question、摘要、最近对话和 actions。

### 集成测试

模拟服务重启：

1. 使用临时 SQLite checkpoint DB。
2. 创建 Runner A，运行到 interrupt。
3. 丢弃 Runner A 和 Session。
4. 创建 Runner B，使用同一个 `session_id` 和同一个 DB。
5. 查询 resume-state，能拿到 pending question。
6. 用户回答后，Runner B 能 `Command(resume=...)`。
7. 验证不会重复发同一个追问。
8. 验证图继续进入 intake 并更新事实。

### API 测试

1. `/resume-state` 对 interrupted 返回恢复卡片数据。
2. `/restart` 会清理 checkpoint。
3. `/close` 会清理 checkpoint。
4. 过期会话返回建议重新开始。

## 风险与注意事项

1. SQLite 不适合高并发生产环境；生产应切 Postgres。
2. 恢复卡片不应暴露过多敏感医疗信息，默认只显示摘要。
3. 如果前端保留旧聊天记录，恢复卡片可以更短；如果用户换设备回来，必须提供更多上下文。
4. 分诊完成后要清理或归档 checkpoint，否则新症状可能误接旧图。
5. 用户输入“重新开始”“结束”等命令时，要优先处理，不应当作 pending question 的答案。
6. checkpoint 是图状态，不等于业务工单状态；长期最好把 encounter 状态也持久化到数据库。

## 分阶段落地

### Phase 1：技术恢复

- SQLite checkpointer provider。
- Runner 注入 checkpointer。
- `inspect_thread_state()`。
- interrupted/active/empty 三分支恢复。
- reset 删除 checkpoint。
- 模拟重启集成测试。

### Phase 2：用户恢复体验

- `/resume-state`。
- `ResumeContext` 构建。
- CLI 恢复提示。
- 前端恢复卡片。
- restart/close API。

### Phase 3：生产化

- Postgres checkpointer。
- encounter 状态持久化。
- checkpoint 过期归档。
- time travel / state history 调试页。
- 审计日志和恢复事件追踪。

## 项目亮点表达

可以在 README 或面试材料中描述为：

> Medi 使用 LangGraph 持久化 checkpoint 管理长运行分诊图，以 `session_id` 作为 `thread_id` 保存每一步状态。用户关闭页面、网络中断或服务重启后，系统可以恢复未完成问诊，并通过恢复卡片展示上次问题、已采集摘要和可选动作；只有用户明确结束或重新开始时才清理 checkpoint。这让医疗预问诊具备类似客服系统的会话恢复、关闭确认、可审计和可回放能力。
