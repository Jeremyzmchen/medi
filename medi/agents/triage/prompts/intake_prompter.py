"""Prompt builders for task-focused intake follow-up questions."""

from __future__ import annotations

import json

from medi.agents.triage.clinical_facts import ClinicalFactStore
from medi.agents.triage.intake_protocols import ResolvedIntakePlan
from medi.agents.triage.task_definitions import TASK_BY_ID


PROMPTER_SYSTEM_PROMPT = """你是一位专业的预诊护士，正在通过对话为患者做就诊前信息采集。

你的任务：根据当前选中的采集任务、已知患者信息和对话历史，生成**一个**自然、专业的追问。

要求：
- 只问一个问题，不要一次问多个
- 围绕当前采集任务提问，不要向患者暴露任务编号或技术字段名
- 语气温和、有共情，像真正的护士说话
- 如果患者上一轮的回答相关但不完整，先承认他说的，再追问缺失部分
- 不要给出诊断、建议或安慰
- 用中文回答，直接输出问题文本，不要有任何前缀或解释
"""


def build_prompter_input(
    *,
    task_id: str | None,
    task_instruction: str | None,
    plan: ResolvedIntakePlan,
    store: ClinicalFactStore,
    preconsultation_record: dict,
    task_progress: dict[str, dict],
    messages: list[dict],
) -> str:
    known_facts = store.prompt_context()
    recent_messages = messages[-6:] if len(messages) > 6 else messages
    conv_lines = []
    for message in recent_messages:
        role = "患者" if message.get("role") == "user" else "护士"
        content = str(message.get("content", "")).strip()
        if content:
            conv_lines.append(f"{role}：{content}")
    conv_text = "\n".join(conv_lines) if conv_lines else "（对话刚开始）"

    task = TASK_BY_ID.get(task_id or "")
    task_label = task.label if task else (task_id or "当前采集任务")
    task_desc = task.description if task else ""
    progress = task_progress.get(task_id or "", {})
    missing = [
        detail.get("description") or detail.get("id")
        for detail in progress.get("requirement_details") or []
        if not detail.get("completed")
    ]

    record_text = json.dumps(preconsultation_record, ensure_ascii=False, indent=2)
    if len(record_text) > 1800:
        record_text = record_text[:1800] + "\n..."

    return f"""【当前采集任务】
任务：{task_label}
任务目标：{task_desc or task_instruction or "补齐医生预诊所需信息"}
调度说明：{task_instruction or "根据当前任务完成度继续追问"}
任务完成度：{progress.get("score", 0)}，状态：{progress.get("status", "pending")}
当前缺口：{"；".join(missing) if missing else "由当前任务目标决定"}

【当前已知患者信息】
{known_facts}

【预诊档案视图】
{record_text}

【最近对话记录】
{conv_text}

请生成一个围绕当前采集任务的自然追问。"""

