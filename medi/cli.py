"""
Medi CLI 入口

用法：
  medi                    # 开始分诊对话（guest 用户）
  medi --user-id u1       # 指定用户（加载健康档案）
  medi observe            # 查看最近请求的可观测性数据
  medi observe --session <id>  # 查看单个 session 的完整链路
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from medi.core.context import UnifiedContext, ModelConfig
from medi.core.stream_bus import AsyncStreamBus, EventType
from medi.core.observability import ObservabilityStore, query_recent_sessions, query_session_detail
from medi.agents.triage.runner import TriageGraphRunner
from medi.agents.triage.department_router import DepartmentRouter
from medi.agents.orchestrator import OrchestratorAgent, Intent
from medi.agents.medication.agent import MedicationAgent
from medi.agents.health_report.agent import HealthReportAgent
from medi.memory.health_profile import HealthProfile, load_profile, save_profile
from medi.memory.episodic import EpisodicMemory

app = typer.Typer(name="medi", help="Medi 智能健康 Agent", invoke_without_command=True)
console = Console()


def _hpi_row(label: str, value) -> None:
    """打印 HPI 字段行，值为 None 时跳过"""
    if value:
        console.print(f"[bold]{label}：[/bold]{value}")


def _hpi_severity_label(value) -> str:
    text = str(value or "")
    if any(marker in text for marker in ("体温", "℃", "度")):
        return "最高体温"
    if "次" in text or "回" in text:
        return "频率/次数"
    return "严重程度"


def _hpi_dict_row(title: str, data, labels: dict[str, str]) -> None:
    if not isinstance(data, dict):
        return
    fields = []
    for key, label in labels.items():
        value = data.get(key)
        if value:
            fields.append(f"{label}：{value}")
    if fields:
        console.print(f"[bold]{title}：[/bold]" + "  ".join(fields))


async def _collect_profile(user_id: str) -> HealthProfile:
    """首次使用时引导用户填写健康档案"""
    console.print("\n[yellow]您是第一次使用 Medi，请先完善您的健康档案（更准确的分诊建议）[/yellow]")
    console.print("[dim]直接回车可跳过某项[/dim]\n")

    profile = HealthProfile(user_id=user_id)

    age_input = console.input("年龄：").strip()
    if age_input.isdigit():
        profile.age = int(age_input)

    gender_input = console.input("性别（男/女）：").strip()
    if gender_input in ("男", "女"):
        profile.gender = gender_input

    allergies_input = console.input("过敏史（如：青霉素、磺胺，多个用逗号分隔）：").strip()
    if allergies_input:
        profile.allergies = [a.strip() for a in allergies_input.replace("，", ",").split(",") if a.strip()]

    chronic_input = console.input("慢性病史（如：高血压、糖尿病，多个用逗号分隔）：").strip()
    if chronic_input:
        profile.chronic_conditions = [c.strip() for c in chronic_input.replace("，", ",").split(",") if c.strip()]

    meds_input = console.input("当前用药（如：二甲双胍，多个用逗号分隔）：").strip()
    if meds_input:
        profile.current_medications = [m.strip() for m in meds_input.replace("，", ",").split(",") if m.strip()]

    await save_profile(profile)
    console.print("\n[green]档案已保存[/green]")
    return profile


async def _chat_loop(user_id: str) -> None:
    """
分诊对话主循环：
1. 生成 session_id，加载健康档案
2. 初始化上下文、可观测集、Agent
3. 打印欢迎信息和最近就诊记录（非游客）
4. handle_turn(user_input)
    """


    # 生成会话id
    session_id = str(uuid.uuid4())[:8]

    # 加载健康档案，首次使用引导填写
    profile = await load_profile(user_id)
    if not profile.is_complete() and user_id != "guest":
        profile = await _collect_profile(user_id)
    elif profile.is_complete():
        console.print(f"\n[dim]已加载健康档案：{profile.gender}，{profile.age}岁[/dim]")

    # 初始化可观测集
    obs = ObservabilityStore()
    # 初始化共享上下文
    ctx = UnifiedContext(
        user_id=user_id,
        session_id=session_id,
        model_config=ModelConfig(),
        # ToolRuntime 校验工具可用
        enabled_tools={"search_symptom_kb", "evaluate_urgency", "get_department_info"},
        health_profile=profile,
        observability=obs,
    )
    # 注册门诊路由agent(rag查找诊室标签)
    router = DepartmentRouter()
    # 注册异步事件收发器
    bus = AsyncStreamBus()
    # 注册意图识别agent
    orchestrator = OrchestratorAgent(ctx=ctx, bus=bus)
    # 注册预诊分诊agent
    triage_agent = TriageGraphRunner(
        ctx=ctx,
        bus=bus,
        router=router,
    )
    # 注册药物咨询agent
    medication_agent = MedicationAgent(ctx=ctx, bus=bus)
    # 注册健康报告解读agent
    health_report_agent = HealthReportAgent(ctx=ctx, bus=bus)

    console.print(f"\n[bold green]Medi 分诊助手[/bold green] (会话 {session_id})")
    console.print("请描述您的问题或症状，输入 [bold]quit[/bold] 退出")

    # 非游客就展示最近就诊记录
    if user_id != "guest":
        episodic = EpisodicMemory(user_id)
        recent = await episodic.recent(limit=5)
        if recent:
            console.print("\n[dim]最近就诊记录：[/dim]")
            for r in recent:
                date_str = r.visit_date.strftime("%m-%d")
                console.print(f"[dim]  {date_str} | {r.department}[/dim]")
                if r.chief_complaint:
                    fields = [line.strip() for line in r.chief_complaint.splitlines() if line.strip()]
                    detail = " | ".join(fields)
                    console.print(f"[dim]       {detail}[/dim]")
    console.print()

    async def handle_turn(user_input: str) -> None:
        # 注入异步信息
        nonlocal bus
        bus = AsyncStreamBus()
        triage_agent._bus = bus  # TriageGraphRunner._bus
        orchestrator._bus = bus
        medication_agent._bus = bus
        health_report_agent._bus = bus
        health_report_agent._diet_agent._bus = bus
        health_report_agent._schedule_agent._bus = bus

        # 控制台消费（打印）
        async def consume() -> None:
            async for event in bus.stream():
                # 1.智能体追问
                if event.type == EventType.FOLLOW_UP:
                    console.print(f"\n[cyan]Medi:[/cyan] {event.data['question']}")
                # 2. 分诊结束，智能体输出报告
                elif event.type == EventType.RESULT:
                    patient_out = event.data.get("patient_output")
                    doctor_hpi = event.data.get("doctor_hpi")

                    if patient_out:
                        # ── 患者端：科室推荐 + 紧急程度 + 建议 ──
                        console.print("\n[bold cyan]── 分诊结果 ──[/bold cyan]")
                        primary = patient_out.get("primary_department")
                        alternatives = patient_out.get("alternative_departments") or []
                        if primary is None and patient_out.get("recommended_departments"):
                            legacy_depts = patient_out.get("recommended_departments") or []
                            primary = legacy_depts[0] if legacy_depts else None
                            alternatives = legacy_depts[1:]
                        if primary:
                            conf = int(primary.get("confidence", 0) * 100)
                            console.print(f"[bold]首选科室：[/bold]{primary['department']}（{conf}%）— {primary.get('reason','')}")
                        if alternatives:
                            console.print("[bold]备选科室：[/bold]")
                            for d in alternatives:
                                conf = int(d.get("confidence", 0) * 100)
                                console.print(f"  • {d['department']}（{conf}%）— {d.get('reason','')}")
                        urgency = patient_out.get("urgency_level", "normal")
                        urgency_map = {"emergency": "[bold red]紧急[/bold red]", "urgent": "[yellow]较急[/yellow]", "normal": "[green]普通[/green]", "watchful": "[dim]观察[/dim]"}
                        console.print(f"[bold]紧急程度：[/bold]{urgency_map.get(urgency, urgency)} — {patient_out.get('urgency_reason','')}")
                        console.print(f"[bold]就医建议：[/bold]{patient_out.get('patient_advice','')}")
                        flags = patient_out.get("red_flags_to_watch") or []
                        if flags:
                            console.print("[bold]危险信号（立即就医）：[/bold]" + "、".join(flags))
                    # TODO: 用户端其他输出，目前精简，包括用药咨询、健康报告分析
                    else:
                        console.print("\n[cyan]Medi:[/cyan]")
                        console.print(Markdown(event.data["content"]))

                    if doctor_hpi:
                        # ── 医生端：HPI 预诊报告 ──
                        console.print("\n[bold magenta]── 医生预诊报告（HPI）──[/bold magenta]")
                        patient_meta = []
                        if doctor_hpi.get("user_id"):
                            patient_meta.append(f"用户ID：{doctor_hpi['user_id']}")
                        if doctor_hpi.get("gender"):
                            patient_meta.append(f"性别：{doctor_hpi['gender']}")
                        if doctor_hpi.get("age") is not None:
                            patient_meta.append(f"年龄：{doctor_hpi['age']}岁")
                        if doctor_hpi.get("consultation_time"):
                            patient_meta.append(f"咨询时间：{doctor_hpi['consultation_time']}")
                        if patient_meta:
                            console.print("[bold]患者信息：[/bold]" + "  ".join(patient_meta))
                        console.print(f"[bold]主诉：[/bold]{doctor_hpi.get('chief_complaint','')}")
                        console.print(f"[bold]HPI 叙述：[/bold]{doctor_hpi.get('hpi_narrative','')}")
                        _hpi_row("发作时间", doctor_hpi.get("onset"))
                        _hpi_row("部位", doctor_hpi.get("location"))
                        _hpi_row("持续时间", doctor_hpi.get("duration"))
                        _hpi_row("性质", doctor_hpi.get("character"))
                        _hpi_row("加重/缓解", doctor_hpi.get("alleviating_aggravating_factors"))
                        _hpi_row("放射痛", doctor_hpi.get("radiation"))
                        _hpi_row("时间特征", doctor_hpi.get("timing"))
                        severity_score = doctor_hpi.get("severity_score")
                        severity_label = _hpi_severity_label(severity_score)
                        if severity_label == "最高体温" and severity_score:
                            severity_score = str(severity_score).replace("最高体温", "").strip("：: ")
                        _hpi_row(severity_label, severity_score)
                        assoc = doctor_hpi.get("associated_symptoms") or []
                        if assoc:
                            console.print(f"[bold]伴随症状：[/bold]{', '.join(assoc)}")
                        neg = doctor_hpi.get("pertinent_negatives") or []
                        if neg:
                            console.print(f"[bold]相关阴性：[/bold]{', '.join(neg)}")
                        _hpi_row("检查/诊断经过", doctor_hpi.get("diagnostic_history"))
                        _hpi_row("治疗经过", doctor_hpi.get("therapeutic_history"))
                        _hpi_dict_row("一般情况", doctor_hpi.get("general_condition"), {
                            "mental_status": "精神/意识",
                            "sleep": "睡眠",
                            "appetite": "食欲",
                            "bowel": "大便",
                            "urination": "小便",
                            "weight_change": "体重",
                        })
                        _hpi_dict_row("既往史", doctor_hpi.get("past_history"), {
                            "disease_history": "疾病史",
                            "immunization_history": "接种史",
                            "surgical_history": "手术史",
                            "trauma_history": "外伤史",
                            "blood_transfusion_history": "输血史",
                            "allergy_history": "过敏史",
                            "current_medications": "长期/当前用药",
                        })
                        meds = doctor_hpi.get("current_medications") or []
                        allerg = doctor_hpi.get("allergies") or []
                        console.print(f"[bold]用药：[/bold]{', '.join(meds) if meds else '无'}  [bold]过敏：[/bold]{', '.join(allerg) if allerg else '无'}")
                        _hpi_dict_row("分诊摘要", doctor_hpi.get("triage_summary"), {
                            "protocol_label": "主题",
                            "primary_department": "首选",
                            "secondary_department": "备选",
                            "reason": "依据",
                        })
                        diffs = doctor_hpi.get("differential_diagnoses") or []
                        if diffs:
                            console.print("[bold]鉴别诊断：[/bold]")
                            for d in diffs:
                                console.print(f"  • [{d.get('likelihood','')}] {d.get('condition','')} — {d.get('reasoning','')}")
                        workup = doctor_hpi.get("recommended_workup") or []
                        if workup:
                            console.print(f"[bold]建议检查：[/bold]{', '.join(workup)}")
                        coverage = doctor_hpi.get("record_coverage") or {}
                        missing = coverage.get("missing_or_unknown") or []
                        if missing:
                            console.print(f"[bold]仍未采集：[/bold]{', '.join(missing)}")
                # 红旗预警报告
                elif event.type == EventType.ESCALATION:
                    console.print(f"\n[bold red]警告:[/bold red] {event.data['reason']}")

        async def produce() -> None:
            """
            1. 基于当前上下文进行意图分类
            2. 按意图调用对应 Agent
            3. Agent 将 FOLLOW_UP / RESULT / ERROR 等事件写入 bus
            4. 关闭 bus，通知 consume 本轮结束
            """
            symptom_summary = triage_agent.symptom_summary()
            intent = await orchestrator.classify_intent(user_input, symptom_summary)

            if intent == Intent.OUT_OF_SCOPE:
                await orchestrator.handle_out_of_scope()
            elif intent == Intent.FOLLOWUP:
                await orchestrator.handle_followup(user_input)
            elif intent == Intent.NEW_SYMPTOM:
                ctx.messages.clear()
                triage_agent.reset_graph_state()
                await triage_agent.handle(user_input)
            elif intent == Intent.MEDICATION:
                await medication_agent.handle(user_input)
            elif intent == Intent.HEALTH_REPORT:
                await health_report_agent.handle(user_input)
            else:
                await triage_agent.handle(user_input)

            await bus.close()

        # 并发处理生产者消费者
        await asyncio.gather(consume(), produce())

        # 每轮结束后 flush trace 到 SQLite
        await obs.flush()

    while True:
        try:
            user_input = console.input(f"\n[bold]{user_id}:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        await handle_turn(user_input)

    console.print("\n[dim]会话结束[/dim]")


async def _show_observe(session_id: Optional[str]) -> None:
    if session_id:
        detail = await query_session_detail(session_id)
        _print_session_detail(session_id, detail)
    else:
        sessions = await query_recent_sessions(limit=10)
        _print_sessions_summary(sessions)


def _print_sessions_summary(sessions: list[dict]) -> None:
    if not sessions:
        console.print("[dim]暂无可观测性数据（先运行 medi 对话）[/dim]")
        return

    table = Table(title="最近 10 个会话")
    table.add_column("Session", style="cyan")
    table.add_column("时间", style="dim")
    table.add_column("LLM 调用", justify="right")
    table.add_column("总 Token", justify="right")
    table.add_column("总耗时(ms)", justify="right")
    table.add_column("降级次数", justify="right", style="yellow")
    table.add_column("错误次数", justify="right", style="red")

    for s in sessions:
        table.add_row(
            s["session_id"],
            s["start_time"][:16],
            str(s["llm_calls"]),
            str(s["total_tokens"] or 0),
            str(s["total_llm_ms"] or 0),
            str(s["fallback_count"] or 0),
            str(s["error_count"] or 0),
        )

    console.print(table)
    console.print("[dim]用 --session <id> 查看单个会话详情[/dim]")


def _print_session_detail(session_id: str, detail: dict) -> None:
    console.print(f"\n[bold]Session {session_id} 链路详情[/bold]\n")

    if detail["stages"]:
        table = Table(title="TAOR 阶段耗时")
        table.add_column("阶段")
        table.add_column("耗时(ms)", justify="right")
        for s in detail["stages"]:
            table.add_row(s["stage"], str(s["latency_ms"]))
        console.print(table)

    if detail["llm_calls"]:
        table = Table(title="LLM 调用记录")
        table.add_column("调用类型")
        table.add_column("供应商")
        table.add_column("降级", justify="center")
        table.add_column("Prompt Token", justify="right")
        table.add_column("Completion Token", justify="right")
        table.add_column("耗时(ms)", justify="right")
        table.add_column("成功", justify="center")
        for c in detail["llm_calls"]:
            table.add_row(
                c.get("call_type", "-"),
                c["provider"],
                "Y" if c["is_fallback"] else "-",
                str(c["prompt_tokens"]),
                str(c["completion_tokens"]),
                str(c["latency_ms"]),
                "Y" if c["success"] else "[red]N[/red]",
            )
        console.print(table)

    if detail["tool_calls"]:
        table = Table(title="工具调用记录")
        table.add_column("工具名")
        table.add_column("耗时(ms)", justify="right")
        table.add_column("成功", justify="center")
        table.add_column("错误")
        for t in detail["tool_calls"]:
            table.add_row(
                t["tool_name"],
                str(t["latency_ms"]),
                "Y" if t["success"] else "[red]N[/red]",
                t["error_msg"] or "-",
            )
        console.print(table)


@app.command()
def chat(
    user_id: str = typer.Option("guest", "--user-id", "-u", help="用户 ID"),
) -> None:
    """开始分诊对话"""
    asyncio.run(_chat_loop(user_id))


@app.command()
def observe(
    session: Optional[str] = typer.Option(None, "--session", "-s", help="查看指定 session 的详情"),
) -> None:
    """查看可观测性数据（LLM 调用、工具调用、阶段耗时）"""
    asyncio.run(_show_observe(session))


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="监听地址"),
    port: int = typer.Option(8000, "--port", "-p", help="监听端口"),
    reload: bool = typer.Option(False, "--reload", help="开发模式：代码变更自动重载"),
) -> None:
    """启动 FastAPI HTTP 服务（支持 SSE 流式对话）"""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]缺少 uvicorn，请运行：pip install uvicorn[/red]")
        raise typer.Exit(1)

    console.print(f"[bold green]Medi API[/bold green] 启动于 http://{host}:{port}")
    console.print(f"[dim]文档：http://{host}:{port}/docs[/dim]")
    uvicorn.run(
        "medi.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    app()
