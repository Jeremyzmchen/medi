"""
文件上传路由

POST /upload/report  — 上传体检报告 PDF，提取文字后直接触发 HealthReportAgent Pipeline

流程：
  1. 接收 PDF 文件（multipart/form-data）
  2. pypdf 提取全部页面文字
  3. 文字过少（<50字）→ 返回 422，提示用户手动输入
  4. 文字正常 → 作为 user_input 传入 HealthReportAgent.handle()
  5. 收集所有事件，返回 JSON 列表（同 POST /chat）

注：SSE 版本（/upload/report/stream）暂不实现，
    上传场景用户等待解析结果，单次响应更简洁。
"""

from __future__ import annotations

import asyncio
import io

from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from pypdf import PdfReader

from medi.api.routes.chat import _event_to_dict
from medi.api.schemas import ChatResponse
from medi.api.session_store import get_or_create_session, rebind_bus
from medi.core.stream_bus import AsyncStreamBus, EventType

router = APIRouter(prefix="/upload", tags=["upload"])

_MIN_TEXT_LENGTH = 50  # 少于此字符数视为无效提取


def _extract_pdf_text(content: bytes) -> str:
    """从 PDF bytes 提取纯文字，按页拼接"""
    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())
    return "\n\n".join(pages)


@router.post("/report", response_model=list[ChatResponse])
async def upload_report(
    file: UploadFile = File(..., description="体检报告 PDF 文件"),
    user_id: str = Query(default="guest"),
    session_id: str | None = Query(default=None),
) -> list[ChatResponse]:
    """
    上传体检报告 PDF，自动解读并返回健康方案。

    - 仅支持 PDF 格式（电子版，含文字层）
    - 扫描件无法提取文字，请手动输入关键指标
    - 返回格式与 POST /chat 一致
    """
    # 校验文件类型
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=415,
                detail="仅支持 PDF 格式，请上传 .pdf 文件",
            )

    content = await file.read()

    # 提取文字
    try:
        text = _extract_pdf_text(content)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"PDF 解析失败：{e}")

    if len(text.strip()) < _MIN_TEXT_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=(
                "无法从该 PDF 提取足够文字（可能是扫描件）。"
                "请手动输入体检关键指标，例如：空腹血糖 7.2，总胆固醇 6.8，血压 145/95。"
            ),
        )

    # 构造给 LLM 的输入：加前缀说明来源
    user_input = f"以下是我的体检报告内容：\n\n{text}"

    # 复用 HealthReportAgent Pipeline
    session = await get_or_create_session(session_id, user_id)
    bus = AsyncStreamBus()
    rebind_bus(session, bus)

    events: list[dict] = []

    async def consume() -> None:
        async for event in bus.stream():
            if event.type == EventType.RESULT:
                events.append(_event_to_dict(
                    "result",
                    event.data.get("content", ""),
                    session.session_id,
                ))
            elif event.type == EventType.ERROR:
                events.append(_event_to_dict(
                    "error",
                    event.data.get("message", "未知错误"),
                    session.session_id,
                ))

    async def produce() -> None:
        await session.health_report_agent.handle(user_input)
        await bus.close()

    await asyncio.gather(consume(), produce())
    await session.obs.flush()

    return [ChatResponse(**e) for e in events]
