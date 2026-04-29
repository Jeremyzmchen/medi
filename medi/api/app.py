"""
FastAPI 应用入口

启动：
  uvicorn medi.api.app:app --reload --port 8000

或通过 CLI：
  medi serve --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from medi.api.routes.chat import router as chat_router
from medi.api.routes.observe import router as observe_router
from medi.api.routes.upload import router as upload_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """应用生命周期：启动时预热，关闭时清理"""
    # 预加载 .env（providers.py 里的 build_*_chain 会读取 API key）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    yield
    # 关闭时可在此做清理（如关闭数据库连接池）


app = FastAPI(
    title="Medi 智能健康 Agent API",
    description=(
        "提供智能分诊对话、用药咨询、体检报告解读服务。\n\n"
        "- `POST /chat` — 单轮对话（JSON 响应）\n"
        "- `GET /chat/stream` — SSE 流式对话\n"
        "- `POST /upload/report` — 上传体检报告 PDF\n"
        "- `GET /observe` — 可观测性数据\n"
    ),
    version="0.4.0",
    lifespan=lifespan,
)

# CORS（开发阶段允许所有来源，生产环境需收窄）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(observe_router)
app.include_router(upload_router)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """健康检查端点，供 k8s liveness probe 使用"""
    return {"status": "ok", "service": "medi-api"}
