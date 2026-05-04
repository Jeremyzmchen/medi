"""LangGraph checkpointer lifecycle management.

The API server owns long-lived persistent checkpointers. Tests and CLI runs can
fall back to memory without requiring the SQLite package to be installed.
"""

from __future__ import annotations

import os
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver


class LangGraphCheckpointerProvider:
    def __init__(self) -> None:
        self._checkpointer: Any | None = None
        self._context: AbstractAsyncContextManager | None = None
        self._backend = "memory"

    async def start(self) -> None:
        if self._checkpointer is not None:
            return

        backend = os.getenv("MEDI_CHECKPOINTER", "sqlite").strip().lower()
        self._backend = backend or "memory"

        if self._backend == "sqlite":
            self._checkpointer = await self._start_sqlite()
            return

        if self._backend == "postgres":
            self._checkpointer = await self._start_postgres()
            return

        self._backend = "memory"
        self._checkpointer = MemorySaver()

    async def stop(self) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)
        self._context = None
        self._checkpointer = None

    def get(self):
        if self._checkpointer is None:
            self._backend = "memory"
            self._checkpointer = MemorySaver()
        return self._checkpointer

    @property
    def backend(self) -> str:
        return self._backend

    async def _start_sqlite(self):
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise RuntimeError(
                "MEDI_CHECKPOINTER=sqlite requires langgraph-checkpoint-sqlite. "
                "Install project dependencies before starting the API."
            ) from exc

        db_path = os.getenv(
            "MEDI_CHECKPOINT_DB",
            "data/langgraph_checkpoints.sqlite",
        )
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._context = AsyncSqliteSaver.from_conn_string(db_path)
        return await self._context.__aenter__()

    async def _start_postgres(self):
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise RuntimeError(
                "MEDI_CHECKPOINTER=postgres requires langgraph-checkpoint-postgres."
            ) from exc

        url = os.getenv("MEDI_CHECKPOINT_POSTGRES_URL")
        if not url:
            raise RuntimeError(
                "MEDI_CHECKPOINT_POSTGRES_URL is required when MEDI_CHECKPOINTER=postgres."
            )
        self._context = AsyncPostgresSaver.from_conn_string(url)
        checkpointer = await self._context.__aenter__()
        setup = getattr(checkpointer, "setup", None)
        if setup is not None:
            maybe_result = setup()
            if hasattr(maybe_result, "__await__"):
                await maybe_result
        return checkpointer


checkpoint_provider = LangGraphCheckpointerProvider()
