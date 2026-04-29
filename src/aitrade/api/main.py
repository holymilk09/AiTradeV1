"""FastAPI app entrypoint. Mounted by `aitrade serve-api`."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aitrade.api.db import engine
from aitrade.api.redis_client import redis
from aitrade.api.routers import account, halt, health


@asynccontextmanager
async def lifespan(_: FastAPI) -> Any:
    engine()  # warm pool
    r = redis()
    await r.ping()
    yield
    await r.aclose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="AiTrade Dashboard API",
        version="0.1.0",
        lifespan=lifespan,
    )
    # Local-only deployment behind Tailscale; permissive CORS for the
    # co-located Next.js dev server is fine.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(account.router)
    app.include_router(halt.router)
    return app


app = create_app()
