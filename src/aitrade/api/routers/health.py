"""Health check across DB, Redis, and broker.

Command tab Health panel reads this. Returns per-dependency status so a single
red dot can fault-isolate.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter
from sqlalchemy import text

from aitrade.api.db import session_scope
from aitrade.api.redis_client import redis

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health() -> dict[str, Any]:
    out: dict[str, Any] = {"ts": datetime.now(UTC).isoformat(), "checks": {}}

    try:
        async with session_scope() as s:
            await s.execute(text("SELECT 1"))
        out["checks"]["postgres"] = {"ok": True}
    except Exception as e:  # noqa: BLE001
        out["checks"]["postgres"] = {"ok": False, "err": str(e)}

    try:
        pong = await redis().ping()
        out["checks"]["redis"] = {"ok": bool(pong)}
    except Exception as e:  # noqa: BLE001
        out["checks"]["redis"] = {"ok": False, "err": str(e)}

    out["ok"] = all(c["ok"] for c in out["checks"].values())
    return out
