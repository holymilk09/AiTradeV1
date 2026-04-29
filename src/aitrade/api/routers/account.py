"""Account toggle endpoints (paper⇄live).

Live requires both ALPACA_LIVE_TRADE=true env AND explicit confirm at the call
site. This endpoint reflects current selection in Redis; the broker factory
in `aitrade.brokers.alpaca.build_client` remains the gate. Selection here is
UI state only — the Executor still consults the gate before any live submit.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aitrade.api.redis_client import redis
from aitrade.api.settings import get_settings

router = APIRouter(prefix="/account", tags=["account"])

_REDIS_KEY = "aitrade:account_selection"


class AccountState(BaseModel):
    selected: Literal["paper", "live"]


@router.get("")
async def get_account() -> AccountState:
    cur = await redis().get(_REDIS_KEY)
    return AccountState(selected=cur or get_settings().default_account)  # type: ignore[arg-type]


class SetAccountBody(BaseModel):
    selected: Literal["paper", "live"]
    confirm: bool = False


@router.post("")
async def set_account(body: SetAccountBody) -> AccountState:
    if body.selected == "live" and not body.confirm:
        raise HTTPException(status_code=400, detail="live selection requires confirm=true")
    await redis().set(_REDIS_KEY, body.selected)
    return AccountState(selected=body.selected)
