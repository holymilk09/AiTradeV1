"""Halt-engine state endpoints.

Halt state lives in Redis so Executor can consult it on every submit without
a DB round-trip. Transitions (HALT_FIRED, HALT_RESUMED) are written to the
journal by the strategy_runner when a rule trips. This router exposes the
manual Halt / Flatten&Halt / Resume controls from the header.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from aitrade.api.redis_client import redis
from aitrade.api.settings import get_settings

router = APIRouter(prefix="/halt", tags=["halt"])

HaltScope = Literal["live", "halt_new", "halt_all"]


class HaltState(BaseModel):
    state: HaltScope
    reason: str | None = None
    since: str | None = None
    tripped_rules: list[str] = []


async def _read_state() -> HaltState:
    raw = await redis().get(get_settings().halt_state_key)
    if not raw:
        return HaltState(state="live")
    return HaltState.model_validate_json(raw)


async def _write_state(s: HaltState) -> None:
    await redis().set(get_settings().halt_state_key, s.model_dump_json())


@router.get("")
async def get_halt() -> HaltState:
    return await _read_state()


class HaltBody(BaseModel):
    scope: Literal["halt_new", "halt_all"]
    reason: str
    confirm_flatten: bool = False


@router.post("")
async def set_halt(body: HaltBody) -> HaltState:
    if body.scope == "halt_all" and not body.confirm_flatten:
        raise HTTPException(status_code=400, detail="halt_all requires confirm_flatten=true")
    state = HaltState(
        state=body.scope,
        reason=body.reason,
        since=datetime.now(UTC).isoformat(),
        tripped_rules=["manual"],
    )
    await _write_state(state)
    # NB: actual order-cancel + flatten is the strategy_runner's job once it
    # observes the state flip; this endpoint is the trigger only.
    return state


class ResumeBody(BaseModel):
    reason: str
    force: bool = False


@router.post("/resume")
async def resume(body: ResumeBody) -> HaltState:
    cur = await _read_state()
    # If non-manual rules tripped, force is required to resume.
    auto_rules = [r for r in cur.tripped_rules if r != "manual"]
    if auto_rules and not body.force:
        raise HTTPException(
            status_code=409,
            detail={"msg": "auto rules still tripped", "rules": auto_rules},
        )
    new = HaltState(state="live", reason=body.reason, since=datetime.now(UTC).isoformat())
    await _write_state(new)
    return new


# Internal helper used by the strategy_runner; not a route.
async def trip_rule(rule_name: str, scope: HaltScope, reason: str) -> HaltState:
    cur = await _read_state()
    rules = sorted(set(cur.tripped_rules + [rule_name]))
    state = HaltState(
        state=scope,
        reason=reason,
        since=cur.since or datetime.now(UTC).isoformat(),
        tripped_rules=rules,
    )
    await _write_state(state)
    return state


__all__ = ["router", "trip_rule", "HaltState"]
