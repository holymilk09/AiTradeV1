"""FastAPI app factory for the bot dashboard.

Phase 7 rebuild: multi-tab dashboard with side nav. Each tab is a
server-rendered Jinja template extending ``_layout.html``. Live tab
auto-refreshes via meta-refresh. Chat tab posts user input to
``/chat`` which calls Claude with the bot's current state as context.

Tabs:
  /overview   — KPIs, today, positions, decisions, round-trips
  /live       — market regime, sector heatmap, candidate board (auto-refresh)
  /strategies — registered strategies + per-strategy realized stats
  /screener   — latest discovery + pattern hits
  /history    — full round-trip table with filters
  /logs       — raw event timeline with type filter
  /chat       — Claude conversation with bot state injected per turn

Backwards-compat: ``/`` redirects to ``/overview``; ``/api/status`` and
``/healthz`` unchanged so existing callers + fly.io health checks keep
working.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.dashboard import state as dash
from aitrade.dashboard.auth import require_auth

_TEMPLATES_DIR = Path(__file__).parent / "templates"
# In-memory chat history. Process-local; cleared on restart.
# Each entry: {"role": "user"|"assistant", "content": "..."}
_CHAT_HISTORY: list[dict[str, str]] = []
_CHAT_MAX_TURNS = 40

# Map common event types to a pill-color class for the Logs tab.
_EVENT_PILL_CLASS = {
    "order_submitted": "info",
    "order_ack": "ok",
    "order_filled": "ok",
    "order_canceled": "warn",
    "order_rejected": "warn",
    "risk_blocked": "warn",
    "round_trip_closed": "ok",
    "stale_data_skip": "warn",
    "time_gated_skip": "warn",
    "deep_dig": "info",
    "stop_plan": "info",
    "floor_trader_decision": "info",
    "candidate_board": "",
    "market_snapshot": "",
    "discovery_scan": "",
    "pattern_detected": "",
    "empty_discovery": "warn",
    "run_start": "ok",
    "run_end": "",
}


def _payload_preview(payload: dict[str, Any], *, max_len: int = 220) -> str:
    """Shrink a journal payload to a single-line preview for the logs table."""
    try:
        text = json.dumps(payload, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(payload)
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


def build_app(*, settings: Settings | None = None, broker_factory: Any = None) -> FastAPI:
    """Assemble the dashboard. ``broker_factory`` is a zero-arg callable that
    returns a ``BrokerClient`` (or None) — injected so tests can stub it
    without touching Alpaca creds.
    """
    s = settings or get_settings()

    app = FastAPI(title="aitrade dashboard", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))
    templates.env.filters["money"] = dash.fmt_money
    templates.env.filters["pct"] = dash.fmt_pct
    templates.env.filters["ago"] = dash.fmt_time_ago
    templates.env.globals["event_pill_class"] = _EVENT_PILL_CLASS.get
    templates.env.globals["payload_preview"] = _payload_preview

    def _broker():  # type: ignore[no-untyped-def]
        if broker_factory is None:
            try:
                from aitrade.brokers.alpaca import build_client

                return build_client(settings=s)
            except Exception:  # pragma: no cover — broker init noise
                return None
        return broker_factory()

    def _shared_context() -> dict[str, Any]:
        """Header context every page uses (account pill, kill switch state)."""
        broker = _broker()
        return {
            "account": dash.fetch_account(broker),
            "kill_switch_active": dash.kill_switch_active(),
        }

    # ----- backwards-compat: / and /api/status / /healthz -------------------

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/overview", status_code=302)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        # Unauthenticated: used by fly.io / load-balancer health checks.
        return {"status": "ok"}

    # ----- Overview ---------------------------------------------------------

    @app.get("/overview", response_class=HTMLResponse)
    def overview(
        request: Request,
        _user: str = Depends(require_auth),
    ) -> Response:
        broker = _broker()
        account = dash.fetch_account(broker)
        positions = dash.fetch_positions(broker)
        round_trips = dash.fetch_round_trips(s.aitrade_log_dir, limit=20)
        today = dash.fetch_today_card(s.aitrade_log_dir)
        recent_decisions = dash.fetch_recent_decisions(s.aitrade_log_dir, limit=8)
        total_pnl, wins, losses = dash.aggregate_pnl(round_trips)
        return templates.TemplateResponse(
            request,
            "overview.html",
            {
                "active_tab": "overview",
                "account": account,
                "kill_switch_active": dash.kill_switch_active(),
                "positions": positions,
                "round_trips": round_trips,
                "today": today,
                "recent_decisions": recent_decisions,
                "total_pnl": total_pnl,
                "wins": wins,
                "losses": losses,
            },
        )

    # ----- Live -------------------------------------------------------------

    @app.get("/live", response_class=HTMLResponse)
    def live(
        request: Request,
        _user: str = Depends(require_auth),
    ) -> Response:
        snapshot = dash.fetch_latest_market_snapshot(s.aitrade_log_dir)
        built_at, candidates = dash.fetch_latest_candidate_board(
            s.aitrade_log_dir, limit=20
        )
        return templates.TemplateResponse(
            request,
            "live.html",
            {
                "active_tab": "live",
                **_shared_context(),
                "snapshot": snapshot,
                "candidates": candidates,
                "board_built_at": built_at,
            },
        )

    # ----- Strategies -------------------------------------------------------

    @app.get("/strategies", response_class=HTMLResponse)
    def strategies(
        request: Request,
        _user: str = Depends(require_auth),
    ) -> Response:
        from aitrade.strategy.registry import list_strategies

        registered = [
            (name, _strategy_blurb(name)) for name in list_strategies()
        ]
        stats = dash.fetch_strategy_stats(s.aitrade_log_dir)
        active = {row.strategy_id for row in stats}
        return templates.TemplateResponse(
            request,
            "strategies.html",
            {
                "active_tab": "strategies",
                **_shared_context(),
                "registered": registered,
                "active_strategies": active,
                "stats": stats,
            },
        )

    # ----- Screener ---------------------------------------------------------

    @app.get("/screener", response_class=HTMLResponse)
    def screener(
        request: Request,
        _user: str = Depends(require_auth),
    ) -> Response:
        # Latest discovery scan event
        scans = dash.fetch_recent_events(
            s.aitrade_log_dir, event_type="discovery_scan", limit=1
        )
        last_discovery_ts = scans[0].timestamp if scans else None
        discovered_tickers: list[dict[str, object]] = []
        if scans:
            raw = scans[0].payload.get("tickers", [])
            if isinstance(raw, list):
                discovered_tickers = [
                    {
                        "symbol": t.get("symbol"),
                        "buzz_score": float(t.get("buzz_score", 0.0) or 0.0),
                    }
                    for t in raw
                    if isinstance(t, dict)
                ]
        # Pattern hits from the same cycle (most recent N pattern_detected events)
        pattern_events = dash.fetch_recent_events(
            s.aitrade_log_dir, event_type="pattern_detected", limit=20
        )
        return templates.TemplateResponse(
            request,
            "screener.html",
            {
                "active_tab": "screener",
                **_shared_context(),
                "last_discovery_ts": last_discovery_ts,
                "discovered_tickers": discovered_tickers,
                "pattern_events": pattern_events,
            },
        )

    # ----- History ----------------------------------------------------------

    @app.get("/history", response_class=HTMLResponse)
    def history(
        request: Request,
        symbol: str | None = None,
        pattern: str | None = None,
        bucket: str | None = None,
        regime: str | None = None,
        _user: str = Depends(require_auth),
    ) -> Response:
        rows = dash.fetch_filtered_round_trips(
            s.aitrade_log_dir,
            symbol=symbol or None,
            pattern=pattern or None,
            bucket=bucket or None,
            regime=regime or None,
            limit=200,
        )
        known_regimes = [
            "risk_on_low_vol",
            "risk_on_high_vol",
            "risk_off_low_vol",
            "risk_off_high_vol",
        ]
        return templates.TemplateResponse(
            request,
            "history.html",
            {
                "active_tab": "history",
                **_shared_context(),
                "rows": rows,
                "filters": {
                    "symbol": symbol,
                    "pattern": pattern,
                    "bucket": bucket,
                    "regime": regime,
                },
                "known_regimes": known_regimes,
                "limit": 200,
            },
        )

    # ----- Logs -------------------------------------------------------------

    @app.get("/logs", response_class=HTMLResponse)
    def logs(
        request: Request,
        event_type: str | None = None,
        _user: str = Depends(require_auth),
    ) -> Response:
        events = dash.fetch_recent_events(
            s.aitrade_log_dir,
            event_type=event_type or None,
            limit=200,
        )
        types = dash.fetch_event_types(s.aitrade_log_dir)
        return templates.TemplateResponse(
            request,
            "logs.html",
            {
                "active_tab": "logs",
                **_shared_context(),
                "events": events,
                "event_types": types,
                "filter_type": event_type,
                "limit": 200,
            },
        )

    # ----- Chat -------------------------------------------------------------

    @app.get("/chat", response_class=HTMLResponse)
    def chat(
        request: Request,
        _user: str = Depends(require_auth),
    ) -> Response:
        return templates.TemplateResponse(
            request,
            "chat.html",
            {
                "active_tab": "chat",
                **_shared_context(),
                "history": list(_CHAT_HISTORY),
                "has_api_key": bool(s.anthropic_api_key.get_secret_value()),
            },
        )

    @app.post("/chat")
    def chat_post(
        message: str = Form(...),
        _user: str = Depends(require_auth),
    ) -> RedirectResponse:
        text = (message or "").strip()
        if not text:
            return RedirectResponse(url="/chat", status_code=303)
        if not s.anthropic_api_key.get_secret_value():
            return RedirectResponse(url="/chat", status_code=303)
        _CHAT_HISTORY.append({"role": "user", "content": text})
        try:
            answer = _ask_claude(text, s, _broker)
        except Exception as e:  # pragma: no cover — chat is best-effort
            logger.warning("chat call failed: {}", e)
            answer = f"_(Chat error: {type(e).__name__}.)_"
        _CHAT_HISTORY.append({"role": "assistant", "content": answer})
        # Trim to bounded history.
        if len(_CHAT_HISTORY) > _CHAT_MAX_TURNS:
            del _CHAT_HISTORY[: len(_CHAT_HISTORY) - _CHAT_MAX_TURNS]
        return RedirectResponse(url="/chat", status_code=303)

    @app.post("/chat/clear")
    def chat_clear(_user: str = Depends(require_auth)) -> RedirectResponse:
        _CHAT_HISTORY.clear()
        return RedirectResponse(url="/chat", status_code=303)

    # ----- Controls ---------------------------------------------------------

    @app.post("/control/halt")
    def halt(_user: str = Depends(require_auth)) -> RedirectResponse:
        dash.set_kill_switch(True)
        return RedirectResponse(url="/overview", status_code=303)

    @app.post("/control/resume")
    def resume(_user: str = Depends(require_auth)) -> RedirectResponse:
        dash.set_kill_switch(False)
        return RedirectResponse(url="/overview", status_code=303)

    @app.post("/control/flatten")
    def flatten(_user: str = Depends(require_auth)) -> RedirectResponse:
        broker = _broker()
        if broker is not None:
            with contextlib.suppress(Exception):  # pragma: no cover — best-effort
                broker.cancel_all()
            with contextlib.suppress(Exception):  # pragma: no cover — best-effort
                from aitrade.execution.orders import (
                    OrderRequest,
                    OrderType,
                    Side,
                    TimeInForce,
                )

                for p in broker.get_positions():
                    if p.qty == 0:
                        continue
                    side = Side.SELL if p.qty > 0 else Side.BUY
                    broker.submit_order(
                        OrderRequest(
                            symbol=p.symbol,
                            side=side,
                            qty=abs(p.qty),
                            order_type=OrderType.MARKET,
                            time_in_force=TimeInForce.DAY,
                            strategy_id="dashboard_flatten",
                        )
                    )
        # Halt new orders too — flatten is intentional, not a hiccup.
        dash.set_kill_switch(True)
        return RedirectResponse(url="/overview", status_code=303)

    @app.post("/control/reconcile")
    def reconcile(_user: str = Depends(require_auth)) -> RedirectResponse:
        with contextlib.suppress(Exception):  # pragma: no cover
            dash.reconcile_now(s.aitrade_log_dir)
        return RedirectResponse(url="/overview", status_code=303)

    # ----- API JSON (kept for backwards-compat) -----------------------------

    @app.get("/api/status")
    def status_json(_user: str = Depends(require_auth)) -> dict[str, Any]:
        broker = _broker()
        account = dash.fetch_account(broker)
        positions = dash.fetch_positions(broker)
        today = dash.fetch_today_card(s.aitrade_log_dir)
        return {
            "account": (
                {
                    "cash": account.cash,
                    "equity": account.equity,
                    "buying_power": account.buying_power,
                    "is_paper": account.is_paper,
                    "currency": account.currency,
                }
                if account
                else None
            ),
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": p.qty,
                    "avg_entry_price": p.avg_entry_price,
                    "market_value": p.market_value,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in positions
            ],
            "today": {
                "n_decisions": today.n_decisions,
                "n_orders": today.n_orders,
                "n_round_trips": today.n_round_trips,
                "realized_pnl_today_usd": today.realized_pnl_today_usd,
            },
            "kill_switch_active": dash.kill_switch_active(),
        }

    return app


# ----- Helpers ------------------------------------------------------------------


_BLURBS: dict[str, str] = {
    "sma_crossover": (
        "Long when fast SMA crosses above slow; flat below. "
        "Trend follower; one symbol at a time."
    ),
}


def _strategy_blurb(name: str) -> str:
    return _BLURBS.get(name, "—")


_CHAT_SYSTEM_PROMPT = (
    "You are aitrade-bot's in-dashboard assistant. The user sees the dashboard "
    "and chats with you about their paper-trading bot. Be concise, plain-language, "
    "and reference the bot state JSON included in each turn — account, open "
    "positions, recent decisions, recent round-trips, market snapshot. Do NOT "
    "invent numbers. If asked about something not in the state, say so. Never "
    "give live-trading recommendations beyond the existing paper-mode guardrails."
)


def _build_chat_state_blob(s: Settings, broker_fn: Any) -> dict[str, Any]:
    """Assemble the bot-state JSON injected into every chat turn."""
    broker = broker_fn() if callable(broker_fn) else None
    account = dash.fetch_account(broker)
    positions = dash.fetch_positions(broker)
    decisions = dash.fetch_recent_decisions(s.aitrade_log_dir, limit=5)
    round_trips = dash.fetch_round_trips(s.aitrade_log_dir, limit=5)
    snapshot = dash.fetch_latest_market_snapshot(s.aitrade_log_dir)
    return {
        "fetched_at": time.time(),
        "account": (
            {
                "cash": account.cash,
                "equity": account.equity,
                "buying_power": account.buying_power,
                "is_paper": account.is_paper,
                "currency": account.currency,
            }
            if account
            else None
        ),
        "positions": [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "avg_entry_price": p.avg_entry_price,
                "market_value": p.market_value,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ],
        "recent_decisions": decisions,
        "recent_round_trips": [
            {
                "symbol": r.symbol,
                "qty": r.qty,
                "entry_price": r.entry_price,
                "exit_price": r.exit_price,
                "pnl_usd": r.pnl_usd,
                "pnl_pct": r.pnl_pct,
                "pnl_bucket": r.pnl_bucket,
                "exit_reason": r.exit_reason,
                "thesis": r.thesis,
            }
            for r in round_trips
        ],
        "market_snapshot": snapshot,
        "kill_switch_active": dash.kill_switch_active(),
    }


def _ask_claude(user_text: str, s: Settings, broker_fn: Any) -> str:
    """One-shot Claude call. Synchronous — keeps the route simple.

    The system prompt is stable and prompt-cache-eligible; the bot state
    blob is injected as the first part of the user message so it doesn't
    invalidate the cache when chat history doesn't change.
    """
    import anthropic

    client = anthropic.Anthropic(
        api_key=s.anthropic_api_key.get_secret_value(),
    )
    state = _build_chat_state_blob(s, broker_fn)
    user_message = (
        "Bot state (JSON):\n"
        + json.dumps(state, default=str, indent=2)
        + "\n\nUser message:\n"
        + user_text
    )
    response = client.messages.create(
        model=s.aitrade_reasoner_model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": _CHAT_SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_message}],
    )
    parts: list[str] = []
    for block in getattr(response, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts).strip() or "_(Empty response.)_"
