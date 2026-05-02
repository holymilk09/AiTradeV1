"""FastAPI app factory for the bot dashboard.

The factory takes a settings object so tests can inject a non-default
config (e.g. a fixed password). Routes live in this module too — keeping
the dashboard small enough that fanning across files would obscure more
than it'd reveal.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from aitrade.config import Settings, get_settings
from aitrade.dashboard import state as dash
from aitrade.dashboard.auth import require_auth

_TEMPLATES_DIR = Path(__file__).parent / "templates"


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

    def _broker():  # type: ignore[no-untyped-def]
        if broker_factory is None:
            try:
                from aitrade.brokers.alpaca import build_client

                return build_client(settings=s)
            except Exception:  # pragma: no cover — broker init noise
                return None
        return broker_factory()

    @app.get("/", response_class=HTMLResponse)
    def home(
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
        kill = dash.kill_switch_active()
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "account": account,
                "positions": positions,
                "round_trips": round_trips,
                "today": today,
                "recent_decisions": recent_decisions,
                "total_pnl": total_pnl,
                "wins": wins,
                "losses": losses,
                "kill_switch_active": kill,
            },
        )

    @app.post("/control/halt")
    def halt(_user: str = Depends(require_auth)) -> RedirectResponse:
        dash.set_kill_switch(True)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/control/resume")
    def resume(_user: str = Depends(require_auth)) -> RedirectResponse:
        dash.set_kill_switch(False)
        return RedirectResponse(url="/", status_code=303)

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
        return RedirectResponse(url="/", status_code=303)

    @app.post("/control/reconcile")
    def reconcile(_user: str = Depends(require_auth)) -> RedirectResponse:
        with contextlib.suppress(Exception):  # pragma: no cover
            dash.reconcile_now(s.aitrade_log_dir)
        return RedirectResponse(url="/", status_code=303)

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

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        # Unauthenticated: used by fly.io / load-balancer health checks.
        return {"status": "ok"}

    # Form-bound resume button (browsers send POST from <form>).
    _ = Form  # silence unused-import warning when no form parses are added later
    return app
