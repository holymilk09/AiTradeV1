# Claude guidance for AiTradeV1

## Project shape

Algo trading foundation on Alpaca with NautilusTrader backtesting. Python 3.11, uv,
Typer CLI. See `README.md` for commands.

## Safety invariants — never break these

1. **Default paper.** The broker factory (`aitrade.brokers.alpaca.build_client`) must
   default to paper. Live trading requires BOTH `ALPACA_LIVE_TRADE=true` AND an
   explicit `confirm_live=True` kwarg at the call site. Never remove either gate.
2. **Risk gates run before every order.** `aitrade.execution.risk.RiskGate` must be
   consulted before `executor.submit`. Never bypass it "just for a test" — write a
   fixture instead.
3. **Every order/fill/cancel is logged.** The trade journal is the substrate for
   learning; an unlogged order is a bug.

## Dev loop

```bash
uv run pytest
uv run ruff check
uv run mypy src
```

CI runs all three on push.

## Adding a new strategy

1. Create `src/aitrade/strategy/examples/<name>.py` implementing the `Strategy`
   Protocol from `aitrade.strategy.base`.
2. Register it in `aitrade.strategy.registry` (simple dict).
3. Unit-test against synthetic bars in `tests/`.
4. Backtest via `uv run aitrade backtest <name> --symbol ... --start ... --end ...`.
5. Paper-run via `uv run aitrade paper <name>` — watch the JSONL trade log.

## Adding a new broker (future)

Implement `aitrade.brokers.base.BrokerClient` Protocol. Do not change call-site
code — the abstraction exists so Kalshi/etc. plug in cleanly.

## Don't

- Don't `pip install` — use `uv add`.
- Don't commit `.env`, `data/`, or `logs/`.
- Don't `--no-verify` on commits.
- Don't place live orders from tests or notebooks.
