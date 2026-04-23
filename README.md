# AiTradeV1

Algo trading foundation on Alpaca with NautilusTrader-powered backtesting.

Built to be professional-grade from day 1:

- **Paper-only guardrail** — live trading requires two explicit opt-ins (env var + call-site flag).
- **Broker abstraction** so new venues (Kalshi, etc.) plug in without rewrites.
- **NautilusTrader backtester** with realistic slippage/fees.
- **Structured trade log** (JSONL + SQLite) as the substrate for learning and optimization.
- **Typer CLI** to verify accounts, backtest, run paper loops, and analyze results.
- **Alpaca MCP server** wired into Claude Code for conversational research/trading.

## Setup

Requires Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
# 1. install deps
uv sync --extra dev

# 2. configure credentials
cp .env.example .env
# edit .env — paste your Alpaca paper keys

# 3. sanity check the account
uv run aitrade verify
```

## Commands

```bash
# verify Alpaca connection + account balance
uv run aitrade verify

# backtest a strategy on historical bars
uv run aitrade backtest sma_crossover --symbol AAPL --start 2024-01-01 --end 2024-06-30

# run a paper-trading session
uv run aitrade paper sma_crossover --symbol AAPL --duration 10m

# analyze a past run from the trade journal
uv run aitrade analyze <run_id>
```

## Safety model

`ALPACA_LIVE_TRADE=true` **alone is not enough** to place live orders. The broker
factory (`aitrade.brokers.alpaca.build_client`) also requires `confirm_live=True`
at the call site. This mirrors the Alpaca CLI's own explicit-opt-in design.

The default mode everywhere — CLI, tests, notebooks — is paper.

## Layout

```
src/aitrade/
├── config.py          # pydantic Settings + live guardrail
├── brokers/           # Alpaca adapter behind a BrokerClient Protocol
├── data/              # historical + streaming data, parquet cache
├── strategy/          # base Protocol + example strategies
├── execution/         # order router + risk gates
├── backtest/          # NautilusTrader runner + adapters
├── logging/           # trade journal (JSONL + SQLite)
├── observability/     # loguru setup
├── bots/              # runner loop
└── cli.py             # typer entrypoint
```

## Development

```bash
uv run pytest
uv run ruff check
uv run mypy src
```

## License

MIT
