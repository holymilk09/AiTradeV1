# Hand-off — continuing in VS Code / Cursor

Picked up in the Claude Code web sandbox, parking here so the next session
(local agent on macOS with terminal access) can resume without copy-paste.

## Where we are right now

- **Active branch**: `claude/trading-bot-foundation-mdBNL`
- **Last commit pushed**: `0a70b9b fix(runner): network failures in the poll loop don't crash the session`
- **Working tree**: clean (no local-only changes in the web sandbox)
- **CI / dev loop**: green — `uv run pytest && uv run ruff check && uv run mypy src` passes.
- **Sister branch with parallel work**: `claude/trading-dashboard-setup-MlUMQ` — contains an engine-first plan (walk-forward harness, conviction scorers, value-test) that has NOT been merged here yet.

## What was fixed in the last 3 commits

1. **`542c446`** — `is_tradable` was silently rejecting every real symbol because
   `str(AssetStatus.ACTIVE).lower()` returns `"assetstatus.active"`, not
   `"active"`. Now reads `.value`. Also added a per-symbol circuit breaker
   that blacklists a symbol after 3 consecutive `not tradable` returns so the
   paper runner can't spam orders.
2. **`2ac5dce`** — paper runner now captures `session_start` at boot and
   silently warms the strategy on bars older than that. Logs a single
   `warmup complete` line on the first live signal. Stops the
   9-orders-in-6-seconds bug where every historical SMA crossover in the
   last 5 days was firing a real order.
3. **`0a70b9b`** — `fetch_stock_bars` and `broker.get_positions` failures
   inside the poll loop now log + sleep + continue instead of crashing the
   whole session. 8h paper run can ride out a DNS blip.

## Verified live behavior (Mac paper run, 2026-05-13 18:46 ET)

- Warmup completed cleanly.
- Exactly one BUY order on the first live SMA crossover (`buy 6 AAPL @ market`).
- A real DNS outage (`Failed to resolve 'data.alpaca.markets'`) was caught,
  logged, and backed off — session stayed alive.
- Session ended cleanly at the 30-min duration mark.
- Open paper position: 6 AAPL shares. Manually flatten in the Alpaca paper
  dashboard, or leave it for the next session's bearish cross.

## Outstanding decisions

- **VPS migration is overdue.** From China + a closed laptop + flaky
  Tailscale, the Mac is not a viable long-term paper-run host. Move to a
  small US-based VPS (Hetzner CPX11 ~$5/mo or DO $6 droplet) as soon as
  there's evidence a strategy is worth running 24/7.
- **Edge is still unproven.** Only ONE live paper trade so far. Backtest
  the existing strategy before investing more in operationalization.

## Next tasks for the local agent — in order

Each is self-contained. The local agent can run shell commands directly,
so it can finish the cherry-pick, run the backtest, and inspect logs
without copy-paste back to the operator.

### Task 1 — Cherry-pick the walk-forward backtest harness onto this branch

The engine-first work on `claude/trading-dashboard-setup-MlUMQ` includes a
walk-forward backtest harness with richer metrics. Bring it over so we
can backtest the strategies currently registered on this branch
(sma_crossover and any phase-1..8 additions) without leaving the active
branch.

The commits to cherry-pick (in order, oldest first):

```
2d2e58f feat(backtest): walk-forward harness, richer metrics, multi-symbol fix (M1)
603cc38 feat(strategy): add bollinger_reversion + donchian_breakout (M2)
395a13d feat(signals): conviction component contract + calibration harness (M3)
99187e5 feat(signals): reasoner value-test harness + adapter (M4)
9f1b392 feat(m5): macOS notify + weekly review + launchd plist
```

Expected conflicts: `src/aitrade/backtest/metrics.py`,
`src/aitrade/backtest/simple_runner.py`, `src/aitrade/strategy/registry.py`,
`src/aitrade/cli.py`. Prefer the engine-first versions of metrics +
simple_runner; for cli.py keep the phase-1..8 commands and add the
walkforward command from MlUMQ. For registry.py merge both
strategy-registration blocks.

```bash
git checkout claude/trading-bot-foundation-mdBNL
git cherry-pick 2d2e58f 603cc38 395a13d 99187e5 9f1b392
# resolve conflicts; verify
uv run pytest && uv run ruff check && uv run mypy src
git push origin claude/trading-bot-foundation-mdBNL
```

### Task 2 — Run the first real backtest

After Task 1:

```bash
uv run aitrade walkforward sma_crossover \
    --symbol AAPL --start 2024-01-01 --end 2026-01-01 \
    --train-days 180 --test-days 60
```

Show the per-fold table + aggregate stats. If the operator's Alpaca
historical data feed doesn't go back to 2024, try `--start 2025-01-01`.

Decision criteria for whether to bother with a VPS / live:

- `test_sharpe_mean > 0.5` and `test_sharpe_std < test_sharpe_mean` →
  there's something here, proceed to VPS.
- `test_sharpe_mean <= 0` or wildly inconsistent across folds → strategy
  has no edge, fix it before spending VPS hours on it.

### Task 3 — VPS bootstrap

After Task 2 (regardless of result, because even a fixed strategy needs
somewhere to run):

Write `scripts/vps_bootstrap.sh` for Ubuntu 24.04 that:

- Installs `uv` (curl installer + path).
- Installs Tailscale, asks operator for an auth key, joins the tailnet.
- Clones the repo from GitHub (operator pastes their PAT or uses ssh).
- Copies `.env.example` to `.env` and prompts for the four secrets:
  `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `ANTHROPIC_API_KEY`,
  `ALPHA_VANTAGE_API_KEY`. Never commits `.env`.
- Drops a systemd unit at `/etc/systemd/system/aitrade-paper.service` that
  runs `uv run aitrade paper sma_crossover --symbol AAPL --duration 8h`
  under a non-root user with Restart=on-failure.
- `journalctl -u aitrade-paper -f` is the log tail command.

Plus `ops/VPS.md` documenting: provider, region, sizing, manual one-time
steps (Alpaca whitelist, Tailscale ACL), and the runbook for restart /
flatten / upgrade.

## Things explicitly NOT to do without checking in

- Don't enable live trading (`ALPACA_LIVE_TRADE=true`). Paper only until
  backtest + 4 weeks paper expectancy both agree.
- Don't merge `claude/trading-dashboard-setup-MlUMQ` wholesale — the
  cherry-pick is targeted; merging brings dashboard scaffolding code that
  isn't ready.
- Don't push to `main`. PRs to `main` only after the operator reviews.

## How to verify nothing was forgotten

```bash
uv run pytest                    # expect 300+ tests, all pass
uv run ruff check                # zero issues
uv run mypy src                  # zero issues
uv run aitrade --help            # paper, backtest, walkforward (after Task 1)
```
