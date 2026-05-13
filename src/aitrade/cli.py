"""aitrade CLI — verify, backtest, paper, analyze."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from aitrade.config import get_settings
from aitrade.data.models import Timeframe
from aitrade.observability.setup import configure_logging

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _parse_duration(text: str) -> timedelta:
    text = text.strip().lower()
    unit = text[-1]
    value = int(text[:-1])
    if unit == "s":
        return timedelta(seconds=value)
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    raise typer.BadParameter(f"duration must end in s/m/h; got {text!r}")


@app.command()
def verify() -> None:
    """Ping Alpaca paper account and print a summary."""
    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)
    if not s.has_credentials:
        console.print(
            "[red]Credentials missing.[/red] "
            "Set ALPACA_API_KEY + ALPACA_SECRET_KEY in .env."
        )
        raise typer.Exit(code=1)

    from aitrade.brokers.alpaca import build_client

    broker = build_client()
    acct = broker.get_account()
    table = Table(title=f"Alpaca {'PAPER' if acct.is_paper else 'LIVE'} account")
    table.add_column("field")
    table.add_column("value", justify="right")
    for k, v in [
        ("account_id", acct.account_id),
        ("cash", f"{acct.cash:,.2f} {acct.currency}"),
        ("equity", f"{acct.equity:,.2f} {acct.currency}"),
        ("buying_power", f"{acct.buying_power:,.2f} {acct.currency}"),
    ]:
        table.add_row(k, str(v))
    console.print(table)

    positions = broker.get_positions()
    if positions:
        ptable = Table(title="Open positions")
        for col in ["symbol", "qty", "avg_entry", "market_value", "unrealized_pnl"]:
            ptable.add_column(col, justify="right")
        for p in positions:
            ptable.add_row(
                p.symbol,
                f"{p.qty:.4f}",
                f"{p.avg_entry_price:,.2f}",
                f"{p.market_value:,.2f}",
                f"{p.unrealized_pnl:,.2f}",
            )
        console.print(ptable)


@app.command()
def backtest(
    strategy: str = typer.Argument(..., help="Strategy name from the registry"),
    symbol: str = typer.Option(..., "--symbol", "-s"),
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD"),
    timeframe: Timeframe = typer.Option(Timeframe.DAY_1, "--timeframe", "-t"),
    engine: str = typer.Option("simple", "--engine", help="simple | nautilus"),
) -> None:
    """Backtest a strategy over historical bars."""
    from aitrade.backtest.alpaca_adapter import df_to_bars
    from aitrade.backtest.simple_runner import BacktestConfig, run_backtest
    from aitrade.data.alpaca_data import AlpacaDataClient
    from aitrade.strategy.registry import get_strategy

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)

    strat = get_strategy(strategy, symbol=symbol)
    data = AlpacaDataClient(s)
    df = data.fetch_stock_bars(
        symbol,
        timeframe,
        datetime.fromisoformat(start).replace(tzinfo=UTC),
        datetime.fromisoformat(end).replace(tzinfo=UTC),
    )
    if df.empty:
        console.print(f"[yellow]No bars returned for {symbol}[/yellow]")
        raise typer.Exit(code=1)

    if engine == "nautilus":
        from aitrade.backtest import nautilus_runner

        if not nautilus_runner.is_available():
            console.print(
                "[yellow]nautilus-trader not installed; falling back to simple engine.[/yellow]"
            )
            engine = "simple"

    if engine == "simple":
        result = run_backtest(strat, df_to_bars(df, symbol), BacktestConfig())
        m = result.metrics
        t = Table(title=f"Backtest {strategy} {symbol} [{start} → {end}]  bars={len(df)}")
        t.add_column("metric")
        t.add_column("value", justify="right")
        t.add_row("total_return_pct", f"{m.total_return_pct:+.2f}%")
        t.add_row("cagr_pct", f"{m.cagr_pct:+.2f}%")
        t.add_row("sharpe", f"{m.sharpe:.2f}")
        t.add_row("sortino", f"{m.sortino:.2f}")
        t.add_row("max_drawdown_pct", f"{m.max_drawdown_pct:.2f}%")
        t.add_row("mar", f"{m.mar:.2f}")
        t.add_row("num_trades", str(m.num_trades))
        t.add_row("hit_rate", f"{m.hit_rate:.2%}")
        t.add_row("avg_win", f"{m.avg_win:+.2f}")
        t.add_row("avg_loss", f"{m.avg_loss:+.2f}")
        t.add_row("median_trade", f"{m.median_trade:+.2f}")
        t.add_row("profit_factor", f"{m.profit_factor:.2f}")
        t.add_row("expectancy", f"{m.expectancy:+.2f}")
        t.add_row("longest_losing_streak", str(m.longest_losing_streak))
        t.add_row("return_skew", f"{m.return_skew:+.2f}")
        t.add_row("return_kurtosis", f"{m.return_kurtosis:+.2f}")
        t.add_row("final_equity", f"{m.final_equity:,.2f}")
        console.print(t)

        out_dir = s.aitrade_data_dir / "backtests" / f"{strategy}_{symbol}_{start}_{end}"
        out_dir.mkdir(parents=True, exist_ok=True)
        result.equity_curve.to_frame("equity").to_parquet(out_dir / "equity.parquet")
        console.print(f"[green]Equity curve → {out_dir / 'equity.parquet'}[/green]")
    else:
        from aitrade.backtest.nautilus_runner import NautilusBacktestConfig, run_nautilus_backtest

        report = run_nautilus_backtest(
            df,
            symbol,
            datetime.fromisoformat(start).replace(tzinfo=UTC),
            datetime.fromisoformat(end).replace(tzinfo=UTC),
            config=NautilusBacktestConfig(
                symbol=symbol, timeframe_secs=86_400 if timeframe is Timeframe.DAY_1 else 60
            ),
        )
        console.print(report["report"])


@app.command()
def walkforward(
    strategy: str = typer.Argument(..., help="Strategy name from the registry"),
    symbol: str = typer.Option(..., "--symbol", "-s"),
    start: str = typer.Option(..., "--start", help="YYYY-MM-DD"),
    end: str = typer.Option(..., "--end", help="YYYY-MM-DD"),
    timeframe: Timeframe = typer.Option(Timeframe.DAY_1, "--timeframe", "-t"),
    train_days: int = typer.Option(180, "--train-days"),
    test_days: int = typer.Option(60, "--test-days"),
    step_days: int = typer.Option(0, "--step-days", help="0 = non-overlapping test windows"),
) -> None:
    """Walk-forward backtest a strategy. Prints per-fold table + aggregate."""
    import pandas as pd

    from aitrade.backtest.alpaca_adapter import df_to_bars
    from aitrade.backtest.simple_runner import BacktestConfig
    from aitrade.backtest.walk_forward import summary_to_dataframe, walk_forward
    from aitrade.data.alpaca_data import AlpacaDataClient
    from aitrade.strategy.registry import get_strategy

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)

    data = AlpacaDataClient(s)
    df = data.fetch_stock_bars(
        symbol,
        timeframe,
        datetime.fromisoformat(start).replace(tzinfo=UTC),
        datetime.fromisoformat(end).replace(tzinfo=UTC),
    )
    if df.empty:
        console.print(f"[yellow]No bars returned for {symbol}[/yellow]")
        raise typer.Exit(code=1)
    bars = list(df_to_bars(df, symbol))

    cfg = BacktestConfig(timeframe=timeframe)
    summary = walk_forward(
        lambda: get_strategy(strategy, symbol=symbol),
        bars,
        train=pd.Timedelta(days=train_days),
        test=pd.Timedelta(days=test_days),
        step=pd.Timedelta(days=step_days) if step_days > 0 else None,
        config=cfg,
    )
    per_fold = summary_to_dataframe(summary)

    title = (
        f"Walk-forward {strategy} {symbol} [{start} → {end}] "
        f"train={train_days}d test={test_days}d folds={len(summary.folds)}"
    )
    t = Table(title=title)
    for col in [
        "fold",
        "test_start",
        "test_end",
        "trades",
        "sharpe",
        "cagr%",
        "maxDD%",
        "profitF",
        "expect",
        "hitRate",
    ]:
        t.add_column(col, justify="right")
    for _, row in per_fold.iterrows():
        t.add_row(
            str(row["fold"]),
            str(row["test_start"].date()),
            str(row["test_end"].date()),
            str(int(row["test_trades"])),
            f"{row['test_sharpe']:.2f}",
            f"{row['test_cagr_pct']:+.1f}",
            f"{row['test_max_dd_pct']:.1f}",
            f"{row['test_profit_factor']:.2f}",
            f"{row['test_expectancy']:+.2f}",
            f"{row['test_hit_rate']:.0%}",
        )
    console.print(t)

    agg = Table(title="Aggregate")
    agg.add_column("metric")
    agg.add_column("value", justify="right")
    agg.add_row("test_sharpe_mean", f"{summary.test_sharpe_mean:.2f}")
    agg.add_row("test_sharpe_std", f"{summary.test_sharpe_std:.2f}")
    agg.add_row("test_expectancy_mean", f"{summary.test_expectancy_mean:+.2f}")
    agg.add_row("test_profit_factor_median", f"{summary.test_profit_factor_median:.2f}")
    agg.add_row("test_max_dd_worst_pct", f"{summary.test_max_dd_pct_worst:.2f}%")
    agg.add_row("test_trade_count_total", str(summary.test_trade_count_total))
    agg.add_row("train_test_sharpe_corr", f"{summary.train_test_sharpe_corr:+.2f}")
    console.print(agg)

    out_dir = s.aitrade_data_dir / "walkforward" / f"{strategy}_{symbol}_{start}_{end}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_fold.to_parquet(out_dir / "folds.parquet")
    console.print(f"[green]Per-fold metrics → {out_dir / 'folds.parquet'}[/green]")


@app.command()
def paper(
    strategy: str = typer.Argument(...),
    symbol: str = typer.Option(..., "--symbol", "-s"),
    duration: str = typer.Option("10m", "--duration", "-d", help="e.g. 30s, 15m, 2h"),
    timeframe: Timeframe = typer.Option(Timeframe.MIN_1, "--timeframe", "-t"),
    target_notional: float = typer.Option(2_000.0, "--notional"),
    poll_secs: int = typer.Option(60, "--poll-secs"),
) -> None:
    """Run a paper-trading session."""
    from aitrade.bots.runner import PaperRunConfig, run_paper_session
    from aitrade.brokers.alpaca import build_client
    from aitrade.data.alpaca_data import AlpacaDataClient
    from aitrade.execution.executor import Executor
    from aitrade.execution.risk import RiskGate
    from aitrade.logging.trade_logger import TradeLogger
    from aitrade.strategy.registry import get_strategy

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)

    broker = build_client()  # paper by default
    data = AlpacaDataClient(s)
    risk = RiskGate(
        max_position_usd=s.aitrade_max_position_usd,
        max_daily_loss_usd=s.aitrade_max_daily_loss_usd,
        max_orders_per_min=s.aitrade_max_orders_per_min,
        kill_switch_path=Path("KILL_SWITCH"),
    )
    exe = Executor(broker, risk)
    strat = get_strategy(strategy, symbol=symbol)
    with TradeLogger(log_dir=s.aitrade_log_dir, strategy_id=strat.strategy_id) as journal:
        cfg = PaperRunConfig(
            symbol=symbol,
            timeframe=timeframe,
            duration=_parse_duration(duration),
            poll_interval_secs=poll_secs,
            target_position_notional=target_notional,
        )
        run_paper_session(strat, broker, data, exe, journal, cfg)
        console.print(f"[green]Paper run complete.[/green] run_id={journal.run_id}")


@app.command()
def analyze(
    run_id: str = typer.Argument(..., help="Run id from a previous paper/backtest session"),
) -> None:
    """Summarize a past run from the trade journal."""
    from aitrade.logging.trade_logger import TradeLogger

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)
    journal = TradeLogger(log_dir=s.aitrade_log_dir)
    events = journal.events_for_run(run_id)
    if not events:
        console.print(f"[yellow]No events for run_id={run_id}[/yellow]")
        raise typer.Exit(code=1)
    t = Table(title=f"Run {run_id} — {len(events)} events")
    for col in ["timestamp", "event_type", "symbol", "summary"]:
        t.add_column(col)
    for ev in events:
        payload = ev["payload"]
        summary_parts: list[str] = []
        if "order" in payload:
            o = payload["order"]
            summary_parts.append(f"{o.get('side')} {o.get('qty')} @ {o.get('order_type')}")
        if "reason" in payload:
            summary_parts.append(f"reason={payload['reason']}")
        t.add_row(ev["timestamp"], ev["event_type"], ev["symbol"], " ".join(summary_parts))
    console.print(t)


@app.command()
def strategies() -> None:
    """List registered strategies."""
    from aitrade.strategy.registry import list_strategies

    for name in list_strategies():
        console.print(f"- {name}")


@app.command("smoke-trade")
def smoke_trade(
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    qty: float = typer.Option(1.0, "--qty", help="Shares to buy then sell"),
    hold_secs: int = typer.Option(5, "--hold-secs", help="Seconds to hold before closing"),
) -> None:
    """Place a visible paper scalp (BUY market -> hold -> SELL market).

    Meant for sanity-checking the end-to-end pipeline: you should see the
    order and fill appear on the Alpaca paper dashboard and in the JSONL
    trade journal. Defaults are tiny; the risk gate still runs.
    """
    import time

    from aitrade.brokers.alpaca import build_client
    from aitrade.data.alpaca_data import AlpacaDataClient
    from aitrade.data.models import Timeframe
    from aitrade.execution.executor import Executor
    from aitrade.execution.orders import OrderRequest, OrderType, Side, TimeInForce
    from aitrade.execution.risk import RiskGate
    from aitrade.logging.trade_logger import TradeLogger

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)

    broker = build_client()  # paper by default
    if not broker.is_paper:
        console.print("[red]Refusing smoke-trade on a non-paper broker.[/red]")
        raise typer.Exit(code=1)

    data = AlpacaDataClient(s)
    df = data.fetch_stock_bars(
        symbol,
        Timeframe.MIN_1,
        datetime.now(UTC) - timedelta(days=5),
        datetime.now(UTC),
        use_cache=False,
    )
    if df.empty:
        console.print(f"[yellow]No recent bars for {symbol} — can't size risk check.[/yellow]")
        raise typer.Exit(code=1)
    reference_price = float(df["close"].iloc[-1])

    risk = RiskGate(
        max_position_usd=s.aitrade_max_position_usd,
        max_daily_loss_usd=s.aitrade_max_daily_loss_usd,
        max_orders_per_min=s.aitrade_max_orders_per_min,
        kill_switch_path=Path("KILL_SWITCH"),
    )
    exe = Executor(broker, risk)

    with TradeLogger(log_dir=s.aitrade_log_dir, strategy_id="smoke_trade") as journal:
        buy = OrderRequest(
            symbol=symbol,
            side=Side.BUY,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id="smoke_trade",
        )
        console.print(f"[cyan]BUY  {qty} {symbol} @ market (ref={reference_price:.2f})[/cyan]")
        journal.log_submit(buy)
        r1 = exe.submit(buy, reference_price=reference_price)
        if not r1.accepted:
            console.print(f"[red]BUY blocked:[/red] {r1.reason}")
            journal.log_risk_block(buy, r1.reason)
            raise typer.Exit(code=1)
        if r1.ack:
            journal.log_ack(buy, r1.ack)
            console.print(
                f"[green]BUY ack:[/green] "
                f"broker_id={r1.ack.broker_order_id} status={r1.ack.status.value}"
            )

        console.print(f"[dim]holding for {hold_secs}s…[/dim]")
        time.sleep(hold_secs)

        sell = OrderRequest(
            symbol=symbol,
            side=Side.SELL,
            qty=qty,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            strategy_id="smoke_trade",
        )
        console.print(f"[cyan]SELL {qty} {symbol} @ market[/cyan]")
        journal.log_submit(sell)
        r2 = exe.submit(sell, reference_price=reference_price, current_position_qty=qty)
        if not r2.accepted:
            console.print(f"[red]SELL blocked:[/red] {r2.reason}")
            journal.log_risk_block(sell, r2.reason)
            raise typer.Exit(code=1)
        if r2.ack:
            journal.log_ack(sell, r2.ack)
            console.print(
                f"[green]SELL ack:[/green] "
                f"broker_id={r2.ack.broker_order_id} status={r2.ack.status.value}"
            )

        console.print(
            f"\n[green]done[/green]  run_id={journal.run_id}  "
            f"(see Alpaca paper dashboard + logs/trades.jsonl)"
        )


@app.command("reason-paper")
def reason_paper(
    symbol: str = typer.Option("AAPL", "--symbol", "-s"),
    duration: str = typer.Option("30m", "--duration", "-d", help="e.g. 30s, 15m, 2h"),
    timeframe: Timeframe = typer.Option(Timeframe.MIN_5, "--timeframe", "-t"),
    poll_secs: int = typer.Option(60, "--poll-secs"),
) -> None:
    """Run a paper session where Claude decides every trade.

    On each poll: compute indicators -> call reasoner -> risk gate -> Alpaca -> journal.
    Reasoner calls are rate-limited per symbol (default: 5 min) to bound token cost.
    Every decision (including `reason`) is saved to the trade journal for post-mortem.
    """
    from pathlib import Path

    from aitrade.bots.reasoner_runner import ReasonerRunConfig, run_reasoner_session
    from aitrade.brokers.alpaca import build_client
    from aitrade.data.alpaca_data import AlpacaDataClient
    from aitrade.execution.executor import Executor
    from aitrade.execution.risk import RiskGate
    from aitrade.logging.trade_logger import TradeLogger
    from aitrade.reasoning.claude_reasoner import ClaudeReasoner

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)
    if not s.anthropic_api_key.get_secret_value():
        console.print(
            "[red]ANTHROPIC_API_KEY missing.[/red] Add it to .env and re-run."
        )
        raise typer.Exit(code=1)

    broker = build_client()
    data = AlpacaDataClient(s)
    risk = RiskGate(
        max_position_usd=s.aitrade_max_position_usd,
        max_daily_loss_usd=s.aitrade_max_daily_loss_usd,
        max_orders_per_min=s.aitrade_max_orders_per_min,
        kill_switch_path=Path("KILL_SWITCH"),
    )
    exe = Executor(broker, risk)
    reasoner = ClaudeReasoner(settings=s)

    with TradeLogger(log_dir=s.aitrade_log_dir, strategy_id="llm_advised") as journal:
        cfg = ReasonerRunConfig(
            symbol=symbol,
            timeframe=timeframe,
            duration=_parse_duration(duration),
            poll_interval_secs=poll_secs,
        )
        run_reasoner_session(reasoner, broker, data, exe, journal, cfg)
        console.print(f"[green]Reasoner run complete.[/green] run_id={journal.run_id}")


@app.command("chat")
def chat() -> None:
    """Open Claude Code conversationally with Alpaca MCP already wired.

    Ad-hoc mode: ask Claude to check positions, place paper orders, or
    research in plain English. Uses the MCP server configured in
    .claude/settings.json. This is NOT the 24/7 bot — it's manual
    interactive trading for research and one-offs.
    """
    console.print(
        "\n[bold]Conversational trading mode[/bold]\n\n"
        "1. Install Claude Code on this machine: https://claude.com/claude-code\n"
        "2. Run [cyan]claude[/cyan] in this project directory.\n"
        "3. The Alpaca MCP server auto-starts from .claude/settings.json.\n"
        "4. Try: [dim]'show my paper positions'[/dim] or "
        "[dim]'place a paper buy for 1 AAPL'[/dim].\n\n"
        "The risk gate DOES NOT run in this mode — orders go straight to Alpaca. "
        "Use small paper sizes only.\n"
    )


@app.command("discover")
def discover_cmd(
    top_n: int = typer.Option(20, "--top-n", "-n", help="Top N tickers to surface"),
) -> None:
    """One-shot dump of the current buzz-driven candidate board.

    Useful for sanity-checking discovery before running the full engine.
    Calls Claude with web_search; costs a small number of input tokens.
    """
    from aitrade.discovery.agent import DiscoveryAgent
    from aitrade.discovery.extractor import TickerExtractor, load_alpaca_active_equities
    from aitrade.discovery.scorer import BuzzScorer

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)
    if not s.anthropic_api_key.get_secret_value():
        console.print("[red]ANTHROPIC_API_KEY missing.[/red] Add it to .env.")
        raise typer.Exit(code=1)
    if not s.has_credentials:
        console.print("[red]Alpaca credentials missing.[/red] Needed for ticker validation.")
        raise typer.Exit(code=1)

    valid = load_alpaca_active_equities(s)
    extractor = TickerExtractor(valid_tickers=valid)
    scorer = BuzzScorer()
    agent = DiscoveryAgent(settings=s, extractor=extractor, scorer=scorer)

    tickers = agent.discover(top_n=top_n)
    if not tickers:
        console.print("[yellow]No tickers surfaced this cycle.[/yellow]")
        return

    t = Table(title=f"Discovered tickers (top {len(tickers)})")
    for col in ["symbol", "buzz", "mentions", "src_w", "age_min", "snippet"]:
        t.add_column(col)
    for d in tickers:
        snippet = d.evidence[0][:50] + "…" if d.evidence else "-"
        t.add_row(
            d.symbol,
            f"{d.buzz_score:.3f}",
            str(d.mention_count),
            f"{d.source_weight:.2f}",
            f"{d.recency_minutes:.1f}",
            snippet,
        )
    console.print(t)


@app.command("engine-paper")
def engine_paper(
    duration: str = typer.Option(
        "1h", "--duration", "-d", help="e.g. 30m, 2h — bot runs for this long"
    ),
    cycle_secs: int | None = typer.Option(
        None,
        "--cycle-secs",
        help="Override cycle interval (default from AITRADE_ENGINE_INTERVAL_SECS).",
    ),
    top_n: int | None = typer.Option(
        None, "--top-n", help="Override discovery top-N (default from env)."
    ),
    target_notional: float = typer.Option(
        2_000.0, "--notional", help="Target USD per trade. Capped by RiskGate."
    ),
    use_web_discovery: bool = typer.Option(
        False,
        "--use-web-discovery/--no-web-discovery",
        help="Add Claude+web_search sentiment overlay. Off by default — "
        "watchlist + Alpaca movers are usually enough and don't need network reliability.",
    ),
    no_watchlist: bool = typer.Option(
        False, "--no-watchlist", help="Disable the static watchlist source."
    ),
    no_movers: bool = typer.Option(
        False, "--no-movers", help="Disable the Alpaca movers source."
    ),
) -> None:
    """Run the full Phase-1 signal engine in paper mode.

    Each cycle: build universe (watchlist + movers + optional web) →
    scan multi-TF patterns → rank candidates → Claude floor-trader picks
    one (or passes) → risk gate → Alpaca paper. Every event lands in
    the trade journal; round-trips are reconciled after each cycle.
    """
    from aitrade.bots.engine_runner import EngineConfig, run_engine
    from aitrade.brokers.alpaca import build_client
    from aitrade.data.alpaca_data import AlpacaDataClient
    from aitrade.discovery.agent import DiscoveryAgent
    from aitrade.discovery.extractor import TickerExtractor, load_alpaca_active_equities
    from aitrade.discovery.movers import MoversFinder
    from aitrade.discovery.scorer import BuzzScorer
    from aitrade.execution.executor import Executor
    from aitrade.execution.risk import RiskGate
    from aitrade.journal.round_trips import RoundTripReconciler
    from aitrade.logging.trade_logger import TradeLogger
    from aitrade.market.snapshot import MarketSnapshotFetcher
    from aitrade.reasoning.floor_trader import FloorTraderReasoner

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)
    if not s.anthropic_api_key.get_secret_value():
        console.print("[red]ANTHROPIC_API_KEY missing.[/red] Add it to .env.")
        raise typer.Exit(code=1)
    if not s.has_credentials:
        console.print("[red]Alpaca credentials missing.[/red] Add them to .env.")
        raise typer.Exit(code=1)

    broker = build_client()  # paper by default
    data = AlpacaDataClient(s)
    risk = RiskGate(
        max_position_usd=s.aitrade_max_position_usd,
        max_daily_loss_usd=s.aitrade_max_daily_loss_usd,
        max_orders_per_min=s.aitrade_max_orders_per_min,
        kill_switch_path=Path("KILL_SWITCH"),
    )
    exe = Executor(broker, risk)

    movers = MoversFinder(settings=s) if not no_movers else None

    discovery_agent: DiscoveryAgent | None = None
    if use_web_discovery:
        valid = load_alpaca_active_equities(s)
        extractor = TickerExtractor(valid_tickers=valid)
        scorer = BuzzScorer()
        discovery_agent = DiscoveryAgent(settings=s, extractor=extractor, scorer=scorer)

    market_fetcher = MarketSnapshotFetcher(
        data,
        ttl_secs=s.aitrade_market_snapshot_ttl_secs,
        max_stale_secs=s.aitrade_market_snapshot_max_stale_secs,
    )
    reasoner = FloorTraderReasoner(settings=s)

    cfg = EngineConfig(
        cycle_secs=cycle_secs or s.aitrade_engine_interval_secs,
        discovery_top_n=top_n or s.aitrade_discovery_top_n,
        target_notional_per_trade=target_notional,
        duration=_parse_duration(duration),
        use_watchlist=not no_watchlist,
        use_movers=not no_movers,
        use_web_discovery=use_web_discovery,
    )

    with TradeLogger(log_dir=s.aitrade_log_dir, strategy_id="engine") as journal:
        reconciler = RoundTripReconciler(journal)
        sources = []
        if cfg.use_watchlist:
            sources.append("watchlist")
        if cfg.use_movers:
            sources.append("movers")
        if cfg.use_web_discovery:
            sources.append("web")
        console.print(
            f"[cyan]Engine starting[/cyan]: cycle={cfg.cycle_secs}s "
            f"top_n={cfg.discovery_top_n} notional=${cfg.target_notional_per_trade:.0f} "
            f"duration={cfg.duration} sources=[{', '.join(sources)}]"
        )
        run_engine(
            discovery=discovery_agent,
            movers=movers,
            market_fetcher=market_fetcher,
            data=data,
            broker=broker,
            executor=exe,
            reasoner=reasoner,
            journal=journal,
            reconciler=reconciler,
            cfg=cfg,
        )
        console.print(f"[green]Engine done.[/green] run_id={journal.run_id}")


@app.command("journal")
def journal_cmd(
    symbol: str | None = typer.Option(None, "--symbol", "-s", help="Filter by symbol"),
    wins_only: bool = typer.Option(False, "--wins-only", help="Only WIN round-trips"),
    losses_only: bool = typer.Option(False, "--losses-only", help="Only LOSS round-trips"),
    limit: int = typer.Option(50, "--limit", "-n", help="Max rows to show"),
) -> None:
    """Pretty-print round-trips from the trade journal.

    Reconciles open BUY/SELL pairs first so the latest state is shown.
    """
    from aitrade.journal.round_trips import PnlBucket, RoundTripReconciler
    from aitrade.logging.trade_logger import TradeLogger

    s = get_settings()
    configure_logging(s.aitrade_log_dir, s.aitrade_log_level)
    journal = TradeLogger(log_dir=s.aitrade_log_dir)
    reconciler = RoundTripReconciler(journal)
    reconciler.reconcile()

    rows = reconciler.all_round_trips(symbol=symbol)
    if wins_only:
        rows = [r for r in rows if r.pnl_bucket is PnlBucket.WIN]
    if losses_only:
        rows = [r for r in rows if r.pnl_bucket is PnlBucket.LOSS]
    rows = rows[:limit]

    if not rows:
        console.print("[yellow]No round-trips match those filters yet.[/yellow]")
        return

    t = Table(title=f"Round-trips ({len(rows)} shown)")
    for col in ["entry_ts", "symbol", "qty", "entry", "exit", "pnl_pct", "pnl_usd",
                "bucket", "exit_reason", "thesis"]:
        t.add_column(col)
    total_pnl = 0.0
    wins = losses = 0
    for r in rows:
        total_pnl += r.pnl_usd
        if r.pnl_bucket is PnlBucket.WIN:
            wins += 1
        elif r.pnl_bucket is PnlBucket.LOSS:
            losses += 1
        thesis = (r.entry_thesis or "-")[:60]
        t.add_row(
            r.entry_ts.strftime("%Y-%m-%d %H:%M"),
            r.symbol,
            f"{r.qty:.2f}",
            f"{r.entry_price:.2f}",
            f"{r.exit_price:.2f}",
            f"{r.pnl_pct * 100:+.2f}%",
            f"{r.pnl_usd:+.2f}",
            r.pnl_bucket.value,
            r.exit_reason.value,
            thesis,
        )
    console.print(t)
    console.print(
        f"\n[bold]Total P&L:[/bold] ${total_pnl:+,.2f}  "
        f"[bold]Wins/Losses:[/bold] {wins}/{losses}  "
        f"[bold]Hit rate:[/bold] "
        f"{wins / max(1, wins + losses):.1%}"
    )


if __name__ == "__main__":
    app()
