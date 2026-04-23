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
        t.add_row("sharpe", f"{m.sharpe:.2f}")
        t.add_row("sortino", f"{m.sortino:.2f}")
        t.add_row("max_drawdown_pct", f"{m.max_drawdown_pct:.2f}%")
        t.add_row("num_trades", str(m.num_trades))
        t.add_row("hit_rate", f"{m.hit_rate:.2%}")
        t.add_row("avg_win", f"{m.avg_win:+.2f}")
        t.add_row("avg_loss", f"{m.avg_loss:+.2f}")
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


if __name__ == "__main__":
    app()
