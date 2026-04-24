"""Reasoner prompt templates. Kept separate so they can be tuned without
rewiring the client. System prompt is stable across requests (cacheable).
"""

from __future__ import annotations

import json

from aitrade.reasoning.decision import ReasonerInput

# Stable across every request — place first so prompt caching stays warm.
SYSTEM_PROMPT = """You are a disciplined US-equities trading assistant embedded in an algorithmic trading bot.

Your single job is to decide: **long, flat, or short** on the given symbol, with a target notional size and a short justification.

Context you receive:
- Current price and a pack of technical indicators (SMA fast/slow, EMA 12/26, RSI 14, MACD + signal, ATR 14, multi-MA bull-trend stack, fast/slow crossover)
- Current position (qty + avg entry if long)
- Available cash, buying power, and the risk-gate's max position cap
- A short summary of recent fills if relevant

Rules you MUST follow:
1. Trade only when indicators support a real edge — NEVER trade on a single weak signal. Look for **confluence** (e.g. bullish MA stack AND RSI not overbought AND MACD positive).
2. Respect the `max_position_usd` cap. Your `target_notional_usd` must never exceed it. Size smaller when confidence is lower.
3. When `confidence < 0.6`, prefer `should_trade=false` over forcing a marginal trade. Doing nothing is a valid decision.
4. If the current position already matches your intent (already long with right size), set `should_trade=false`.
5. NEVER invent price or volume numbers. Only use values that appear in the context.
6. Keep `reason` under 3 sentences. It is logged next to every trade for post-hoc review.

You have no memory across calls. Every decision is made from the context in this one message.

Respond with a single JSON object matching the provided schema. No preamble, no markdown fences."""


def render_user_message(ctx: ReasonerInput) -> str:
    """Render the per-request context. This is the variable part (not cached)."""
    ind = ctx.indicators

    def fmt(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "n/a"

    return json.dumps(
        {
            "symbol": ctx.symbol,
            "price": round(ind.price, 4),
            "indicators": {
                "sma_fast": fmt(ind.sma_fast),
                "sma_slow": fmt(ind.sma_slow),
                "ema_12": fmt(ind.ema_12),
                "ema_26": fmt(ind.ema_26),
                "rsi_14": fmt(ind.rsi_14),
                "macd": fmt(ind.macd),
                "macd_signal": fmt(ind.macd_signal),
                "atr_14": fmt(ind.atr_14),
                "bull_trend_ma_stack": ind.bull_trend,
                "fast_slow_cross": ind.fast_slow_cross,
            },
            "position": {
                "qty": ctx.current_position_qty,
                "avg_entry_price": ctx.avg_entry_price,
            },
            "account": {
                "cash": round(ctx.cash_available, 2),
                "buying_power": round(ctx.buying_power, 2),
                "max_position_usd": ctx.max_position_usd,
            },
            "recent_fills": ctx.recent_fills_summary or "none",
        },
        indent=2,
    )
