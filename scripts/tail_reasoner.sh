#!/usr/bin/env bash
# Tail Claude floor-trader output from the trade journal.
#
# Usage:
#   ./scripts/tail_reasoner.sh decisions   # live: one-line per decision (default)
#   ./scripts/tail_reasoner.sh board       # live: top-5 of each candidate board
#   ./scripts/tail_reasoner.sh last        # snapshot: full payload of most recent decision
#   ./scripts/tail_reasoner.sh stops       # snapshot: stop / target ATR multipliers per pick
#
# The journal is append-only JSON Lines at $AITRADE_LOG_DIR/trades.jsonl
# (defaults to ./logs/trades.jsonl).
set -euo pipefail

LOG="${AITRADE_LOG_DIR:-./logs}/trades.jsonl"
if [[ ! -f "$LOG" ]]; then
  echo "no journal yet at $LOG — start the engine first (aitrade engine-paper)" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required: brew install jq" >&2
  exit 1
fi

mode="${1:-decisions}"

case "$mode" in
  decisions)
    echo "▸ live decisions  (ctrl-c to stop) — $LOG"
    tail -n 0 -f "$LOG" | jq -c --unbuffered '
      select(.event=="FLOOR_TRADER_DECISION")
      | {
          ts: .ts,
          sym: .symbol,
          trade: .payload.should_trade,
          dir: .payload.direction,
          conf: .payload.confidence,
          notional: .payload.target_notional_usd,
          thesis: (.payload.thesis // ""),
          pass_why: (.payload.reason_for_pass // null)
        }'
    ;;
  board)
    echo "▸ live boards  (ctrl-c to stop) — $LOG"
    tail -n 0 -f "$LOG" | jq -c --unbuffered '
      select(.event=="CANDIDATE_BOARD")
      | {
          ts: .ts,
          n: (.payload.candidates | length),
          top5: [.payload.candidates[0:5][] | {sym: .symbol, score: .combined_score}]
        }'
    ;;
  last)
    echo "▸ last full decision payload — $LOG"
    grep '"event":"FLOOR_TRADER_DECISION"' "$LOG" | tail -1 | jq .
    ;;
  stops)
    echo "▸ recent decisions with stop / target plans"
    grep '"event":"FLOOR_TRADER_DECISION"' "$LOG" | tail -20 | jq -c '
      {
        ts: .ts,
        sym: .payload.pick_symbol,
        conf: .payload.confidence,
        stop_atr: .payload.stop_atr_mult,
        target_atr: .payload.target_atr_mult,
        rr: (if .payload.stop_atr_mult and .payload.target_atr_mult
             then (.payload.target_atr_mult / .payload.stop_atr_mult) else null end),
        thesis: .payload.thesis
      } | select(.sym != null)'
    ;;
  *)
    echo "Usage: $(basename "$0") [decisions|board|last|stops]" >&2
    exit 2
    ;;
esac
