#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: https://docs.astral.sh/uv/" >&2
  exit 1
fi

uv sync --extra dev

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from template. Paste Alpaca paper keys and re-run."
fi

echo "Bootstrap complete. Try: uv run aitrade verify"
