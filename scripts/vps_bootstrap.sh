#!/usr/bin/env bash
# AiTradeV1 — VPS bootstrap for Ubuntu 24.04.
#
# Run as a non-root sudoer on a fresh box. Installs uv, Tailscale, clones
# the repo into the invoking user's home, prompts for secrets, drops a
# systemd unit that runs `aitrade paper` and restarts on failure.
#
# Idempotent: re-running skips installs that are already in place and
# refuses to overwrite an existing .env (the operator can edit it manually).
#
# Usage:
#   curl -fsSL <raw url to this file> | bash -s --
#   # or
#   ./scripts/vps_bootstrap.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Pre-flight
# ---------------------------------------------------------------------------
if [[ "$EUID" -eq 0 ]]; then
  echo "Run as a non-root sudoer, not root. Aborting." >&2
  exit 1
fi
if ! command -v sudo >/dev/null 2>&1; then
  echo "sudo is required. Install it as root first." >&2
  exit 1
fi

# shellcheck disable=SC1091
. /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "Warning: tested on Ubuntu 24.04; ${ID:-unknown} is best-effort." >&2
fi

user="$(id -un)"
home="$HOME"
repo_dir="$home/AiTradeV1"

prompt_secret() {
  # $1 = var name, $2 = description shown to operator
  local var="$1" desc="$2" val=""
  printf '%s\n' "$desc" >&2
  printf '  %s = ' "$var" >&2
  IFS= read -rs val
  printf '\n' >&2
  printf '%s' "$val"
}

# ---------------------------------------------------------------------------
# 1. APT essentials
# ---------------------------------------------------------------------------
echo "==> Installing apt essentials (git, curl, build deps)..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
  ca-certificates curl git gnupg lsb-release \
  build-essential pkg-config

# ---------------------------------------------------------------------------
# 2. uv
# ---------------------------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # The installer drops uv at $HOME/.local/bin; ensure it's on PATH for
  # this shell and future logins.
  export PATH="$home/.local/bin:$PATH"
  if ! grep -q '.local/bin' "$home/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >>"$home/.bashrc"
  fi
else
  echo "==> uv already installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 3. Tailscale
# ---------------------------------------------------------------------------
if ! command -v tailscale >/dev/null 2>&1; then
  echo "==> Installing Tailscale..."
  curl -fsSL https://tailscale.com/install.sh | sh
else
  echo "==> Tailscale already installed: $(tailscale version | head -1)"
fi

if ! sudo tailscale status >/dev/null 2>&1; then
  echo
  echo "==> Tailscale not joined. Paste an auth key from"
  echo "    https://login.tailscale.com/admin/settings/keys"
  echo "    (leave blank to skip and run 'sudo tailscale up' manually later)"
  printf '    auth key: '
  IFS= read -rs ts_key
  printf '\n'
  if [[ -n "$ts_key" ]]; then
    sudo tailscale up --authkey "$ts_key" --ssh --hostname "aitrade-$(hostname -s)"
    unset ts_key
  fi
fi

# ---------------------------------------------------------------------------
# 4. Clone repo
# ---------------------------------------------------------------------------
if [[ ! -d "$repo_dir/.git" ]]; then
  echo
  echo "==> Clone source. Options:"
  echo "   1) HTTPS + GitHub PAT  (recommended for one-shot setup)"
  echo "   2) SSH  (you'll need to add the box's ssh key to GitHub first)"
  printf '   choice [1/2]: '
  read -r clone_choice
  case "$clone_choice" in
    2)
      git clone git@github.com:holymilk09/AiTradeV1.git "$repo_dir"
      ;;
    *)
      printf '   GitHub username: '
      read -r gh_user
      gh_pat="$(prompt_secret GITHUB_PAT '   GitHub PAT (input hidden, scope: repo):')"
      git clone "https://${gh_user}:${gh_pat}@github.com/holymilk09/AiTradeV1.git" "$repo_dir"
      unset gh_pat
      # Strip the PAT out of the remote URL so it's not stored on disk.
      git -C "$repo_dir" remote set-url origin "https://github.com/holymilk09/AiTradeV1.git"
      ;;
  esac
else
  echo "==> Repo already at $repo_dir; pulling latest on default branch..."
  git -C "$repo_dir" fetch --quiet origin
  git -C "$repo_dir" pull --ff-only --quiet || \
    echo "   (non-fast-forward — leaving working tree alone)"
fi

# ---------------------------------------------------------------------------
# 5. .env
# ---------------------------------------------------------------------------
env_file="$repo_dir/.env"
if [[ -f "$env_file" ]]; then
  echo "==> $env_file already exists; not overwriting."
else
  echo "==> Creating $env_file from template + secret prompts..."
  cp "$repo_dir/.env.example" "$env_file"
  chmod 600 "$env_file"

  alpaca_key="$(prompt_secret ALPACA_API_KEY 'Alpaca paper API key (Key ID):')"
  alpaca_secret="$(prompt_secret ALPACA_SECRET_KEY 'Alpaca paper API secret:')"
  anthropic_key="$(prompt_secret ANTHROPIC_API_KEY 'Anthropic API key (blank to disable LLM reasoner):')"
  fmp_key="$(prompt_secret FMP_API_KEY 'FMP API key for earnings calendar (blank to skip):')"

  # In-place edit; the regex preserves comments above each key.
  sed -i \
    -e "s|^ALPACA_API_KEY=.*|ALPACA_API_KEY=${alpaca_key}|" \
    -e "s|^ALPACA_SECRET_KEY=.*|ALPACA_SECRET_KEY=${alpaca_secret}|" \
    -e "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=${anthropic_key}|" \
    -e "s|^FMP_API_KEY=.*|FMP_API_KEY=${fmp_key}|" \
    "$env_file"

  unset alpaca_key alpaca_secret anthropic_key fmp_key
fi

# ---------------------------------------------------------------------------
# 6. uv sync + smoke check
# ---------------------------------------------------------------------------
echo "==> Syncing Python environment (this is the slow step)..."
(cd "$repo_dir" && uv sync --extra dev)

echo "==> Smoke check: aitrade --help"
(cd "$repo_dir" && uv run aitrade --help) | head -5

# ---------------------------------------------------------------------------
# 7. systemd unit
# ---------------------------------------------------------------------------
unit=/etc/systemd/system/aitrade-paper.service
uv_bin="$home/.local/bin/uv"

echo "==> Writing $unit..."
sudo tee "$unit" >/dev/null <<EOF
[Unit]
Description=AiTradeV1 paper trading runner
After=network-online.target tailscaled.service
Wants=network-online.target

[Service]
Type=simple
User=${user}
Group=${user}
WorkingDirectory=${repo_dir}
EnvironmentFile=${repo_dir}/.env
# Default command — edit the ExecStart line if you switch strategies.
# Currently chosen: sma_crossover on AAPL, 8h sessions. The runner exits
# at the duration mark; Restart=always re-launches it for the next session.
ExecStart=${uv_bin} run aitrade paper sma_crossover --symbol AAPL --duration 8h
Restart=on-failure
RestartSec=30s
# Don't hammer the API if something is durably broken.
StartLimitIntervalSec=600
StartLimitBurst=5
StandardOutput=journal
StandardError=journal
# Resource caps — small VPS, leave headroom for OS.
MemoryMax=512M
CPUQuota=80%

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable aitrade-paper.service >/dev/null

echo
echo "==> Bootstrap complete."
echo
echo "Next steps:"
echo "  1. Review $env_file (chmod 600, never commit)."
echo "  2. Start the runner:  sudo systemctl start aitrade-paper"
echo "  3. Tail logs:         journalctl -u aitrade-paper -f"
echo "  4. Stop:              sudo systemctl stop aitrade-paper"
echo
echo "Edit the strategy / symbol / duration in:"
echo "  $unit"
echo "then: sudo systemctl daemon-reload && sudo systemctl restart aitrade-paper"
