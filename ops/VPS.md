# VPS deployment — AiTradeV1 paper runner

Production-quality host for the paper trading runner. The Mac with a closed
laptop and flaky Tailscale is fine for development; for anything that needs
to be up while the operator is asleep / travelling / in China, use a VPS.

This doc is the **operator-facing runbook**. The bootstrap script lives at
`scripts/vps_bootstrap.sh` and automates everything below the "Manual one-time
steps" line.

## Provider & sizing

| | Hetzner CPX11 | DigitalOcean Basic |
|-|-|-|
| Price | €4.51/mo (~$5) | $6/mo |
| vCPU | 2 (AMD EPYC) | 1 |
| RAM | 2 GB | 1 GB |
| Disk | 40 GB NVMe | 25 GB SSD |
| Bandwidth | 20 TB | 1 TB |
| US region | Ashburn, Hillsboro | NYC, SFO, ATL |
| Recommendation | **Preferred** — more headroom, cheaper | Acceptable fallback |

**Region**: pick the one closest to Alpaca's API. Alpaca's market data + REST
endpoints front-end through CloudFront, so any US region is fine; East Coast
(Ashburn / NYC) is marginally lower latency to NYSE-adjacent feeds.

**OS**: Ubuntu 24.04 LTS. The bootstrap script is written for it. 22.04 also
works but tail uv install behavior on older glibc can surprise.

**Firewall**: enable the provider's firewall and allow only:
- inbound: 22/tcp from your Tailscale IP range only (or close port 22 entirely
  once Tailscale SSH is up)
- inbound: none else
- outbound: all

The runner makes only outbound HTTPS calls (Alpaca, Anthropic, FMP, Reddit) —
no inbound traffic is needed.

## Manual one-time steps

These can't be automated.

### 1. Provision the box
- Create the VPS in the chosen provider's UI.
- Add your SSH public key during creation so password auth never opens.

### 2. Create a non-root sudoer
SSH in as `root`, then:
```bash
adduser aitrade
usermod -aG sudo aitrade
rsync --archive --chown=aitrade:aitrade ~/.ssh /home/aitrade/
```
Log out and back in as `aitrade`. The rest of the doc assumes you're this user.

### 3. Tailscale auth key
Generate a **reusable**, **ephemeral** auth key from
[Tailscale admin → Settings → Keys](https://login.tailscale.com/admin/settings/keys).
Reusable so you can re-run bootstrap if you reprovision; ephemeral so a stolen
key auto-expires after 90 days.

The bootstrap script will prompt for it.

### 4. GitHub access
Choose one:
- **HTTPS + Personal Access Token** (simplest). Generate a PAT at
  github.com → Settings → Developer settings → Tokens (classic), scope `repo`.
  Bootstrap will prompt; the PAT is *not* persisted in the git remote URL.
- **SSH**. Run `ssh-keygen` on the box, paste the public key into
  github.com → Settings → SSH and GPG keys, then run bootstrap.

### 5. Secrets ready to paste
Have these in your password manager before running bootstrap:
- Alpaca **paper** API key (Key ID) and secret — from
  `app.alpaca.markets/paper/dashboard/overview`
- Anthropic API key — optional, only needed for the LLM reasoner
- FMP API key — optional, only needed for the earnings calendar

### 6. Run bootstrap
```bash
git clone https://github.com/holymilk09/AiTradeV1.git
./AiTradeV1/scripts/vps_bootstrap.sh
```
Or one-shot (after pushing the script to the default branch):
```bash
curl -fsSL https://raw.githubusercontent.com/holymilk09/AiTradeV1/<branch>/scripts/vps_bootstrap.sh | bash
```

## Runbook

The systemd unit installed by bootstrap is `aitrade-paper.service`.

### Day-to-day

```bash
# Start the runner (also auto-starts at boot).
sudo systemctl start aitrade-paper

# Stop the runner.
sudo systemctl stop aitrade-paper

# Status — is it up? when did it last restart?
sudo systemctl status aitrade-paper

# Tail live logs.
journalctl -u aitrade-paper -f

# Last 200 lines of history.
journalctl -u aitrade-paper -n 200 --no-pager
```

The unit has `Restart=on-failure RestartSec=30s` and a burst limit of
5 restarts per 10 minutes. If the runner crashes 5 times in 10 minutes
systemd backs off — that's a signal to actually look at the logs rather
than letting it loop forever.

### Switching strategy / symbol

Edit the `ExecStart=` line in `/etc/systemd/system/aitrade-paper.service`,
then:
```bash
sudo systemctl daemon-reload
sudo systemctl restart aitrade-paper
```

Strategies currently registered (see `src/aitrade/strategy/registry.py`):
- `sma_crossover` (default; reference)
- `bollinger_reversion`
- `donchian_breakout`

### Flatten paper positions

The runner can't be told to flatten via signal — use the Alpaca paper
dashboard or call the API directly. Quickest path:
```bash
cd ~/AiTradeV1
uv run python - <<'PY'
from aitrade.brokers.alpaca import build_client
b = build_client()
for p in b.get_positions():
    b.close_position(p.symbol)
PY
```

### Pulling updates

```bash
cd ~/AiTradeV1
sudo systemctl stop aitrade-paper
git pull
uv sync --extra dev
sudo systemctl start aitrade-paper
```

### Disaster recovery

Everything stateful is one of:
- `.env` — recreate by re-running bootstrap (it skips overwriting if present).
- `data/` — trade journals, walk-forward outputs. **Back this up**. Easy:
  `rsync -a aitrade@<tailscale-ip>:~/AiTradeV1/data/ ./vps-data/` from the Mac
  weekly.
- `logs/` — disposable. journald is the durable log.

If the box dies, provision a new one, re-run bootstrap, restore `data/`
from your backup. The journal makes the strategy state recoverable.

## Things that will NOT live on the VPS

- `ALPACA_LIVE_TRADE=true` — leave it `false`. Promoting to live requires
  edits to both the `.env` AND a `confirm_live=True` kwarg at the call
  site. Don't bypass either; that gate exists on purpose.
- The dashboard (`aitrade serve`). If you want to view it remotely, port-
  forward via Tailscale (`tailscale serve` or `ssh -L`) — don't open 8080
  to the internet.
- Anything from `tests/` — production paths only.

## Cost & sizing notes

- **Why 2 GB RAM**: `nautilus-trader` + `pandas` + `polars` import alone
  uses ~250 MB. The 1 GB DO droplet runs fine but OOMs if you also try to
  run the dashboard alongside.
- **Why bother with `MemoryMax=512M`** in the unit: bounds the runner's
  resident footprint so a Python memory leak can't take down sshd. The
  steady-state RSS is around 200 MB.
- **Why not Fly.io / Railway / etc.**: they're more expensive at this
  scale and the runner is a single long-lived process, not a request/
  response service. A bare VPS is the right shape.
