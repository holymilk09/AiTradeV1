# aitrade — single-image deploy for fly.io / DigitalOcean / Railway / any
# vanilla Linux container host.
#
# Layout: install uv, lock → install deps into /app/.venv, copy source, run
# the unified `aitrade serve` (engine + dashboard in one process).

FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# uv (Astral) — pinned to match local dev (see CI workflow + bootstrap script)
COPY --from=ghcr.io/astral-sh/uv:0.8.17 /uv /usr/local/bin/uv

WORKDIR /app

# 1. Lockfile-first layer — cached unless deps change.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# 2. Source.
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

# 3. Runtime user — never run as root.
RUN useradd --uid 10001 --create-home --shell /bin/bash aitrade \
    && mkdir -p /app/data /app/logs \
    && chown -R aitrade:aitrade /app
USER aitrade

ENV PATH="/app/.venv/bin:${PATH}" \
    AITRADE_DATA_DIR=/app/data \
    AITRADE_LOG_DIR=/app/logs \
    AITRADE_DASHBOARD_HOST=0.0.0.0

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/healthz || exit 1

# Long duration so the container stays alive across market sessions.
# Override at deploy time with `fly launch ... -- aitrade serve -d 7d`.
CMD ["aitrade", "serve", "-d", "7d"]
