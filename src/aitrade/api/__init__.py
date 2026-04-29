"""FastAPI service powering the trading dashboard.

Mounted via `aitrade.cli serve-api`. Talks to Postgres + Redis. The existing
`aitrade` library remains importable and unchanged — this layer composes it.
"""
