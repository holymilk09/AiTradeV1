"""Slice 0 baseline schema.

Tables: universes, universe_members, strategies, strategy_configs, prompts,
journal_events, halt_rules. Extensions (timescaledb, vector) are pre-loaded
by scripts/postgres-init.sql so Alembic does not need superuser to enable
them.

Revision ID: 0001_baseline
Revises:
Create Date: 2026-04-29
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_baseline"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # universes
    op.create_table(
        "universes",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("color", sa.Text, nullable=False, server_default="#888888"),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_table(
        "universe_members",
        sa.Column(
            "universe_id",
            sa.BigInteger,
            sa.ForeignKey("universes.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("symbol", sa.Text, primary_key=True),
        sa.Column(
            "added_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    # strategies + configs + prompts
    op.create_table(
        "strategies",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("status", sa.Text, nullable=False, server_default="disabled"),
        sa.Column(
            "instruments",
            sa.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "regimes_claimed",
            sa.ARRAY(sa.Text),
            nullable=False,
            server_default=sa.text("'{}'::text[]"),
        ),
        sa.Column(
            "promotion_gate_overrides",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_table(
        "strategy_configs",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "strategy_id",
            sa.BigInteger,
            sa.ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "params",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("prompt_version", sa.Integer, nullable=True),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_strategy_configs_active",
        "strategy_configs",
        ["strategy_id", "active"],
    )
    op.create_table(
        "prompts",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "strategy_id",
            sa.BigInteger,
            sa.ForeignKey("strategies.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("strategy_id", "version", name="uq_prompts_strategy_version"),
    )

    # journal_events — canonical event log; mirrors JSONL on disk
    op.create_table(
        "journal_events",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("event_id", sa.Text, nullable=False, unique=True),
        sa.Column(
            "ts",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("run_id", sa.Text, nullable=False),
        sa.Column("strategy_id", sa.Text, nullable=True),
        sa.Column("account", sa.Text, nullable=True),
        sa.Column("ticker", sa.Text, nullable=True),
        sa.Column("event_type", sa.Text, nullable=False),
        sa.Column("severity", sa.Text, nullable=False, server_default="info"),
        sa.Column(
            "payload",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("source", sa.Text, nullable=False, server_default="auto"),
    )
    op.create_index("ix_journal_ts_desc", "journal_events", [sa.text("ts DESC")])
    op.create_index("ix_journal_run", "journal_events", ["run_id"])
    op.create_index("ix_journal_type_ts", "journal_events", ["event_type", sa.text("ts DESC")])
    op.create_index("ix_journal_ticker", "journal_events", ["ticker"])
    op.execute(
        "CREATE INDEX ix_journal_payload_gin ON journal_events USING GIN (payload jsonb_path_ops)"
    )

    # halt rules — data-driven rule definitions
    op.create_table(
        "halt_rules",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False, unique=True),
        sa.Column(
            "expr",
            sa.dialects.postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("scope", sa.Text, nullable=False, server_default="halt_new"),
        sa.Column("enabled", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column(
            "created_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("halt_rules")
    op.drop_index("ix_journal_payload_gin", table_name="journal_events")
    op.drop_index("ix_journal_ticker", table_name="journal_events")
    op.drop_index("ix_journal_type_ts", table_name="journal_events")
    op.drop_index("ix_journal_run", table_name="journal_events")
    op.drop_index("ix_journal_ts_desc", table_name="journal_events")
    op.drop_table("journal_events")
    op.drop_table("prompts")
    op.drop_index("ix_strategy_configs_active", table_name="strategy_configs")
    op.drop_table("strategy_configs")
    op.drop_table("strategies")
    op.drop_table("universe_members")
    op.drop_table("universes")
