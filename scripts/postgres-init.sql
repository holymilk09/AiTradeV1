-- Bootstrap extensions on first DB init. timescale/timescaledb-ha image
-- ships TimescaleDB and pgvector — enable them here so Alembic migrations
-- can use vector and hypertable APIs immediately.
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
