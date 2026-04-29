"""Runtime config for the API service. Read from env, validated by pydantic."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="AITRADE_",
        extra="ignore",
    )

    database_url: str = Field(
        default="postgresql+asyncpg://aitrade:aitrade@localhost:5432/aitrade",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Bound to localhost; reach via Tailscale per architecture decision.
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)

    # Account toggle default — paper is the safe default everywhere.
    default_account: str = Field(default="paper")  # "paper" | "live"

    # Halt state Redis key.
    halt_state_key: str = Field(default="aitrade:halt_state")


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    return ApiSettings()
