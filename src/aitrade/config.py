"""Settings with an explicit live-trade guardrail.

Default everywhere is paper. Live orders require BOTH:

1. Environment: ``ALPACA_LIVE_TRADE=true``.
2. Call site: ``confirm_live=True`` passed into the broker factory.

This mirrors the Alpaca CLI's own explicit-opt-in design and prevents
config-only mistakes from producing real orders.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="",
        extra="ignore",
    )

    alpaca_api_key: SecretStr = Field(default=SecretStr(""))
    alpaca_secret_key: SecretStr = Field(default=SecretStr(""))
    alpaca_live_trade: bool = Field(default=False)

    aitrade_data_dir: Path = Field(default=Path("./data"))
    aitrade_log_dir: Path = Field(default=Path("./logs"))
    aitrade_log_level: str = Field(default="INFO")

    aitrade_max_position_usd: float = Field(default=5_000.0, gt=0)
    aitrade_max_daily_loss_usd: float = Field(default=500.0, gt=0)
    aitrade_max_orders_per_min: int = Field(default=20, gt=0)

    @property
    def has_credentials(self) -> bool:
        return bool(
            self.alpaca_api_key.get_secret_value()
            and self.alpaca_secret_key.get_secret_value()
        )

    def ensure_dirs(self) -> None:
        self.aitrade_data_dir.mkdir(parents=True, exist_ok=True)
        self.aitrade_log_dir.mkdir(parents=True, exist_ok=True)


class LiveTradeNotUnlockedError(RuntimeError):
    """Raised when a caller asks for live trading without both opt-ins."""


def assert_paper_or_unlocked(*, confirm_live: bool, settings: Settings) -> bool:
    """Return True iff live trading is authorized. Raise if partially enabled.

    - Both flags off → paper (returns False).
    - ``confirm_live`` True AND ``ALPACA_LIVE_TRADE=true`` → live (returns True).
    - Exactly one flag on → raise — mismatched intent is never silently resolved.
    """
    env_live = settings.alpaca_live_trade
    if confirm_live and env_live:
        return True
    if not confirm_live and not env_live:
        return False
    missing = "ALPACA_LIVE_TRADE=true" if confirm_live else "confirm_live=True"
    raise LiveTradeNotUnlockedError(
        f"Live trading requires BOTH env var and call-site flag. Missing: {missing}."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
