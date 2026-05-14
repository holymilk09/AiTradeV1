"""Phase 4b deep-dig — structured second-opinion call per top candidate.

Mirrors the patterns in test_floor_trader.py: stub the Anthropic SDK at the
``client.messages.parse`` boundary so tests never touch the network.
"""

from __future__ import annotations

from pydantic import SecretStr

from aitrade.config import Settings
from aitrade.reasoning.decision import DeepDigVerdict
from aitrade.reasoning.deep_dig import DeepDigger


class _StubClient:
    """Stand-in for the real ``anthropic.Anthropic`` client.

    The digger reaches in via ``client.messages.parse(...)`` and reads
    ``response.parsed_output``. The stub captures call counts so tests can
    verify caching/idempotence behavior, and lets a single instance return
    different verdicts via the ``next_verdict`` slot.
    """

    def __init__(self, verdict: DeepDigVerdict | None) -> None:
        self.messages = self
        self.next_verdict: DeepDigVerdict | None = verdict
        self.calls = 0
        self.last_kwargs: dict[str, object] = {}
        self.raise_next: BaseException | None = None

    def parse(self, **kwargs: object) -> object:
        self.calls += 1
        self.last_kwargs = kwargs
        if self.raise_next is not None:
            err = self.raise_next
            self.raise_next = None
            raise err
        stub = type("R", (), {})()
        stub.parsed_output = self.next_verdict
        stub.usage = None
        return stub


def _verdict(
    *,
    symbol: str = "AAPL",
    rec: str = "trade",
    conf: float = 0.72,
) -> DeepDigVerdict:
    return DeepDigVerdict(
        symbol=symbol,
        bull_case="Volume confirmation across 1H + 5m above prior breakout level.",
        bear_case="RSI 73 on 1H suggests near-term exhaustion.",
        risk_factors=["Earnings in 6 days", "Gap unfilled below"],
        confidence_in_setup=conf,
        recommendation=rec,  # type: ignore[arg-type]
        reasoning="Confluence is real but extension and earnings warrant caution.",
    )


def _make_digger(verdict: DeepDigVerdict | None) -> tuple[DeepDigger, _StubClient]:
    settings = Settings(anthropic_api_key=SecretStr("test"))
    d = DeepDigger(settings=settings)
    stub = _StubClient(verdict)
    d._client = stub  # type: ignore[assignment]
    return d, stub


def _dig_args(symbol: str = "AAPL") -> dict[str, object]:
    return {
        "symbol": symbol,
        "pattern_hits": ["breakout", "volume_trend"],
        "evidence": {"vwap_dist": 0.4},
        "multi_tf": {"1D": {"rsi": 62}, "1H": {"rsi": 73}, "5m": {"rsi": 58}},
        "market_snapshot": {"regime": "risk_on_low_vol"},
        "news": [],
        "prior_experience": [],
    }


def test_dig_returns_verdict_when_model_succeeds() -> None:
    d, stub = _make_digger(_verdict(symbol="AAPL"))
    out = d.dig(**_dig_args("AAPL"))  # type: ignore[arg-type]
    assert out is not None
    assert out.symbol == "AAPL"
    assert out.recommendation == "trade"
    assert stub.calls == 1


def test_dig_returns_none_when_sdk_raises() -> None:
    d, stub = _make_digger(_verdict())
    stub.raise_next = RuntimeError("network blip")
    out = d.dig(**_dig_args("AAPL"))  # type: ignore[arg-type]
    assert out is None
    assert stub.calls == 1  # we did call, it raised, we swallowed


def test_dig_returns_none_when_parsed_output_is_none() -> None:
    d, _stub = _make_digger(verdict=None)
    out = d.dig(**_dig_args("AAPL"))  # type: ignore[arg-type]
    assert out is None


def test_dig_drops_verdict_for_wrong_symbol() -> None:
    """The digger must not surface a verdict for the wrong ticker.

    If the model substitutes a different ticker in ``verdict.symbol``,
    the engine would feed the floor-trader a verdict that doesn't apply
    to the candidate it's evaluating. Better to skip than confuse.
    """
    d, _stub = _make_digger(_verdict(symbol="MSFT"))
    out = d.dig(**_dig_args("AAPL"))  # type: ignore[arg-type]
    assert out is None


def test_dig_passes_haiku_model_and_cache_breakpoint() -> None:
    """Verifies the digger hits Haiku with prompt caching on the system prompt.

    The cache breakpoint matters: without it the system prompt re-bills as
    cache_creation tokens on every candidate × every cycle. We just check
    the call shape; actual caching is server-side.
    """
    d, stub = _make_digger(_verdict())
    d.dig(**_dig_args("AAPL"))  # type: ignore[arg-type]
    kwargs = stub.last_kwargs
    assert "claude-haiku" in str(kwargs["model"])
    system = kwargs["system"]
    assert isinstance(system, list)
    first_block = system[0]
    assert isinstance(first_block, dict)
    assert first_block.get("cache_control") == {"type": "ephemeral"}


def test_max_top_k_default_is_three() -> None:
    """Cost discipline: out of the box the digger caps at 3 candidates/cycle."""
    settings = Settings(anthropic_api_key=SecretStr("test"))
    d = DeepDigger(settings=settings)
    assert d.max_top_k == 3


def test_dig_does_not_mutate_inputs() -> None:
    """The digger receives mutable lists/dicts; mutation would corrupt the
    engine's per-cycle state across candidates."""
    d, _stub = _make_digger(_verdict())
    args = _dig_args("AAPL")
    pattern_hits_before = list(args["pattern_hits"])  # type: ignore[arg-type]
    d.dig(**args)  # type: ignore[arg-type]
    assert args["pattern_hits"] == pattern_hits_before
