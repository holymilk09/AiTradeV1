"""Discovery agent — ask Claude what tickers the market is buzzing about.

The agent calls Claude with the ``web_search_20260209`` and ``web_fetch_20260209``
server-side tools. Claude is instructed to emit a single JSON object as its
final text block; the agent parses it and validates each candidate against the
extractor's universe, then scores by :class:`BuzzScorer`.

Note: we deliberately do **not** use ``output_config={"format": {"type":
"json_schema", ...}}`` here. Structured outputs are not compatible with the
server-side ``web_search_20260209`` / ``web_fetch_20260209`` tools — combining
them causes the API to reject the request and the SDK to surface the failure
as a generic ``APIConnectionError("Connection error.")``. We parse the JSON
ourselves from the trailing text block, which is robust enough.

Errors at any stage (no API key, network, unparseable output) yield an empty
list rather than propagating — discovery is best-effort, never load-bearing.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from aitrade.config import Settings, get_settings
from aitrade.discovery.extractor import TickerExtractor
from aitrade.discovery.scorer import BuzzScorer, DiscoveredTicker

_SYSTEM_PROMPT = """You are a market-discovery analyst. Your job is to identify which
US-listed stock tickers are currently generating the most market buzz.

Use the web_search and web_fetch tools to investigate these queries:
  1. "top stock movers premarket today"
  2. "unusual volume stocks today"
  3. "trending tickers WSB premarket"

For each ticker you find, record:
  - symbol: the uppercase ticker (e.g. "AAPL", "TSLA")
  - mention_count: how many distinct sources or posts mention it
  - source_url: the URL of the most authoritative source you saw it in
  - age_minutes: estimated age of that source in minutes (use 60 if unsure)
  - snippet: a short context snippet (<200 chars) explaining why it's buzzing

CRITICAL OUTPUT FORMAT: After your tool use, your final text response MUST be
a single JSON object — nothing before it, nothing after it, no markdown fences,
no commentary. The exact shape is:

{"tickers": [{"symbol": "...", "mention_count": N, "source_url": "...",
"age_minutes": N, "snippet": "..."}, ...]}

Symbols MUST be uppercase US equity tickers (1-5 letters). Be selective —
quality over quantity. Twenty solid candidates beats fifty noisy ones."""


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class DiscoveryAgent:
    """Use Claude + web tools to surface buzzing tickers, validated and scored.

    The agent is intentionally tolerant of failure: missing API key, network
    errors, malformed JSON, or empty results all yield ``[]`` from
    :meth:`discover` rather than raising. Discovery is a hint, not a contract.

    For tests, swap ``self._client`` with a mock that exposes a ``messages``
    namespace with a ``create`` method returning an object whose ``content``
    is a list of blocks with ``.type == "text"`` and ``.text`` JSON.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        extractor: TickerExtractor | None = None,
        scorer: BuzzScorer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._extractor = extractor
        self._scorer = scorer or BuzzScorer()
        self._client: object | None = None  # lazy; tests may set directly

    def _get_client(self) -> object | None:
        """Construct (and cache) the anthropic client, or ``None`` if no key."""
        if self._client is not None:
            return self._client
        api_key = self._settings.anthropic_api_key.get_secret_value()
        if not api_key:
            logger.warning("DiscoveryAgent: ANTHROPIC_API_KEY missing; returning no candidates")
            return None
        try:
            import anthropic
        except ImportError as e:
            logger.warning("DiscoveryAgent: anthropic SDK not importable: {}", e)
            return None
        self._client = anthropic.Anthropic(api_key=api_key)
        return self._client

    def discover(self, top_n: int = 20) -> list[DiscoveredTicker]:
        """Return up to ``top_n`` discovered tickers, sorted desc by buzz score.

        Always returns a list — empty on any failure. Logs a warning describing
        what went wrong; nothing is raised.
        """
        client = self._get_client()
        if client is None:
            return []

        # Stream the request — web_search can run for 30-90s and a non-streaming
        # connection often gets dropped by intermediate proxies / VPNs mid-flight,
        # surfacing as APIConnectionError. Streaming keeps the channel hot.
        # Even with streaming, transient stream drops (httpx.RemoteProtocolError)
        # do happen on flaky networks; retry up to 3 times with backoff.
        response: object | None = None
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                with client.messages.stream(  # type: ignore[attr-defined]
                    model=self._settings.aitrade_reasoner_model,
                    max_tokens=4096,
                    system=[
                        {
                            "type": "text",
                            "text": _SYSTEM_PROMPT,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Find the most-buzzed US stock tickers right now and "
                                "return them as the JSON object described in the "
                                "system prompt. JSON only, no other text."
                            ),
                        }
                    ],
                    tools=[
                        {"type": "web_search_20260209", "name": "web_search"},
                        {"type": "web_fetch_20260209", "name": "web_fetch"},
                    ],
                    tool_choice={"type": "auto"},
                ) as stream:
                    response = stream.get_final_message()
                break  # success
            except Exception as e:
                last_err = e
                if attempt < 2:
                    logger.warning(
                        "DiscoveryAgent: stream attempt {} failed ({}); retrying…",
                        attempt + 1,
                        type(e).__name__,
                    )
                    time.sleep(2 ** attempt)  # 1s, 2s
                    continue
                tb_tail = traceback.format_exc().splitlines()[-1] if traceback else ""
                logger.warning(
                    "DiscoveryAgent: Claude call failed after 3 attempts: {} ({}) — {}",
                    type(e).__name__,
                    e,
                    tb_tail,
                )
                return []
        if response is None:
            # Belt-and-suspenders — should be unreachable given the loop above.
            logger.warning("DiscoveryAgent: no response captured ({})", last_err)
            return []

        raw = self._extract_json_text(response)
        if raw is None:
            logger.warning("DiscoveryAgent: no JSON text block in response")
            return []

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning(
                "DiscoveryAgent: JSON parse failed: {} (raw head: {!r})",
                e,
                raw[:200],
            )
            return []

        candidates = payload.get("tickers", []) if isinstance(payload, dict) else []
        if not isinstance(candidates, list):
            logger.warning("DiscoveryAgent: tickers field is not a list")
            return []

        discovered = self._build_discovered(candidates)
        discovered.sort(key=lambda d: d.buzz_score, reverse=True)
        return discovered[:top_n]

    def _extract_json_text(self, response: object) -> str | None:
        """Find the JSON object in the response.

        Claude may emit reasoning text mixed with tool use; the final text
        block typically contains the JSON. We:
          1. Walk all text blocks (last to first preferred).
          2. Try direct parse first.
          3. Fall back to grabbing the first ``{...}`` substring via regex.
        """
        content = getattr(response, "content", None)
        if not content:
            return None
        text_blocks: list[str] = []
        for block in content:
            if getattr(block, "type", None) == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    text_blocks.append(text)
        if not text_blocks:
            return None
        # Prefer the last text block — that's typically the final answer
        # after any tool-use reasoning.
        for candidate in reversed(text_blocks):
            stripped = candidate.strip()
            if stripped.startswith("{"):
                return stripped
            match = _JSON_OBJECT_RE.search(stripped)
            if match:
                return match.group(0)
        return None

    def _build_discovered(self, candidates: list[Any]) -> list[DiscoveredTicker]:
        """Validate, score, and dedupe the raw candidate list from Claude."""
        now = datetime.now(UTC)
        seen: dict[str, DiscoveredTicker] = {}

        for item in candidates:
            if not isinstance(item, dict):
                continue
            try:
                symbol = str(item["symbol"]).upper()
                mention_count = int(item["mention_count"])
                source_url = str(item["source_url"])
                age_minutes = float(item["age_minutes"])
                snippet = str(item["snippet"])
            except (KeyError, TypeError, ValueError) as e:
                logger.debug("DiscoveryAgent: skipping malformed candidate {}: {}", item, e)
                continue

            # Optional validation against the tradable universe.
            if self._extractor is not None and symbol not in self._extractor._valid:
                logger.debug("DiscoveryAgent: {} not in tradable universe; skipping", symbol)
                continue

            source = self._scorer.classify_source(source_url)
            weight = self._scorer.SOURCE_WEIGHTS.get(source, self._scorer.SOURCE_WEIGHTS["unknown"])
            buzz = self._scorer.score(mention_count, source, age_minutes)

            existing = seen.get(symbol)
            if existing is None or buzz > existing.buzz_score:
                seen[symbol] = DiscoveredTicker(
                    symbol=symbol,
                    mention_count=mention_count,
                    source_weight=weight,
                    recency_minutes=age_minutes,
                    buzz_score=buzz,
                    discovered_at=now,
                    evidence=[snippet] if snippet else [],
                )
            else:
                existing.evidence.append(snippet)

        return list(seen.values())
