"""Provider-agnostic LLM client with caching, retries and a cost ledger.

One interface over local Ollama and OpenAI so the generator and judge can be
swapped by configuration rather than by editing call sites. Three behaviours
matter more than the abstraction itself:

* **On-disk response cache.** Generation over ~600 corpus sections is long and
  this box is shared, so runs get interrupted. Caching by a hash of
  (model, prompt, temperature) makes a resumed run skip everything already
  done, and makes re-running after an unrelated code change nearly free.
* **Retry with backoff.** Local Ollama stalls under memory pressure and hosted
  APIs rate-limit. Neither should lose an hour of completed work.
* **A cost ledger.** The project claims the paid path costs about $2. That
  claim should be measured, not asserted, so token usage is tallied per call.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ragft.settings import REPO_ROOT

CACHE_DIR = REPO_ROOT / ".llm_cache"

# USD per 1M tokens. Only the models this project actually uses.
# Local models are free, which is the point of the default path.
PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "text-embedding-3-small": (0.02, 0.0),
}


@dataclass
class Usage:
    """Running tally of tokens and spend."""

    calls: int = 0
    cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        m = self.by_model.setdefault(model, {"calls": 0, "input": 0, "output": 0})
        m["calls"] += 1
        m["input"] += input_tokens
        m["output"] += output_tokens

    @property
    def usd(self) -> float:
        total = 0.0
        for model, counts in self.by_model.items():
            rate_in, rate_out = PRICING.get(model, (0.0, 0.0))
            total += counts["input"] / 1e6 * rate_in + counts["output"] / 1e6 * rate_out
        return round(total, 4)

    def summary(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "cached_calls": self.cached_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "by_model": self.by_model,
            "estimated_usd": self.usd,
        }


def cache_key(model: str, prompt: str, temperature: float, system: str | None) -> str:
    payload = json.dumps(
        {"model": model, "prompt": prompt, "temperature": temperature, "system": system},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LLMClient(ABC):
    """Base class holding cache, retry and accounting logic."""

    def __init__(self, model: str, *, use_cache: bool = True, max_retries: int = 5) -> None:
        self.model = model
        self.use_cache = use_cache
        self.max_retries = max_retries
        self.usage = Usage()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def _complete(
        self, prompt: str, temperature: float, system: str | None
    ) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Provider-specific."""

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        system: str | None = None,
    ) -> str:
        key = cache_key(self.model, prompt, temperature, system)
        cache_file = CACHE_DIR / f"{key}.json"

        if self.use_cache and cache_file.exists():
            self.usage.cached_calls += 1
            cached: dict[str, Any] = json.loads(cache_file.read_text(encoding="utf-8"))
            return str(cached["text"])

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                text, in_tok, out_tok = self._complete(prompt, temperature, system)
            except Exception as exc:  # noqa: BLE001 - retried below, re-raised if terminal
                last_error = exc
                # Exponential backoff, capped. Losing an hour of completed
                # generation to a transient stall is the failure being avoided.
                time.sleep(min(2**attempt, 30))
                continue

            self.usage.record(self.model, in_tok, out_tok)
            if self.use_cache:
                cache_file.write_text(
                    json.dumps({"model": self.model, "text": text}, indent=2),
                    encoding="utf-8",
                )
            return text

        raise RuntimeError(
            f"{type(self).__name__}: {self.max_retries} attempts failed for model "
            f"{self.model!r}. Last error: {last_error}"
        ) from last_error
