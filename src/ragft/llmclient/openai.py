"""OpenAI provider: the optional ~$2 path.

Uses the same models VidyaRAG does (`gpt-4o-mini`, `text-embedding-3-small`),
which is the point -- it makes the retrieval arms bit-identical between the two
projects rather than merely similar, and gives a stronger, more defensible
judge. Cost is tallied by the base class rather than asserted.
"""

from __future__ import annotations

from typing import Any

from ragft.llmclient.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        use_cache: bool = True,
        max_retries: int = 5,
    ) -> None:
        super().__init__(model, use_cache=use_cache, max_retries=max_retries)
        from openai import OpenAI

        # max_retries=0: retry/backoff is handled in the base class so that the
        # cache and the cost ledger see exactly one accounted attempt.
        self._client = OpenAI(api_key=api_key, max_retries=0)

    def _complete(
        self, prompt: str, temperature: float, system: str | None
    ) -> tuple[str, int, int]:
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content or "",
            usage.prompt_tokens if usage else 0,
            usage.completion_tokens if usage else 0,
        )
