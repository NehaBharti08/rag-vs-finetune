"""Local Ollama provider: the free default path.

Free, offline, no quota, no signup -- the repo runs end to end without an API
key. The trade-off is a weaker generator and judge than a hosted model, which
is documented rather than hidden: a weak judge is reported as a low
judge/human agreement score, not papered over.
"""

from __future__ import annotations

import httpx

from ragft.llmclient.base import LLMClient


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        use_cache: bool = True,
        max_retries: int = 5,
        timeout: float = 600.0,
    ) -> None:
        super().__init__(model, use_cache=use_cache, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")
        # Generous timeout: a 9.6GB model on a shared GPU can take minutes for a
        # long grounded-generation prompt, and a premature timeout would burn
        # the work rather than wait for it.
        self._client = httpx.Client(timeout=timeout)

    def _complete(
        self, prompt: str, temperature: float, system: str | None
    ) -> tuple[str, int, int]:
        payload: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        resp = self._client.post(f"{self.base_url}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return (
            str(data.get("response", "")),
            int(data.get("prompt_eval_count", 0)),
            int(data.get("eval_count", 0)),
        )

    def health(self) -> bool:
        """True if the daemon is up and this model is actually pulled."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags", timeout=10.0)
            resp.raise_for_status()
            names = {m["name"] for m in resp.json().get("models", [])}
            return self.model in names
        except Exception:  # noqa: BLE001 - health check must never raise
            return False
