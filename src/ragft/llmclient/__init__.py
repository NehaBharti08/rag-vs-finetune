"""Provider-agnostic LLM access.

`build_client` is the only place a provider is chosen, so generation and
judging code never names one.
"""

from __future__ import annotations

from ragft.llmclient.base import CACHE_DIR, LLMClient, Usage
from ragft.llmclient.ollama import OllamaClient
from ragft.settings import Provider, Settings

__all__ = ["CACHE_DIR", "LLMClient", "OllamaClient", "Usage", "build_client"]


def build_client(settings: Settings, role: str = "generation", **kwargs: object) -> LLMClient:
    """Construct the client for a role ("generation" or "judge").

    The family-confound check lives in `Settings` and has already run by the
    time this is called: a Qwen generator or judge cannot reach here, because
    the settings object refuses to construct.
    """
    model = settings.generation_model if role == "generation" else settings.judge_model

    if settings.provider is Provider.OPENAI:
        from ragft.llmclient.openai import OpenAIClient

        assert settings.openai_api_key is not None  # enforced by Settings validator
        return OpenAIClient(model, api_key=settings.openai_api_key, **kwargs)  # type: ignore[arg-type]

    return OllamaClient(model, base_url=settings.ollama_base_url, **kwargs)  # type: ignore[arg-type]
