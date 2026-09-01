"""The retrieval configuration is held fixed, and the reason has changed.

It used to be justified as a mirror of VidyaRAG's frozen `baseline` profile, so
that this benchmark's RAG arm was the same pipeline as the author's RAG project
rather than a lookalike. **That justification is void.** VidyaRAG indexes
OpenStax biology; this corpus is Indian statutes, so there is no longer an
apples-to-apples claim to support.

The configuration is nonetheless still pinned, for a different and still-real
reason: retrieval must be held constant so the legal and biology runs of this
benchmark can be compared *to each other*. A silent change to chunk size or
top-k would make the two domains incomparable while every number still rendered
fine.

Values are written out longhand rather than read from the config file. A test
that loads the same file it is checking passes no matter what that file says.
"""

from __future__ import annotations

from ragft.settings import load_retrieval_config

# Carried over from the biology run of this benchmark, unchanged.
PINNED = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "respect_sentence_boundaries": True,
    "top_k_retrieve": 20,
    "top_k_context": 5,
    "use_hybrid": False,
    "use_reranker": False,
    "use_decomposition": False,
}


class TestHeldFixedAcrossDomains:
    def test_chunking_unchanged(self) -> None:
        c = load_retrieval_config().chunking
        assert c.chunk_size == PINNED["chunk_size"]
        assert c.chunk_overlap == PINNED["chunk_overlap"]
        assert c.respect_sentence_boundaries == PINNED["respect_sentence_boundaries"]

    def test_top_k_unchanged(self) -> None:
        cfg = load_retrieval_config()
        assert cfg.top_k_retrieve == PINNED["top_k_retrieve"]
        assert cfg.top_k_context == PINNED["top_k_context"]

    def test_enhancements_all_off(self) -> None:
        """The RAG arm must stay a clean control, not a tuned stack.

        Comparing a fully-tuned RAG pipeline against an untuned fine-tune would
        be its own confound.
        """
        cfg = load_retrieval_config()
        assert cfg.use_hybrid is False
        assert cfg.use_reranker is False
        assert cfg.use_decomposition is False
