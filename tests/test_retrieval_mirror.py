"""Guards the mirror of VidyaRAG's frozen `baseline` profile.

The retrieval arms of this benchmark claim to be *the same pipeline* as
VidyaRAG's, not a lookalike. That claim is the whole apples-to-apples argument,
and it is exactly the kind of thing that rots silently: someone tunes a chunk
size in one repo six weeks from now and every reported delta quietly stops
being comparable.

So the expected values are written out longhand below rather than read from
`configs/retrieval.yaml`. A test that loads the same file it is checking would
pass no matter what the file said.

Upstream source:
    https://github.com/NehaBharti08/VidyaRAG
    config/default.yaml + config/profiles/baseline.yaml
"""

from __future__ import annotations

import pytest

from ragft.settings import load_retrieval_config

# Transcribed by hand from VidyaRAG's committed config. Changing a number here
# is a deliberate act that shows up in a diff and needs justifying in a PR.
UPSTREAM_BASELINE = {
    "chunk_size": 512,
    "chunk_overlap": 64,
    "respect_sentence_boundaries": True,
    "top_k_retrieve": 20,
    "top_k_context": 5,
    "use_hybrid": False,
    "use_reranker": False,
    "use_decomposition": False,
}


class TestMirrorsUpstream:
    def test_chunking_matches(self) -> None:
        chunking = load_retrieval_config().chunking
        assert chunking.chunk_size == UPSTREAM_BASELINE["chunk_size"]
        assert chunking.chunk_overlap == UPSTREAM_BASELINE["chunk_overlap"]
        assert (
            chunking.respect_sentence_boundaries == UPSTREAM_BASELINE["respect_sentence_boundaries"]
        )

    def test_top_k_matches(self) -> None:
        cfg = load_retrieval_config()
        assert cfg.top_k_retrieve == UPSTREAM_BASELINE["top_k_retrieve"]
        assert cfg.top_k_context == UPSTREAM_BASELINE["top_k_context"]

    def test_enhancements_are_all_off(self) -> None:
        """The RAG arm must be the frozen control, not VidyaRAG's best stack.

        Comparing a fully-tuned RAG pipeline against an untuned fine-tune would
        be its own confound. The 2x2 needs a clean control on this axis.
        """
        cfg = load_retrieval_config()
        assert cfg.use_hybrid is False
        assert cfg.use_reranker is False
        assert cfg.use_decomposition is False

    def test_points_at_the_right_upstream(self) -> None:
        cfg = load_retrieval_config()
        assert cfg.upstream_repo == "NehaBharti08/VidyaRAG"
        assert cfg.upstream_profile == "baseline"


class TestProvenance:
    @pytest.mark.xfail(
        reason="upstream_commit is pinned in Phase 2, once VidyaRAG's ingest pipeline exists",
        strict=False,
    )
    def test_upstream_commit_is_pinned(self) -> None:
        """A mirror with no recorded source commit is a mirror of nothing.

        Expected to fail until Phase 2. Kept visible rather than omitted so the
        gap is tracked instead of forgotten.
        """
        assert load_retrieval_config().upstream_commit is not None
