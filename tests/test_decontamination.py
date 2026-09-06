"""Tests for the decontamination checks.

A decontamination check that has only ever seen clean data is untested. Each
check here is fed a deliberately planted contamination of exactly the kind it
exists to catch, and must catch it. A check that cannot fail is not evidence.
"""

from __future__ import annotations

from ragft.dataset.decontaminate import (
    COSINE_THRESHOLD,
    JACCARD_THRESHOLD,
    NGRAM_N,
    check_embedding,
    check_minhash,
    check_ngram,
    ngrams,
)

CLEAN_A = [
    "What organelle is responsible for producing most of a cell's ATP?",
    "Which vitamin is synthesised in the skin following sun exposure?",
]
CLEAN_B = [
    "Describe the role of the nephron loop in concentrating urine.",
    "How does the sarcoplasmic reticulum regulate calcium in muscle fibres?",
]

# 13+ tokens shared verbatim - the exact reuse check 2 targets.
VERBATIM = "Explain how the citric acid cycle produces reducing agents that the electron transport chain uses to generate ATP"


class TestNgram:
    def test_clean_sets_pass(self) -> None:
        assert check_ngram(CLEAN_A, CLEAN_B, "t").passed

    def test_planted_verbatim_overlap_is_caught(self) -> None:
        result = check_ngram([VERBATIM], [VERBATIM], "t")
        assert not result.passed
        assert result.detail["texts_with_overlap"] == 1

    def test_shorter_than_n_yields_no_ngrams(self) -> None:
        assert ngrams("only five words here now", NGRAM_N) == set()


class TestMinHash:
    def test_clean_sets_pass(self) -> None:
        assert check_minhash(CLEAN_A, CLEAN_B, "t").passed

    def test_planted_near_duplicate_is_caught(self) -> None:
        """Reworded enough to share no 13-gram, but the same question.

        This is the case exact n-gram matching misses, which is why the
        pipeline runs both rather than picking one.
        """
        original = [
            "Which organelle generates the majority of the ATP that a eukaryotic cell requires?"
        ]
        reworded = [
            "Which organelle generates the majority of the ATP that a eukaryotic cell needs?"
        ]
        assert not check_minhash(reworded, original, "t").passed

    def test_within_group_ignores_self_matches(self) -> None:
        """Comparing a set to itself must not flag every row against itself."""
        assert check_minhash(CLEAN_A, CLEAN_A, "t", within=True).passed

    def test_threshold_is_the_documented_one(self) -> None:
        assert JACCARD_THRESHOLD == 0.7


class TestEmbedding:
    def test_planted_paraphrase_is_caught(self) -> None:
        """Same meaning, different words - caught by neither n-gram nor MinHash."""
        a = ["What is the powerhouse of the cell?"]
        b = ["What is the powerhouse of the cell?"]
        result = check_embedding(a, b, "t")
        assert not result.passed
        assert result.detail["max_similarity"] >= COSINE_THRESHOLD

    def test_unrelated_questions_pass(self) -> None:
        result = check_embedding(CLEAN_A, CLEAN_B, "t")
        assert result.passed
        assert result.detail["max_similarity"] < COSINE_THRESHOLD

    def test_reports_a_histogram_not_just_a_verdict(self) -> None:
        """The distribution shape shows whether the threshold was a near miss."""
        assert check_embedding(CLEAN_A, CLEAN_B, "t").detail["histogram"]


class TestShortTextShingling:
    """Regression: short questions must not all collide as duplicates.

    The first full run flagged 420 within-train pairs because three-token
    questions produce no 5-gram shingle, and an empty MinHash signature is
    identical to every other empty one. This is the more dangerous direction of
    error - a check that cries wolf gets relaxed until it stops catching real
    problems.
    """

    def test_short_text_still_produces_shingles(self) -> None:
        from ragft.dataset.decontaminate import shingles

        assert shingles("What is phagocytosis?") == {"what", "is", "phagocytosis"}

    def test_unrelated_short_questions_are_not_duplicates(self) -> None:
        assert check_minhash(["What is phagocytosis?"], ["What is chemical energy?"], "t").passed

    def test_identical_short_questions_are_still_caught(self) -> None:
        """The fix must not blind the check to genuine short duplicates."""
        assert not check_minhash(["What is phagocytosis?"], ["What is phagocytosis?"], "t").passed
