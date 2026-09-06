"""The delivered type mix must match the mix that was declared.

This exists because of a defect these tests did not previously catch. The
unanswerable share came out at 1.5% against a declared 10%: generation produced
all 335 items correctly, but the filter rejected 87% of them under
`passage_reference`, because the unanswerable *prompt* instructed the model to
"state what the passage would need to contain" - a phrase the *filter* is built
to reject.

Nothing failed. No exception, no error count, no failing check. The pipeline
reported a healthy 21.4% rejection rate and a decontamination pass, and the
dataset was quietly missing the category the dataset card calls load-bearing.

That is the shape of bug worth guarding against: one component silently
undoing another, visible only if someone reads a distribution. Declaring a mix
and never verifying it is a promise, not a control.

Skipped when the dataset is absent, so CI stays green on a checkout without it.
"""

from __future__ import annotations

import json
from collections import Counter

import pytest

from ragft.dataset.schema import TYPE_MIX, QAType
from ragft.settings import REPO_ROOT

CLEAN = REPO_ROOT / "data" / "qa" / "clean.jsonl"

# Generous: generation yield varies by type and the point is to catch a
# category collapsing, not to police a few percentage points.
REL_TOLERANCE = 0.5


def load_rows() -> list[dict[str, object]]:
    if not CLEAN.exists():
        pytest.skip("dataset not built in this checkout")
    return [json.loads(line) for line in CLEAN.open(encoding="utf-8") if line.strip()]


@pytest.fixture(scope="module")
def shares() -> dict[str, float]:
    rows = load_rows()
    counts = Counter(str(r["qa_type"]) for r in rows)
    return {k: v / len(rows) for k, v in counts.items()}


class TestDeclaredMixIsDelivered:
    @pytest.mark.parametrize("qa_type", list(TYPE_MIX))
    def test_share_is_within_tolerance(self, shares: dict[str, float], qa_type: QAType) -> None:
        target = TYPE_MIX[qa_type]
        actual = shares.get(qa_type.value, 0.0)
        assert actual >= target * (1 - REL_TOLERANCE), (
            f"{qa_type.value} is {actual:.1%} of the dataset against a declared "
            f"{target:.0%}. A category collapsing usually means a generation prompt "
            f"and a filter gate disagree - check reports/dataset_card.md rejections "
            f"by criterion."
        )

    def test_every_declared_type_is_present(self, shares: dict[str, float]) -> None:
        missing = [t.value for t in TYPE_MIX if shares.get(t.value, 0.0) == 0.0]
        assert missing == [], f"declared types absent from the dataset: {missing}"


class TestRefusalTrainingSurvives:
    """Refusal is the category whose absence is least visible and most costly."""

    def test_unanswerable_share_is_material(self, shares: dict[str, float]) -> None:
        actual = shares.get(QAType.UNANSWERABLE.value, 0.0)
        assert actual >= 0.05, (
            f"unanswerable is {actual:.1%}. Without refusal training the fine-tuned "
            f"arm's hallucination rate measures this dataset's omission rather than "
            f"the method, which invalidates the comparison it is meant to support."
        )

    def test_unanswerable_rows_actually_refuse(self) -> None:
        from ragft.dataset.schema import REFUSAL_TEXT

        rows = [r for r in load_rows() if r["qa_type"] == QAType.UNANSWERABLE.value]
        if not rows:
            pytest.skip("no unanswerable rows yet")
        bad = [r for r in rows if REFUSAL_TEXT.lower()[:40] not in str(r["answer"]).lower()]
        assert bad == [], f"{len(bad)} unanswerable rows do not contain the refusal text"


class TestPassageSplitting:
    """Regression tests from the domain switch.

    Both cases were invisible on a textbook corpus whose sections averaged
    ~13,000 characters, and both fired immediately on statutes.
    """

    def _section(self, text: str):  # type: ignore[no-untyped-def]
        from ragft.corpus.parse import Section

        return Section(
            act_slug="test",
            act_name="The Test Act, 2023",
            act_year=2023,
            era="legacy",
            section_id="test:1",
            label="1",
            title="Test.",
            text=text,
            repealed=False,
            ministry="Test",
            char_count=len(text),
        )

    def test_exactly_two_passages_with_a_short_tail_does_not_crash(self) -> None:
        """`passages[-2] = ... passages.pop()` raised IndexError.

        Python evaluates the right-hand side first, so the pop left one element
        before the index was applied. Only reachable when there are exactly two
        passages, which long textbook sections never produced.
        """
        from ragft.dataset.passages import PASSAGE_CHARS, to_passages

        long_part = ("This is a sentence of moderate length about the statute. " * 40)[
            : PASSAGE_CHARS + 200
        ]
        section = self._section(long_part + " Short tail.")
        assert to_passages(section)  # must not raise

    def test_a_typical_statute_section_survives(self) -> None:
        """At the old 500-char floor a median statute section vanished entirely."""
        from ragft.dataset.passages import to_passages

        # ~550 characters: the median for this corpus.
        text = "Whoever commits the offence shall be punished accordingly. " * 9
        assert len(to_passages(self._section(text))) == 1
