"""Tests for the judge-free hallucination metric.

The citation validator converts "is this answer grounded?" into a lookup against
a finite registry of acts and section numbers. If it accepted a fabricated
citation, the headline hallucination number would be wrong in the flattering
direction.

Legal citation raises one case the textbook corpus never had: a model answering
from **repealed** law. The Indian Penal Code, the Code of Criminal Procedure and
the Indian Evidence Act were replaced in 2023, so a pre-2024 model will
confidently cite them. Those must resolve as out-of-corpus rather than valid.
"""

from __future__ import annotations

import pytest

from ragft.corpus.toc import CitationVerdict, registry
from ragft.eval.metrics.citation import aggregate, score_one


@pytest.fixture(scope="module")
def reg():  # type: ignore[no-untyped-def]
    return registry()


class TestValidCitations:
    def test_canonical_form(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert reg.validate("**Source.** The Bharatiya Nyaya Sanhita, 2023, §103").is_valid

    @pytest.mark.parametrize(
        "text",
        [
            "Source. Bharatiya Nyaya Sanhita, section 103",
            "Source. The Indian Contract Act, 1872, s. 11",
            "Source. The Indian Contract Act, 1872, Sec 73",
            "Source. The Bharatiya Sakshya Adhiniyam, 2023, §66",
        ],
    )
    def test_accepts_the_forms_a_model_actually_writes(self, reg, text) -> None:  # type: ignore[no-untyped-def]
        """Marker and shorthand variation is not hallucination.

        Scoring `s. 11` as unparseable would inflate the very metric this
        exists to measure.
        """
        assert reg.validate(text).is_valid

    def test_lettered_section_labels(self, reg) -> None:  # type: ignore[no-untyped-def]
        """Indian statutes carry inserted sections such as 498A - labels are strings."""
        check = reg.validate("The Bharatiya Nyaya Sanhita, 2023, §9999A")
        assert check.verdict is CitationVerdict.UNKNOWN_SECTION
        assert check.cited_section == "9999A"


class TestFabrications:
    def test_fabricated_section_is_rejected(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert (
            reg.validate("The Bharatiya Nyaya Sanhita, 2023, §9999").verdict
            is CitationVerdict.UNKNOWN_SECTION
        )

    @pytest.mark.parametrize(
        "repealed",
        [
            "The Indian Penal Code, 1860, §302",
            "The Code of Criminal Procedure, 1973, §154",
            "The Indian Evidence Act, 1872, §65B",
        ],
    )
    def test_repealed_pre_2023_law_is_out_of_corpus(self, reg, repealed) -> None:  # type: ignore[no-untyped-def]
        """The signature failure of a pre-2024 model on this corpus.

        Answering with IPC 302 for murder is the trained-in reflex. It is not a
        valid citation against statutes now in force, and must not score as one.
        """
        assert reg.validate(repealed).verdict is CitationVerdict.UNKNOWN_ACT

    def test_no_citation_at_all(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert reg.validate("Murder is punishable by death.").verdict is CitationVerdict.UNPARSEABLE


class TestGradedLevels:
    """Existence and correctness are different questions.

    In the biology run of this benchmark 99.3% of citations named a real
    section and only 2.2% named the right one. A metric stopping at "exists"
    would have reported near-perfect grounding.
    """

    def test_correct_section_scores_every_level(self) -> None:
        s = score_one("**Source.** The Bharatiya Nyaya Sanhita, 2023, §103", ["bns2023:103"])
        assert (s.parseable, s.act_exists, s.section_exists, s.section_correct) == (
            True,
            True,
            True,
            True,
        )

    def test_real_but_wrong_section_is_caught(self) -> None:
        """The failure mode a single validity number hides."""
        s = score_one("The Bharatiya Nyaya Sanhita, 2023, §103", ["bns2023:64"])
        assert s.section_exists is True
        assert s.section_correct is False

    def test_repealed_act_fails_at_the_act_level(self) -> None:
        s = score_one("The Indian Penal Code, 1860, §302", ["bns2023:103"])
        assert s.parseable is True
        assert s.act_exists is False
        assert s.section_correct is False

    def test_fabrication_and_out_of_corpus_are_distinguished(self) -> None:
        agg = aggregate(
            [
                "The Bharatiya Nyaya Sanhita, 2023, §103",  # correct
                "The Bharatiya Nyaya Sanhita, 2023, §103",  # real section, wrong one
                "The Indian Penal Code, 1860, §302",  # repealed act
                "no citation here",
            ],
            [["bns2023:103"], ["bns2023:64"], ["bns2023:103"], ["bns2023:103"]],
        )
        assert agg["section_correct_rate"] == 0.25
        assert agg["fabrication_rate"] == 0.25
        assert agg["out_of_corpus_act_rate"] == 0.25
        assert agg["parseable_rate"] == 0.75


class TestRegistryCoverage:
    def test_registry_is_populated(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert len(reg.section_ids) > 1000

    def test_every_section_validates_its_own_citation(self, reg) -> None:  # type: ignore[no-untyped-def]
        """Generated training data cites sections this way, so it must round-trip.

        If this failed, every training example would carry a citation the
        metric would later score as a hallucination.
        """
        bad = [s.section_id for s in reg.sections if not reg.validate(s.citation).is_valid]
        assert bad == []
