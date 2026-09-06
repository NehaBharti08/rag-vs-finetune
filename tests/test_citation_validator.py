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

from ragft.corpus.parse import Section
from ragft.corpus.toc import CitationVerdict, SectionRegistry, registry
from ragft.eval.metrics.citation import aggregate, score_one
from ragft.settings import REPO_ROOT

CORPUS = REPO_ROOT / "data" / "corpus" / "sections.jsonl"


def _section(act_slug: str, act_name: str, year: int, label: str) -> Section:
    return Section(
        act_slug=act_slug,
        act_name=act_name,
        act_year=year,
        era="recodified_2023" if year > 2000 else "legacy",
        section_id=f"{act_slug}:{label}",
        label=label,
        title=f"Test section {label}",
        text="Synthetic text for validator tests.",
        repealed=False,
        ministry="Test",
        char_count=40,
    )


@pytest.fixture(scope="module")
def reg() -> SectionRegistry:
    """A small SYNTHETIC registry, not the real corpus.

    The validator is this project's headline judge-free metric, so it has to be
    tested wherever the code runs -- including CI, where `data/corpus/` does not
    exist because corpus data is gitignored. Building the registry from a handful
    of hand-made sections keeps that coverage; the previous version called
    `registry()` and 17 tests passed only on the author's machine.

    Act NAMES resolve from `ragft.corpus.acts`, which is code rather than data,
    so out-of-corpus and repealed-act behaviour is exercised faithfully here.
    Only claims about the real corpus's SIZE need the real thing -- see
    TestRegistryCoverage.
    """
    return SectionRegistry(
        [
            _section("bns2023", "The Bharatiya Nyaya Sanhita, 2023", 2023, "103"),
            _section("bns2023", "The Bharatiya Nyaya Sanhita, 2023", 2023, "64"),
            _section("bsa2023", "The Bharatiya Sakshya Adhiniyam, 2023", 2023, "66"),
            _section("contract1872", "The Indian Contract Act, 1872", 1872, "11"),
            _section("contract1872", "The Indian Contract Act, 1872", 1872, "73"),
        ]
    )


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

    def test_correct_section_scores_every_level(self, reg: SectionRegistry) -> None:
        s = score_one("**Source.** The Bharatiya Nyaya Sanhita, 2023, §103", ["bns2023:103"], reg)
        assert (s.parseable, s.act_exists, s.section_exists, s.section_correct) == (
            True,
            True,
            True,
            True,
        )

    def test_real_but_wrong_section_is_caught(self, reg: SectionRegistry) -> None:
        """The failure mode a single validity number hides."""
        s = score_one("The Bharatiya Nyaya Sanhita, 2023, §103", ["bns2023:64"], reg)
        assert s.section_exists is True
        assert s.section_correct is False

    def test_repealed_act_fails_at_the_act_level(self, reg: SectionRegistry) -> None:
        s = score_one("The Indian Penal Code, 1860, §302", ["bns2023:103"], reg)
        assert s.parseable is True
        assert s.act_exists is False
        assert s.section_correct is False

    def test_fabrication_and_out_of_corpus_are_distinguished(self, reg: SectionRegistry) -> None:
        agg = aggregate(
            [
                "The Bharatiya Nyaya Sanhita, 2023, §103",  # correct
                "The Bharatiya Nyaya Sanhita, 2023, §103",  # real section, wrong one
                "The Indian Penal Code, 1860, §302",  # repealed act
                "no citation here",
            ],
            [["bns2023:103"], ["bns2023:64"], ["bns2023:103"], ["bns2023:103"]],
            reg,
        )
        assert agg["section_correct_rate"] == 0.25
        assert agg["fabrication_rate"] == 0.25
        assert agg["out_of_corpus_act_rate"] == 0.25
        assert agg["parseable_rate"] == 0.75


class TestRoundTrip:
    """A section's own `citation` property must validate against the registry.

    Runs everywhere, on the fixture. The real-corpus version of the same claim
    is in TestRegistryCoverage and needs the corpus to be built.
    """

    def test_fixture_sections_validate_their_own_citations(self, reg: SectionRegistry) -> None:
        bad = [s.section_id for s in reg.sections if not reg.validate(s.citation).is_valid]
        assert bad == []


@pytest.mark.skipif(not CORPUS.exists(), reason="corpus not built in this checkout")
class TestRegistryCoverage:
    """The only claims here that need the REAL corpus rather than a fixture."""

    def test_registry_is_populated(self) -> None:
        assert len(registry().section_ids) > 1000

    def test_every_real_section_validates_its_own_citation(self) -> None:
        """Every one of the ~1,090 real sections must round-trip.

        Generated training data cites sections this way. If this failed, every
        training example would carry a citation the metric later scores as a
        hallucination.
        """
        real = registry()
        bad = [s.section_id for s in real.sections if not real.validate(s.citation).is_valid]
        assert bad == []
