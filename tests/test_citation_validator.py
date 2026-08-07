"""Tests for the judge-free hallucination metric.

The citation validator converts "is this answer grounded?" into a lookup
against a finite registry. If it accepted a fabricated citation, the headline
hallucination number would be wrong in the flattering direction.
"""

from __future__ import annotations

import pytest

from ragft.corpus.toc import CitationVerdict, registry


@pytest.fixture(scope="module")
def reg():  # type: ignore[no-untyped-def]
    return registry()


class TestValidCitations:
    def test_full_form_with_licence(self, reg) -> None:  # type: ignore[no-untyped-def]
        check = reg.validate("**Source.** Biology, §7.3, p.198 (OpenStax, CC BY 4.0)")
        assert check.is_valid
        assert check.has_licence_attribution

    def test_missing_licence_is_valid_but_flagged(self, reg) -> None:  # type: ignore[no-untyped-def]
        """CC BY attribution is a licence obligation, so its absence is recorded."""
        check = reg.validate("**Source.** Biology, §7.3, p.198")
        assert check.is_valid
        assert not check.has_licence_attribution


class TestFabrications:
    def test_fabricated_page_is_rejected(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert reg.validate("Biology, §7.3, p.999").verdict is CitationVerdict.PAGE_OUT_OF_RANGE

    def test_fabricated_section_is_rejected(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert reg.validate("Biology, §99.9, p.198").verdict is CitationVerdict.UNKNOWN_SECTION

    def test_book_outside_the_corpus_is_rejected(self, reg) -> None:  # type: ignore[no-untyped-def]
        """Microbiology is real, but CC BY-NC-SA and deliberately not indexed."""
        assert reg.validate("Microbiology, §2.1, p.50").verdict is CitationVerdict.UNKNOWN_BOOK

    def test_no_citation_at_all(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert reg.validate("Osmosis is water diffusion.").verdict is CitationVerdict.UNPARSEABLE


class TestAggregate:
    def test_valid_rate_over_a_mixed_batch(self, reg) -> None:  # type: ignore[no-untyped-def]
        stats = reg.validate_all(
            [
                "Biology, §7.3, p.198 (OpenStax, CC BY 4.0)",
                "Biology, §7.3, p.198 (OpenStax, CC BY 4.0)",
                "Biology, §7.3, p.999",
                "no citation here",
            ]
        )
        assert stats["n"] == 4
        assert stats["valid_rate"] == 0.5
        assert stats["licence_attribution_rate"] == 0.5


class TestRegistryCoverage:
    def test_registry_is_populated(self, reg) -> None:  # type: ignore[no-untyped-def]
        assert len(reg.section_ids) > 300

    def test_every_section_validates_its_own_citation(self, reg) -> None:  # type: ignore[no-untyped-def]
        """Generated training data cites sections this way, so it must round-trip.

        If this failed, every training example would carry a citation the
        metric would later score as a hallucination.
        """
        bad = [s.section_id for s in reg.sections if not reg.validate(s.citation).is_valid]
        assert bad == []
