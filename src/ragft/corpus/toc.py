"""Canonical section registry, and the citation validator built on it.

This is the machinery that makes hallucination measurable **without an LLM
judge**. The corpus has a finite, known set of sections and page ranges, so a
citation either resolves against it or it does not. That turns a subjective
property ("is this answer grounded?") into a lookup: free, unbiased, and
exactly reproducible across runs.

It is also why the answer format carries a page-level citation at all. A model
that invents `Biology, S7.3, p.999` is detectably wrong even when the prose
around it is fluent and plausible -- which is precisely the failure mode a
fine-tuned model is expected to exhibit when it has to recall citations from
parameters instead of reading them off retrieved context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from ragft.corpus.books import BOOKS
from ragft.corpus.parse import Section, load_sections

# `Biology, S7.3, p.214 (OpenStax, CC BY 4.0)` - the trailing licence block is
# optional when parsing (a model may drop it) but its absence is itself
# reported, because CC BY attribution is a licence obligation and a missing
# credit is a real defect rather than a formatting nit.
CITATION_RE = re.compile(
    r"(?P<book>[A-Za-z][A-Za-z &]+?)\s*,\s*"
    r"(?:§|S)\s*(?P<chapter>\d+)\.(?P<number>\d+)\s*,\s*"
    r"p\.?\s*(?P<page>\d+)"
    r"(?P<licence>\s*\(OpenStax,\s*CC\s*BY\s*4\.0\))?",
    re.IGNORECASE,
)

_TITLE_TO_SLUG = {b.short_title.lower(): b.slug for b in BOOKS}


class CitationVerdict(StrEnum):
    VALID = "valid"
    UNPARSEABLE = "unparseable"
    UNKNOWN_BOOK = "unknown_book"
    UNKNOWN_SECTION = "unknown_section"
    PAGE_OUT_OF_RANGE = "page_out_of_range"


@dataclass(frozen=True)
class CitationCheck:
    """Outcome of validating one citation."""

    raw: str
    verdict: CitationVerdict
    section_id: str | None = None
    cited_page: int | None = None
    expected_pages: tuple[int, int] | None = None
    has_licence_attribution: bool = False

    @property
    def is_valid(self) -> bool:
        return self.verdict is CitationVerdict.VALID


class SectionRegistry:
    """Every section in the corpus, indexed for validation."""

    def __init__(self, sections: list[Section]) -> None:
        self.sections = sections
        self.by_id = {s.section_id: s for s in sections}

    @property
    def section_ids(self) -> set[str]:
        return set(self.by_id)

    def validate(self, text: str) -> CitationCheck:
        """Validate the first citation found in ``text``."""
        m = CITATION_RE.search(text)
        if not m:
            return CitationCheck(raw=text.strip()[:120], verdict=CitationVerdict.UNPARSEABLE)

        raw = m.group(0)
        has_licence = m.group("licence") is not None
        slug = _TITLE_TO_SLUG.get(m.group("book").strip().lower())
        if slug is None:
            return CitationCheck(
                raw, CitationVerdict.UNKNOWN_BOOK, has_licence_attribution=has_licence
            )

        section_id = f"{slug}:{m.group('chapter')}.{m.group('number')}"
        section = self.by_id.get(section_id)
        if section is None:
            return CitationCheck(
                raw,
                CitationVerdict.UNKNOWN_SECTION,
                section_id,
                has_licence_attribution=has_licence,
            )

        page = int(m.group("page"))
        expected = (section.printed_page_start, section.printed_page_end)
        # A one-page tolerance: sections legitimately begin mid-page, so a
        # reader could reasonably cite either side of a boundary. Wider than
        # that is a fabricated page, not a rounding disagreement.
        if not (expected[0] - 1 <= page <= expected[1] + 1):
            return CitationCheck(
                raw, CitationVerdict.PAGE_OUT_OF_RANGE, section_id, page, expected, has_licence
            )

        return CitationCheck(raw, CitationVerdict.VALID, section_id, page, expected, has_licence)

    def validate_all(self, texts: list[str]) -> dict[str, float | int]:
        """Aggregate validity over many answers - the hallucination metric."""
        checks = [self.validate(t) for t in texts]
        n = len(checks) or 1
        counts = {v: sum(c.verdict is v for c in checks) for v in CitationVerdict}
        return {
            "n": len(checks),
            "valid_rate": round(counts[CitationVerdict.VALID] / n, 4),
            "licence_attribution_rate": round(
                sum(c.has_licence_attribution for c in checks) / n, 4
            ),
            **{f"n_{v.value}": counts[v] for v in CitationVerdict},
        }


@lru_cache(maxsize=1)
def registry() -> SectionRegistry:
    return SectionRegistry(load_sections())
