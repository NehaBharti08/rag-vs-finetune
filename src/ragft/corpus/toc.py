"""Canonical section registry, and the statute citation validator built on it.

This is what makes hallucination measurable **without an LLM judge**. The corpus
is a finite, known set of acts and section numbers, so a citation either resolves
against it or it does not. That turns a subjective property ("is this answer
grounded?") into a lookup: free, unbiased, and exactly reproducible.

Legal citation suits this better than the textbook citation it replaces. A
fabricated case or section is a real, consequential failure mode in legal
practice rather than a benchmark curiosity, and the correct form is exact:

    The Bharatiya Nyaya Sanhita, 2023, S103

Statutes are cited by section, never by page, so the page component of the
previous format is gone. The four grading levels survive the change intact,
because section number is the exact analogue of the old chapter.section:

    parseable        produced a citation at all
    act exists       named a real act in the corpus
    section exists   named a real section OF THAT ACT
    section correct  named the section the question came from

Collapsing those into one "validity" number is what hides the interesting
result, so `ragft.eval.metrics.citation` reports all four.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from ragft.corpus.acts import ACTS
from ragft.corpus.parse import Section, load_sections

# `The Bharatiya Nyaya Sanhita, 2023, S103` / `... , section 103` / `... s. 498A`.
# Deliberately permissive about the section marker and about a trailing "of
# 2023": a model that gets the act and section right but writes the marker
# differently has not hallucinated, and scoring it as though it had would
# overstate the very metric this exists to measure.
CITATION_RE = re.compile(
    # Two name shapes, and both must parse. Suffix form covers "Bharatiya Nyaya
    # Sanhita" and "Indian Contract Act"; prefix form covers "Code of Criminal
    # Procedure", where the noun leads.
    #
    # The prefix form matters for a specific measurement. If a repealed-Act
    # citation failed to parse it would be counted as "produced no citation"
    # rather than "cited an act outside the corpus" - understating exactly the
    # behaviour this corpus was chosen to expose, namely a pre-2024 model
    # answering from the IPC, CrPC or Evidence Act.
    r"(?P<act>(?:The\s+)?(?:"
    # U+2019 is deliberate: statute titles carry typographic apostrophes
    # (e.g. "The Bankers' Books Evidence Act"), and a model will reproduce them.
    r"[A-Z][A-Za-z'’\- ]{4,70}?(?:Sanhita|Adhiniyam|Act|Code|Sahita)"  # noqa: RUF001
    r"|Code\s+of\s+[A-Z][A-Za-z'’\- ]{4,60}?"  # noqa: RUF001
    r"))"
    r"\s*,?\s*(?:of\s+)?(?P<year>1[6-9]\d{2}|20\d{2})?"
    # Longest alternatives first: alternation is first-match, so `[Ss]\.` must
    # come after `[Ss]ection` or "section 103" would match on the bare "s".
    r"\s*[,;]?\s*(?:§|[Ss]ection|[Ss]ec\.?|[Ss]\.)\s*" r"(?P<section>\d{1,4}[A-Z]{0,3})",
    re.UNICODE,
)


class CitationVerdict(StrEnum):
    VALID = "valid"
    UNPARSEABLE = "unparseable"
    UNKNOWN_ACT = "unknown_act"
    UNKNOWN_SECTION = "unknown_section"


@dataclass(frozen=True)
class CitationCheck:
    """Outcome of validating one citation."""

    raw: str
    verdict: CitationVerdict
    act_slug: str | None = None
    section_id: str | None = None
    cited_section: str | None = None

    @property
    def is_valid(self) -> bool:
        """True when the cited section genuinely exists in a corpus act."""
        return self.verdict is CitationVerdict.VALID


def _normalise_act(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower().removeprefix("the ")).strip(" ,.")


# Repealed predecessors are deliberately NOT in this lookup: they must resolve
# as `unknown_act`, not as valid. They are listed in `acts.py` under each act's
# `replaces` field for reporting.
_ACT_LOOKUP: dict[str, str] = {}
for _act in ACTS:
    _ACT_LOOKUP[_normalise_act(_act.exact_name)] = _act.slug
    _ACT_LOOKUP[_normalise_act(_act.short_name)] = _act.slug
    # Bare "Nyaya Sanhita" and similar shorthands a model plausibly writes.
    _ACT_LOOKUP[_normalise_act(_act.short_name.split(" ", 1)[-1])] = _act.slug


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
        slug = _ACT_LOOKUP.get(_normalise_act(m.group("act")))
        if slug is None:
            return CitationCheck(raw, CitationVerdict.UNKNOWN_ACT)

        section = m.group("section")
        section_id = f"{slug}:{section}"
        if section_id not in self.by_id:
            return CitationCheck(raw, CitationVerdict.UNKNOWN_SECTION, slug, None, section)

        return CitationCheck(raw, CitationVerdict.VALID, slug, section_id, section)

    def validate_all(self, texts: list[str]) -> dict[str, float | int]:
        """Aggregate validity over many answers."""
        checks = [self.validate(t) for t in texts]
        n = len(checks) or 1
        counts = {v: sum(c.verdict is v for c in checks) for v in CitationVerdict}
        return {
            "n": len(checks),
            "valid_rate": round(counts[CitationVerdict.VALID] / n, 4),
            **{f"n_{v.value}": counts[v] for v in CitationVerdict},
        }


@lru_cache(maxsize=1)
def registry() -> SectionRegistry:
    return SectionRegistry(load_sections())
