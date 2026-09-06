"""Citation validity -- the judge-free hallucination metric.

Grounding is scored at four increasingly strict levels, because collapsing them
into one number hides the finding. Measured on the baseline arm before any
training:

    parseable            99.3%   produces a citation at all
    section exists       99.3%   cites a real section of the corpus
    section CORRECT       2.2%   cites the section the question came from
    page in range         0.0%   cites a page inside that section

A single "citation validity" number reported at the first two levels would read
as near-perfect grounding. It is the opposite: the model has learned the
*shape* of an OpenStax citation - plausible chapter.section numbering, a
plausible page, the licence suffix - without the content mapping underneath.
Confident, well-formatted, and almost entirely fabricated.

This is why the answer format carries a **page-level** citation rather than a
section-level one. A section-only format would have scored ~99% here and hidden
the entire phenomenon.

The levels are defined before the harness is frozen. After freezing, a new
level would be a metric invented after seeing results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragft.corpus.toc import CitationVerdict, SectionRegistry, registry


@dataclass(frozen=True)
class CitationScore:
    parseable: bool
    section_exists: bool
    section_correct: bool
    page_in_range: bool
    has_licence: bool
    cited_section: str | None
    verdict: str


def score_one(
    response: str, source_section_ids: list[str], reg: SectionRegistry | None = None
) -> CitationScore:
    reg = reg or registry()
    check = reg.validate(response)
    parseable = check.verdict is not CitationVerdict.UNPARSEABLE
    exists = check.section_id in reg.by_id if check.section_id else False
    return CitationScore(
        parseable=parseable,
        section_exists=exists,
        section_correct=bool(check.section_id and check.section_id in source_section_ids),
        page_in_range=check.is_valid,
        has_licence=check.has_licence_attribution,
        cited_section=check.section_id,
        verdict=check.verdict.value,
    )


def aggregate(
    responses: list[str], source_ids: list[list[str]], reg: SectionRegistry | None = None
) -> dict[str, Any]:
    reg = reg or registry()
    scores = [score_one(r, s, reg) for r, s in zip(responses, source_ids, strict=True)]
    n = len(scores) or 1
    return {
        "n": len(scores),
        "parseable_rate": round(sum(s.parseable for s in scores) / n, 4),
        "section_exists_rate": round(sum(s.section_exists for s in scores) / n, 4),
        "section_correct_rate": round(sum(s.section_correct for s in scores) / n, 4),
        "page_in_range_rate": round(sum(s.page_in_range for s in scores) / n, 4),
        "licence_attribution_rate": round(sum(s.has_licence for s in scores) / n, 4),
        "fabrication_rate": round(
            sum(s.section_exists and not s.section_correct for s in scores) / n, 4
        ),
        "note": (
            "fabrication_rate is the share citing a REAL section that is not the "
            "right one - a plausible, well-formed, wrong citation. It is the "
            "quantity that a single 'citation validity' number would hide."
        ),
    }
