"""Citation validity -- the judge-free hallucination metric.

Grounding is scored at four increasingly strict levels, because collapsing them
into one number is what hides the finding. The biology run this benchmark
started from made that vivid:

    parseable            99.3%   produced a citation at all
    source exists        99.3%   named a real section of the corpus
    source CORRECT        2.2%   named the section the question came from
    page in range         0.0%   named a page inside that section

A single "citation validity" number reported at either of the first two levels
would have read as near-perfect grounding. It was the opposite: the model had
learned the *shape* of a citation without the content mapping underneath.

The levels survive the move to statutes; only their meaning shifts. Statutes are
cited by section rather than by page, so `page_in_range` is gone and its role --
the strictest, most easily fabricated component -- is taken by the section
number itself. Section numbers are a harsher test than page numbers were: a
model that has read a lot of pre-2024 Indian law will confidently answer with
IPC section numbers, which are not in this corpus at all and resolve as
`unknown_act`.

`fabrication_rate` is the quantity worth watching: cites a REAL act but the
wrong section of it. Plausible, well-formed, and wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragft.corpus.toc import CitationVerdict, SectionRegistry, registry


@dataclass(frozen=True)
class CitationScore:
    parseable: bool
    act_exists: bool
    section_exists: bool
    section_correct: bool
    cited_section: str | None
    verdict: str


def score_one(
    response: str, source_section_ids: list[str], reg: SectionRegistry | None = None
) -> CitationScore:
    reg = reg or registry()
    check = reg.validate(response)
    return CitationScore(
        parseable=check.verdict is not CitationVerdict.UNPARSEABLE,
        # An act was recognised even if the section was invented.
        act_exists=check.verdict in (CitationVerdict.VALID, CitationVerdict.UNKNOWN_SECTION),
        section_exists=check.verdict is CitationVerdict.VALID,
        section_correct=bool(check.section_id and check.section_id in source_section_ids),
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
        "act_exists_rate": round(sum(s.act_exists for s in scores) / n, 4),
        "section_exists_rate": round(sum(s.section_exists for s in scores) / n, 4),
        "section_correct_rate": round(sum(s.section_correct for s in scores) / n, 4),
        # Cites a real act, real section, but not the right one.
        "fabrication_rate": round(
            sum(s.section_exists and not s.section_correct for s in scores) / n, 4
        ),
        # Cites an act outside the corpus. On this corpus that is the signature
        # of a model answering from repealed pre-2024 law (IPC, CrPC, Evidence
        # Act) rather than the statutes now in force.
        "out_of_corpus_act_rate": round(
            sum(s.parseable and not s.act_exists for s in scores) / n, 4
        ),
        "note": (
            "fabrication_rate = cites a REAL section that is not the right one. "
            "out_of_corpus_act_rate = cites an act not in the corpus at all, "
            "which on this corpus usually means repealed pre-2024 law."
        ),
    }
