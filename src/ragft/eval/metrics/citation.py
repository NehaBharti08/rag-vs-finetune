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

**`act_correct` was added POST-HOC, after seeing A3's responses.** The frozen-eval
pre-commit hook caught it and blocked the commit, which is the hook working
exactly as intended: a metric invented after seeing results is not a metric.

It is kept, under three conditions that are the reason it is not p-hacking:

1. The **pre-registered primary metric is unchanged** -- correct-section rate,
   on an eval set whose digest did not move. A3 still scores 0.0%. No headline
   was rescued by adding this.
2. It is labelled **secondary and exploratory** everywhere it is reported, and
   in `docs/THREATS_TO_VALIDITY.md` as a named threat.
3. It is a **decomposition** of an already-frozen metric, not a replacement for
   one. `act_exists` and `section_correct` were both already frozen; this splits
   the gap between them.

The honest summary is that it explains an existing result rather than changing
one, and a reader is told it was found afterwards.

**`act_correct` is a separate rung from `act_exists`, and adding it changed the
headline of this project.** `act_exists` only asks whether the cited act is real;
`act_correct` asks whether it is the act the answer actually lives in. Collapsing
them hid the single largest effect fine-tuning had here: the adapter moved
act-level routing from 47.7% to 90.3% while section-level accuracy stayed at
zero. Read only at the section rung, fine-tuning looks like it learned nothing.
Read at both, it learned the low-cardinality mapping (4 acts) and none of the
high-cardinality one (~1,090 sections) -- which is a claim about what a thin
adapter can absorb, not a null result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragft.corpus.toc import CitationVerdict, SectionRegistry, registry


@dataclass(frozen=True)
class CitationScore:
    parseable: bool
    act_exists: bool
    # The cited act is the one the gold section belongs to. Distinct from
    # act_exists, which only asks whether the act is real -- see module docstring.
    act_correct: bool
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
        act_correct=bool(
            check.act_slug is not None
            and check.act_slug in {sid.split(":")[0] for sid in source_section_ids}
        ),
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
        "act_correct_rate": round(sum(s.act_correct for s in scores) / n, 4),
        "section_exists_rate": round(sum(s.section_exists for s in scores) / n, 4),
        "section_correct_rate": round(sum(s.section_correct for s in scores) / n, 4),
        # Right statute, wrong section: the most plausible-looking error a legal
        # model can make, and the most dangerous. Separated from fabrication_rate
        # because a wrong-act citation is obviously wrong to a lawyer, while a
        # right-act/wrong-section one reads as authoritative.
        "right_act_wrong_section_rate": round(
            sum(s.act_correct and not s.section_correct for s in scores) / n, 4
        ),
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
            "which on this corpus usually means repealed pre-2024 law. "
            "act_correct_rate = cites the act the answer actually lives in, which "
            "is a strictly weaker claim than section_correct_rate and the rung "
            "where fine-tuning shows its only real gain. POST-HOC and SECONDARY: "
            "added after seeing results, on an unchanged eval set, alongside an "
            "unchanged pre-registered primary metric. See the module docstring."
        ),
    }
