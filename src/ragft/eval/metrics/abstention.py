"""Abstention behaviour -- mechanical, no judge.

Correctly declining to answer is not something a faithfulness score captures,
so it is measured separately, using VidyaRAG's definitions so the two projects
report the same quantities:

* **precision** -- of the questions the system declined, how many were genuinely
  unanswerable.
* **recall** -- of the genuinely unanswerable questions, how many it declined.
* **false abstention rate** -- answerable questions wrongly refused. The cost of
  over-abstaining, which a recall-only view hides entirely.

A model that refuses everything scores perfect recall, which is why all three
are reported together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ragft.dataset.schema import REFUSAL_TEXT

# The trained refusal string, plus the common ways a model expresses the same
# thing without having been trained to. Both count as abstention: the metric is
# about behaviour, not about reciting a phrase.
_REFUSAL_PATTERNS = re.compile(
    r"(?i)"
    + re.escape(REFUSAL_TEXT[:40])
    + r"|i don'?t have enough information"
    + r"|not enough information (?:in|to)"
    + r"|(?:the )?source material does not (?:cover|contain|provide|include)"
    + r"|cannot be answered (?:from|with|using)"
    + r"|i (?:cannot|can'?t) answer"
    + r"|no information (?:is )?(?:available|provided)"
)


def is_abstention(response: str) -> bool:
    return bool(_REFUSAL_PATTERNS.search(response))


@dataclass(frozen=True)
class AbstentionStats:
    n: int
    n_unanswerable: int
    true_abstentions: int
    false_abstentions: int
    missed_abstentions: int

    @property
    def precision(self) -> float:
        declined = self.true_abstentions + self.false_abstentions
        return round(self.true_abstentions / declined, 4) if declined else 0.0

    @property
    def recall(self) -> float:
        return round(self.true_abstentions / self.n_unanswerable, 4) if self.n_unanswerable else 0.0

    @property
    def false_abstention_rate(self) -> float:
        answerable = self.n - self.n_unanswerable
        return round(self.false_abstentions / answerable, 4) if answerable else 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "n_unanswerable": self.n_unanswerable,
            "abstention_precision": self.precision,
            "abstention_recall": self.recall,
            "false_abstention_rate": self.false_abstention_rate,
            "true_abstentions": self.true_abstentions,
            "false_abstentions": self.false_abstentions,
            "missed_abstentions": self.missed_abstentions,
        }


def score(responses: list[str], unanswerable_flags: list[bool]) -> AbstentionStats:
    if len(responses) != len(unanswerable_flags):
        raise ValueError("responses and flags must be the same length")

    true_abs = false_abs = missed = 0
    for response, unanswerable in zip(responses, unanswerable_flags, strict=True):
        declined = is_abstention(response)
        if unanswerable and declined:
            true_abs += 1
        elif unanswerable and not declined:
            missed += 1
        elif not unanswerable and declined:
            false_abs += 1

    return AbstentionStats(
        n=len(responses),
        n_unanswerable=sum(unanswerable_flags),
        true_abstentions=true_abs,
        false_abstentions=false_abs,
        missed_abstentions=missed,
    )
