"""The QA pair record, and the answer format contract it is trained on."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class QAType(StrEnum):
    FACTUAL = "factual"
    DEFINITION = "definition"
    APPLIED = "applied"
    MULTIHOP = "multihop"
    UNANSWERABLE = "unanswerable"


# Declared BEFORE generation, so the mix cannot be adjusted after seeing which
# arm it favours. Two entries are load-bearing and deliberately pull in
# opposite directions:
#
#   MULTIHOP      favours fine-tuning - chunk retrieval is structurally weak at
#                 synthesising across sections.
#   UNANSWERABLE  is what teaches refusal. Without it the fine-tuned arm's
#                 hallucination rate would measure this dataset's omission
#                 rather than the method.
#
# Omitting either would strawman one side of the comparison.
TYPE_MIX: dict[QAType, float] = {
    QAType.FACTUAL: 0.40,
    QAType.MULTIHOP: 0.20,
    QAType.DEFINITION: 0.15,
    QAType.APPLIED: 0.15,
    QAType.UNANSWERABLE: 0.10,
}

REFUSAL_TEXT = "I don't have enough information in the source material to answer that."


@dataclass
class QAPair:
    """One training example, with full provenance.

    `source_section_ids` and `source_chunk_sha256` are not bookkeeping - they
    are what decontamination checks against. Every claim that a training pair
    and an eval question came from different passages is verified through
    these fields.
    """

    qa_id: str
    qa_type: str
    question: str
    answer: str
    why: str
    citation: str
    source_section_ids: list[str]
    source_chunk_sha256: list[str]
    split: str
    act_slugs: list[str]
    generator_model: str

    def formatted_answer(self) -> str:
        """The answer format contract, adopted verbatim from VidyaRAG.

        The `Source.` line is built from verified section metadata, never from
        the generator. A model asked to write its own citation will sometimes
        write a wrong one, and training on wrong citations would teach the
        exact failure the citation metric is meant to detect.
        """
        return f"**Answer.** {self.answer}\n\n**Why.** {self.why}\n\n**Source.** {self.citation}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["formatted_answer"] = self.formatted_answer()
        return d


def make_qa_id(question: str, section_ids: list[str]) -> str:
    payload = "|".join(sorted(section_ids)) + "||" + question.strip().lower()
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def chunk_sha256(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()
