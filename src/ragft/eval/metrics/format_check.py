"""Format adherence -- mechanical, no judge.

⚠️ This metric is CIRCULAR by construction and must be read as such. The
fine-tuned model is trained on exactly the format scored here, so it scoring
well is a manipulation check -- evidence that training did what it was meant to
-- not a finding. Reporting "fine-tuning wins on format" as a result would be
dishonest (threat 8).

The non-trivial question the format enables is different: *does hitting the
format cost accuracy?*
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ANSWER_RE = re.compile(r"\*\*Answer\.\*\*\s*(.+?)(?=\n\s*\*\*|$)", re.DOTALL)
WHY_RE = re.compile(r"\*\*Why\.\*\*\s*(.+?)(?=\n\s*\*\*|$)", re.DOTALL)
SOURCE_RE = re.compile(r"\*\*Source\.\*\*\s*(.+?)(?=\n\s*\*\*|$)", re.DOTALL)


@dataclass(frozen=True)
class FormatCheck:
    has_answer: bool
    has_why: bool
    has_source: bool
    answer: str
    why: str
    source: str

    @property
    def is_valid(self) -> bool:
        """Answer and Why are required; Source is optional on a refusal."""
        return self.has_answer and self.has_why

    @property
    def is_complete(self) -> bool:
        return self.has_answer and self.has_why and self.has_source


def check_format(response: str) -> FormatCheck:
    a = ANSWER_RE.search(response)
    w = WHY_RE.search(response)
    s = SOURCE_RE.search(response)
    return FormatCheck(
        has_answer=a is not None,
        has_why=w is not None,
        has_source=s is not None,
        answer=a.group(1).strip() if a else "",
        why=w.group(1).strip() if w else "",
        source=s.group(1).strip() if s else "",
    )


def aggregate(responses: list[str]) -> dict[str, float | int]:
    checks = [check_format(r) for r in responses]
    n = len(checks) or 1
    return {
        "n": len(checks),
        "valid_rate": round(sum(c.is_valid for c in checks) / n, 4),
        "complete_rate": round(sum(c.is_complete for c in checks) / n, 4),
        "has_answer_rate": round(sum(c.has_answer for c in checks) / n, 4),
        "has_why_rate": round(sum(c.has_why for c in checks) / n, 4),
        "has_source_rate": round(sum(c.has_source for c in checks) / n, 4),
    }
