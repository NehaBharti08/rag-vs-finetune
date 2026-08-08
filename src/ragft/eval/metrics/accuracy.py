"""Factual accuracy via LLM judge, with bootstrap confidence intervals.

This is the **pre-registered primary metric**, and the only headline metric that
depends on a judge. Everything about it is therefore held to a stricter standard
than the mechanical metrics:

* The judge is never Qwen-family, enforced in `settings.py` (threat 5).
* The judge runs at temperature 0, and its self-consistency is measurable.
* Its agreement with human labels is reported as Cohen's kappa. If that comes
  out below 0.6 the judge is not trustworthy, the analysis says so, and the
  headline leans on the judge-free metrics instead.
* Scores carry bootstrap CIs, because four arms times several strata offers
  many chances for a difference that is really noise (threat 13).

The judge on the free path is a local 9.6 GB model, which is materially weaker
than a hosted one. That is a stated limitation (threat 4b), not a hidden one.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragft.llmclient import LLMClient
from ragft.settings import REPO_ROOT

PROMPT_DIR = REPO_ROOT / "prompts" / "judge"
MAX_SCORE = 2
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


@dataclass(frozen=True)
class Judgement:
    score: int
    reason: str
    parsed: bool

    @property
    def normalised(self) -> float:
        """0.0-1.0, so accuracy reads as a rate rather than a raw 0-2 score."""
        return self.score / MAX_SCORE


def parse_judgement(raw: str) -> Judgement:
    """Extract the verdict, and mark unparseable output rather than guessing.

    An unparseable judge response scored as 0 would silently penalise the arm
    being judged for the judge's own failure, so it is flagged instead.
    """
    m = _JSON_RE.search(raw.strip())
    if not m:
        return Judgement(0, "judge output unparseable", parsed=False)
    try:
        data = json.loads(m.group(0))
        score = int(data.get("score", 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return Judgement(0, "judge output unparseable", parsed=False)
    return Judgement(max(0, min(MAX_SCORE, score)), str(data.get("reason", ""))[:200], True)


def judge_one(client: LLMClient, question: str, reference: str, response: str) -> Judgement:
    prompt = (
        (PROMPT_DIR / "accuracy.md")
        .read_text(encoding="utf-8")
        .format(question=question, reference=reference, response=response)
    )
    return parse_judgement(client.complete(prompt, temperature=0.0))


def bootstrap_ci(
    values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean."""
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 4), round(float(hi), 4))


def aggregate(judgements: list[Judgement]) -> dict[str, Any]:
    scores = [j.normalised for j in judgements]
    unparsed = sum(not j.parsed for j in judgements)
    lo, hi = bootstrap_ci(scores)
    n = len(judgements) or 1
    return {
        "n": len(judgements),
        "accuracy": round(float(np.mean(scores)), 4) if scores else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
        "exact_correct_rate": round(sum(j.score == MAX_SCORE for j in judgements) / n, 4),
        "partial_rate": round(sum(j.score == 1 for j in judgements) / n, 4),
        "incorrect_rate": round(sum(j.score == 0 and j.parsed for j in judgements) / n, 4),
        "judge_unparseable": unparsed,
        "judge_unparseable_rate": round(unparsed / n, 4),
    }
