"""Cohen's kappa between the judge and human labels.

Without this, "the judge is trustworthy" is an assumption doing real work in
the headline metric — and on the free path the judge is a local 9.6 GB model,
so the assumption is doing *more* work than usual.

Kappa rather than raw agreement, because raw agreement is inflated by chance
when one label dominates. If most answers are correct, a judge that always says
"correct" scores high raw agreement and is worthless.

Interpretation used in reporting (Landis & Koch):
    < 0.20 poor | 0.21-0.40 fair | 0.41-0.60 moderate
    0.61-0.80 substantial | > 0.80 almost perfect

Below 0.60 the judge is not trustworthy, and the analysis says so and leans on
the judge-free metrics instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def cohens_kappa(a: list[int], b: list[int], n_classes: int = 3) -> float:
    if len(a) != len(b):
        raise ValueError("label lists must be the same length")
    if not a:
        return 0.0

    matrix = np.zeros((n_classes, n_classes), dtype=np.float64)
    for x, y in zip(a, b, strict=True):
        matrix[x, y] += 1

    n = matrix.sum()
    observed = np.trace(matrix) / n
    expected = float((matrix.sum(axis=0) * matrix.sum(axis=1)).sum()) / (n * n)
    if expected >= 1.0:
        return 1.0
    return round(float((observed - expected) / (1 - expected)), 4)


def interpret(kappa: float) -> str:
    if kappa < 0.20:
        return "poor"
    if kappa < 0.41:
        return "fair"
    if kappa < 0.61:
        return "moderate"
    if kappa < 0.81:
        return "substantial"
    return "almost perfect"


def report(judge_scores: list[int], human_scores: list[int]) -> dict[str, Any]:
    kappa = cohens_kappa(judge_scores, human_scores)
    raw = (
        sum(int(x == y) for x, y in zip(judge_scores, human_scores, strict=True))
        / len(judge_scores)
        if judge_scores
        else 0.0
    )
    trustworthy = kappa >= 0.60
    return {
        "n": len(judge_scores),
        "cohens_kappa": kappa,
        "interpretation": interpret(kappa),
        "raw_agreement": round(raw, 4),
        "judge_trustworthy": trustworthy,
        "verdict": (
            "Judge agreement is adequate; LLM-judged accuracy can carry the headline."
            if trustworthy
            else "Kappa below 0.60. The judge is NOT trustworthy: LLM-judged accuracy "
            "drops to a clearly-labelled secondary metric and the headline rests on "
            "the judge-free measures (citation validity, format, abstention, MMLU)."
        ),
    }
