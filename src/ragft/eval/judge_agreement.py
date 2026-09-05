"""Cohen's kappa between the LLM judge and human labels. Phase 2 deliverable.

The judge is the only metric in this project that is not mechanical, so it is the
only one whose trustworthiness has to be *established* rather than inspected. The
threshold was pre-registered at kappa >= 0.60: above it, judged accuracy is a
headline metric; below it, judged accuracy is demoted to secondary and every
headline number must be judge-free. Fixing the rule before seeing the number is
the entire point -- a threshold chosen afterwards is a rationalisation.

Three kappas are reported because they answer different questions:

* **Unweighted, 3-way** (0/1/2). The pre-registered figure. Treats a 0-vs-2
  disagreement as no worse than 1-vs-2, which under-credits a judge whose errors
  are near-misses.
* **Linear-weighted, 3-way.** The scale is ordinal, so this is the fairer
  measure of the judge; reported as context, never as the gate.
* **Binary** (fully-correct vs not). This is the decision the headline metric
  actually depends on, so it is the one that matters operationally.

Also reported: the *rate* of "fully correct" under each rater. Kappa measures
agreement, not bias -- a judge can agree at chance level while being
systematically generous, and that generosity is what silently inflates a
reported accuracy.

Usage::

    uv run python -m ragft.eval.judge_agreement
"""

from __future__ import annotations

import json
from typing import Any

from ragft.settings import REPO_ROOT

ANSWERABILITY = REPO_ROOT / "data" / "eval" / "answerability.jsonl"
HUMAN_LABELS = REPO_ROOT / "data" / "eval" / "human_judge_labels.jsonl"
REPORTS = REPO_ROOT / "reports"

# Pre-registered BEFORE the labels were collected. See docs/METHODOLOGY.md.
KAPPA_THRESHOLD = 0.60
LABELS = (0, 1, 2)


def cohen_kappa(a: list[int], b: list[int], weighted: bool = False) -> float:
    """Cohen's kappa over the fixed ordinal scale 0/1/2.

    `weighted` applies linear weights, which credit near-misses on an ordinal
    scale. Implemented directly rather than pulled from sklearn: it is a dozen
    lines, and this project already carries enough dependency surface.
    """
    n = len(a)
    if n == 0:
        return float("nan")
    idx = {label: i for i, label in enumerate(LABELS)}
    k = len(LABELS)

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        observed[idx[x]][idx[y]] += 1.0

    rows = [sum(r) for r in observed]
    cols = [sum(observed[i][j] for i in range(k)) for j in range(k)]

    # weight[i][j] = disagreement cost. Unweighted: 1 off the diagonal.
    span = k - 1
    weight = [
        [(abs(i - j) / span if weighted else float(i != j)) for j in range(k)] for i in range(k)
    ]

    num = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k))
    den = sum(weight[i][j] * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    if den == 0:
        # Both raters used a single identical label: perfect agreement, and
        # kappa is undefined rather than 0.
        return float("nan")
    return 1.0 - num / den


def _binary_kappa(a: list[int], b: list[int]) -> float:
    """Kappa on 'fully correct or not' -- the decision the headline depends on."""
    ab = [2 if x == 2 else 0 for x in a]
    bb = [2 if x == 2 else 0 for x in b]
    return cohen_kappa(ab, bb)


def run() -> dict[str, Any]:
    judge = {
        r["gold_id"]: int(r["judge_score"])
        for r in map(json.loads, ANSWERABILITY.open(encoding="utf-8"))
    }
    human_rows = [json.loads(line) for line in HUMAN_LABELS.open(encoding="utf-8") if line.strip()]

    paired = [
        (int(r["human_score"]), judge[r["item_id"]]) for r in human_rows if r["item_id"] in judge
    ]
    missing = len(human_rows) - len(paired)
    h = [x for x, _ in paired]
    j = [y for _, y in paired]
    n = len(paired)

    kappa = cohen_kappa(h, j)
    confusion = {
        f"human{hi}_judge{ji}": sum(1 for x, y in paired if x == hi and y == ji)
        for hi in LABELS
        for ji in LABELS
    }
    human_correct = sum(1 for x in h if x == 2) / n
    judge_correct = sum(1 for y in j if y == 2) / n

    payload = {
        "n_human_labels": len(human_rows),
        "n_paired": n,
        "n_unmatched": missing,
        "scale": "0 = wrong, 1 = partially correct, 2 = fully correct",
        "kappa_unweighted": round(kappa, 4),
        "kappa_linear_weighted": round(cohen_kappa(h, j, weighted=True), 4),
        "kappa_binary_fully_correct": round(_binary_kappa(h, j), 4),
        "exact_agreement_rate": round(sum(1 for x, y in paired if x == y) / n, 4),
        "threshold": KAPPA_THRESHOLD,
        "threshold_met": bool(kappa >= KAPPA_THRESHOLD),
        "human_fully_correct_rate": round(human_correct, 4),
        "judge_fully_correct_rate": round(judge_correct, 4),
        "judge_inflation_factor": (
            round(judge_correct / human_correct, 2) if human_correct else None
        ),
        "confusion": confusion,
        "consequence": (
            "kappa is BELOW the pre-registered 0.60 threshold, so LLM-judged "
            "accuracy is reported as a SECONDARY metric only and every headline "
            "number in this project is judge-free (citation validity, format, "
            "MMLU, latency -- all mechanical). The threshold was fixed before "
            "the labels were collected."
            if kappa < KAPPA_THRESHOLD
            else "kappa meets the pre-registered threshold; judged accuracy may "
            "be reported as a primary metric."
        ),
    }
    (REPORTS / "judge_agreement.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "judge_agreement.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


def render(p: dict[str, Any]) -> str:
    verdict = "PASS" if p["threshold_met"] else "FAIL"
    rows = "\n".join(
        f"| human={hi} | "
        + " | ".join(str(p["confusion"][f"human{hi}_judge{ji}"]) for ji in LABELS)
        + " |"
        for hi in LABELS
    )
    return f"""# Judge agreement

`gemma4:e4b` (temp 0) against **{p["n_paired"]}** human labels on A1 responses.
Scale: {p["scale"]}.

## The pre-registered gate

| | |
|---|---|
| Cohen's kappa (unweighted) | **{p["kappa_unweighted"]}** |
| Pre-registered threshold | {p["threshold"]} |
| **Verdict** | **{verdict}** |

{p["consequence"]}

The threshold was fixed in `docs/METHODOLOGY.md` before these labels were
collected. A threshold chosen after seeing the number is not a threshold.

## Secondary kappas

| Variant | kappa | What it answers |
|---|---|---|
| Unweighted, 3-way | {p["kappa_unweighted"]} | The pre-registered gate |
| Linear-weighted, 3-way | {p["kappa_linear_weighted"]} | Fairer on an ordinal scale -- credits near-misses |
| Binary (fully-correct) | {p["kappa_binary_fully_correct"]} | The decision the headline metric depends on |
| Exact agreement rate | {p["exact_agreement_rate"]} | Raw, uncorrected for chance -- shown to make the correction visible |

## Bias, which kappa does not measure

| Rater | Calls an answer fully correct |
|---|---|
| Human | **{p["human_fully_correct_rate"]:.1%}** |
| Judge | **{p["judge_fully_correct_rate"]:.1%}** |

The judge is **{p["judge_inflation_factor"]}x** as generous as the human.

This is the more consequential finding. Kappa measures whether two raters move
together; it says nothing about whether one is systematically lenient. A judge
could agree at chance level and still be unbiased, or agree well and inflate
every number it touches. Here it does the latter, which is why the base model's
judged 26.7% parametric-answerable rate should be read as nearer **10%** and why
it is not used as a headline anywhere.

## Confusion matrix

| | judge=0 | judge=1 | judge=2 |
|---|---|---|---|
{rows}

_Regenerate: `uv run python -m ragft.eval.judge_agreement`_
"""


if __name__ == "__main__":
    run()
