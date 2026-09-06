"""Label `parametric_answerable` from the base model's own performance.

This is the stratification variable that keeps the 2x2 interpretable (threat 1).
VidyaRAG *excludes* questions answerable without retrieval, which is right for
evaluating RAG and fatal here: it would guarantee the no-retrieval arms fail and
fine-tuning could never win on accuracy.

So the label is **measured, not guessed**: an item is `parametric_answerable`
when the base model answers it correctly with no retrieval at all. That is a
property of the model-corpus pair, not an opinion about the question, and it
partitions the gold set into the two strata results are reported over.

Note the direction of the finding this can produce. If a large share of items
turn out parametrically answerable, Qwen2.5 already knows this corpus - which is
threat 4, and it is the Phase 3 go/no-go gate rather than a nuisance.

Also emits `judge_sample.jsonl`: the stratified sample of responses a human
grades so Cohen's kappa against the judge can be reported.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Any

from ragft.eval.metrics.accuracy import judge_one
from ragft.llmclient import build_client
from ragft.settings import REPO_ROOT, Settings

EVAL_DIR = REPO_ROOT / "data" / "eval"
RESPONSES = EVAL_DIR / "responses" / "A1_base_zeroshot.jsonl"
JUDGE_SAMPLE_N = 100
SAMPLE_SEED = 20260809

# An item counts as parametrically answerable only on a full-credit judgement.
# Partial credit is genuinely ambiguous, and folding it in either direction
# would move the stratum boundary on a coin flip.
ANSWERABLE_MIN_SCORE = 2


def load_responses() -> list[dict[str, Any]]:
    if not RESPONSES.exists():
        raise SystemExit(f"{RESPONSES} missing - run the A1 arm first")
    with RESPONSES.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_scored() -> list[dict[str, Any]]:
    """Whatever has already been judged."""
    out = EVAL_DIR / "answerability.jsonl"
    if not out.exists():
        return []
    with out.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def label(limit: int | None = None) -> dict[str, Any]:
    settings = Settings(_env_file=None)
    judge = build_client(settings, "judge")
    responses = load_responses()
    if limit:
        responses = responses[:limit]

    # Append-and-flush per item so an interrupted run resumes instead of
    # restarting. Writing only at completion meant a stop 7 minutes into a
    # ~30 minute job threw away all 7 minutes.
    out = EVAL_DIR / "answerability.jsonl"
    scored = load_scored()
    done = {r["gold_id"] for r in scored}
    todo = [r for r in responses if r["gold_id"] not in done]

    print(f"judging with {judge.model}: {len(done)} done, {len(todo)} to go")
    with out.open("a", encoding="utf-8") as fh:
        for n, row in enumerate(todo, 1):
            judgement = judge_one(judge, row["question"], row["reference"], row["response"])
            record = {
                **row,
                "judge_score": judgement.score,
                "judge_reason": judgement.reason,
                "judge_parsed": judgement.parsed,
                "parametric_answerable": judgement.score >= ANSWERABLE_MIN_SCORE,
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            fh.flush()
            scored.append(record)
            if n % 25 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}]", flush=True)

    # Stratified sample for human grading: proportional across judge scores, so
    # kappa is not computed on an all-correct sample where a judge that always
    # says "correct" would look good.
    rng = random.Random(SAMPLE_SEED)
    by_score: dict[int, list[dict[str, Any]]] = {}
    for row in scored:
        by_score.setdefault(int(row["judge_score"]), []).append(row)

    sample: list[dict[str, Any]] = []
    for _score, rows in sorted(by_score.items()):
        take = max(1, round(JUDGE_SAMPLE_N * len(rows) / len(scored)))
        sample.extend(rng.sample(rows, min(take, len(rows))))
    rng.shuffle(sample)
    sample = sample[:JUDGE_SAMPLE_N]

    sample_path = EVAL_DIR / "judge_sample.jsonl"
    with sample_path.open("w", encoding="utf-8") as fh:
        for row in sample:
            fh.write(
                json.dumps(
                    {
                        "item_id": row["gold_id"],
                        "arm": row["arm"],
                        "question": row["question"],
                        "reference": row["reference"],
                        "response": row["response"],
                        # judge_score deliberately omitted: showing it would
                        # anchor the human and inflate agreement.
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    n = len(scored) or 1
    answerable = sum(r["parametric_answerable"] for r in scored)
    summary = {
        "judge_model": judge.model,
        "n": len(scored),
        "parametric_answerable": answerable,
        "parametric_answerable_rate": round(answerable / n, 4),
        "score_distribution": dict(Counter(r["judge_score"] for r in scored)),
        "by_stratum": {
            s: round(
                sum(r["parametric_answerable"] for r in scored if r["stratum"] == s)
                / max(1, sum(r["stratum"] == s for r in scored)),
                4,
            )
            for s in sorted({r["stratum"] for r in scored})
        },
        "judge_unparseable": sum(not r["judge_parsed"] for r in scored),
        "judge_sample_written": len(sample),
        "gate_note": (
            "This rate IS threat 4 measured. A high share means Qwen2.5 already "
            "knows the corpus, every cell compresses toward ceiling, and Phase 4 "
            "training compute buys little. That is the Phase 3 go/no-go decision, "
            "taken before the compute is spent rather than after."
        ),
    }
    (EVAL_DIR / "answerability_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    label(args.limit)


if __name__ == "__main__":
    main()
