"""Abstention pass over the hand-written unanswerable stratum. Phase 6/7.

This is a SEPARATE evaluation from the main 2x2, deliberately.

`data/eval/gold.jsonl` was frozen over 300 answerable items before A3/A4 ran.
Appending 60 unanswerable items to it would change its digest and invalidate
every number already measured against it -- which is precisely what the freeze
exists to prevent. So the unanswerable stratum lives in its own file with its
own freeze, and this runner scores it on its own.

That separation is not a workaround; it is the honest structure. Abstention is
scored by `is_abstention`, which shares no machinery with the citation ladder,
and the two measurements answer different questions. Reporting them from one
file would only create a coupling that costs re-runs.

**False abstention needs answerable questions too.** A model that refuses
everything scores perfect abstention recall, so recall alone is meaningless. A
sample of the answerable gold items is therefore included in the same pass, and
the false-abstention rate is computed over them.

Usage::

    uv run python -m ragft.eval.run_abstention
    uv run python -m ragft.eval.run_abstention --answerable-sample 60
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any

from ragft.eval.arms import ArmRunner, arm_specs, build_prompt
from ragft.eval.metrics.abstention import is_abstention, score
from ragft.eval.metrics.latency import gpu_contention
from ragft.settings import REPO_ROOT

EVAL_DIR = REPO_ROOT / "data" / "eval"
UNANSWERABLE_PATH = EVAL_DIR / "gold_unanswerable.jsonl"
GOLD_PATH = EVAL_DIR / "gold.jsonl"
RESPONSES = EVAL_DIR / "responses_abstention"
REPORTS = REPO_ROOT / "reports"


def _load(path: Any) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def build_items(answerable_sample: int, seed: int = 42) -> list[dict[str, Any]]:
    if not UNANSWERABLE_PATH.exists():
        raise SystemExit(
            f"{UNANSWERABLE_PATH.relative_to(REPO_ROOT)} does not exist.\n"
            "Write the unanswerable stratum first:\n"
            "    uv run python -m ragft.eval.label write"
        )
    unanswerable = _load(UNANSWERABLE_PATH)
    if not unanswerable:
        raise SystemExit("No unanswerable items written yet.")

    # Answerable controls, so false-abstention rate is defined. Sampled with a
    # fixed seed so the pass is reproducible.
    answerable = _load(GOLD_PATH)
    rng = random.Random(seed)
    sample = rng.sample(answerable, min(answerable_sample, len(answerable)))

    items = [{**r, "unanswerable": True} for r in unanswerable]
    items += [{**r, "unanswerable": False} for r in sample]
    return items


def run(adapter: str, answerable_sample: int = 60) -> dict[str, Any]:
    items = build_items(answerable_sample)
    n_unans = sum(i["unanswerable"] for i in items)
    print(f"abstention pass: {n_unans} unanswerable + {len(items) - n_unans} answerable controls")
    print(f"contention: {gpu_contention().to_dict()}")

    runner = ArmRunner()
    RESPONSES.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}

    from ragft.retrieval.retriever import retriever

    for arm in arm_specs(adapter):
        runner.set_adapter(arm.adapter_path)
        ret = retriever() if arm.use_retrieval else None
        rows: list[dict[str, Any]] = []

        for item in items:
            context = ""
            if ret is not None:
                context = ret.format_context(ret.retrieve(item["question"]))
            out = runner.generate(build_prompt(arm, item["question"], context))
            rows.append(
                {
                    "gold_id": item["gold_id"],
                    "question": item["question"],
                    "unanswerable": item["unanswerable"],
                    "kind": item.get("kind"),
                    "response": out["response"],
                    "abstained": is_abstention(out["response"]),
                }
            )

        path = RESPONSES / f"{arm.name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

        stats = score([r["response"] for r in rows], [r["unanswerable"] for r in rows])
        # Which kinds of unanswerable question actually fool it. This is the
        # part a taxonomy buys you over a single rate.
        by_kind: dict[str, dict[str, int]] = {}
        for r in rows:
            if not r["unanswerable"]:
                continue
            k = by_kind.setdefault(str(r["kind"]), {"n": 0, "abstained": 0})
            k["n"] += 1
            k["abstained"] += int(r["abstained"])

        results[arm.name] = {
            **stats.to_dict(),
            "by_kind": {
                k: {**v, "abstention_rate": round(v["abstained"] / v["n"], 4)}
                for k, v in sorted(by_kind.items())
            },
        }
        print(
            f"  {arm.name:20s} recall={stats.recall:.1%} "
            f"precision={stats.precision:.1%} false_abs={stats.false_abstention_rate:.1%}"
        )

    payload = {
        "n_unanswerable": n_unans,
        "n_answerable_controls": len(items) - n_unans,
        "adapter": adapter,
        "arms": results,
        "note": (
            "Scored on a SEPARATE frozen file from the main 2x2 (see module "
            "docstring). Recall alone is meaningless -- a model that refuses "
            "everything scores 100% -- so answerable controls are included and "
            "false_abstention_rate is reported alongside."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "abstention.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {REPORTS / 'abstention.json'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="out/seed42_r16_lr0.0002_e3/checkpoint-354")
    parser.add_argument("--answerable-sample", type=int, default=60)
    args = parser.parse_args()
    run(args.adapter, args.answerable_sample)


if __name__ == "__main__":
    main()
