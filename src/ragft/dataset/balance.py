"""Subsample the filtered set so the delivered mix matches the declared one.

Declaring a mix and hoping generation lands on it has now failed twice in this
project, by two different mechanisms:

* On the biology corpus the unanswerable prompt asked for wording the filter was
  built to reject, so 87% of refusal examples were discarded after generation.
* On this corpus sections were allocated by pair-share while pairs-per-call
  varies by type, so factual over-produced (55.9% against 40%) and unanswerable
  starved (3.5% against 10%).

Both were caught only by reading the distribution. This module makes the mix a
property of the artifact rather than a hope about the pipeline: whatever
generation produces, the published set is subsampled to TYPE_MIX exactly.

Subsampling loses data, so it is bounded by the scarcest type. Anything that
cannot be balanced without falling below `MIN_TOTAL` is reported as a shortfall
to be topped up rather than quietly accepted.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from typing import Any

from ragft.dataset.schema import TYPE_MIX
from ragft.settings import REPO_ROOT

QA_DIR = REPO_ROOT / "data" / "qa"
BALANCE_SEED = 20260902
# Below this the training signal is too thin to be worth balancing for.
MIN_TOTAL = 2500


def plan(counts: Counter[str]) -> tuple[int, dict[str, int]]:
    """Largest total whose per-type quota every type can actually fill."""
    feasible = min(
        (counts.get(t.value, 0) / share for t, share in TYPE_MIX.items() if share > 0),
        default=0.0,
    )
    total = int(feasible)
    return total, {t.value: round(total * share) for t, share in TYPE_MIX.items()}


def run(in_name: str = "clean.jsonl", out_name: str = "balanced.jsonl") -> dict[str, Any]:
    src = QA_DIR / in_name
    rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
    counts = Counter(r["qa_type"] for r in rows)
    total, quota = plan(counts)

    rng = random.Random(BALANCE_SEED)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_type.setdefault(row["qa_type"], []).append(row)

    kept: list[dict[str, Any]] = []
    shortfall: dict[str, int] = {}
    for qa_type, want in quota.items():
        have = by_type.get(qa_type, [])
        if len(have) < want:
            shortfall[qa_type] = want - len(have)
        take = min(want, len(have))
        # Shuffle before slicing so the kept subset is not biased toward
        # whichever sections happened to be generated first.
        pool = list(have)
        rng.shuffle(pool)
        kept.extend(pool[:take])
    rng.shuffle(kept)

    out = QA_DIR / out_name
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    delivered = Counter(r["qa_type"] for r in kept)
    n = len(kept) or 1
    summary = {
        "input_rows": len(rows),
        "kept": len(kept),
        "discarded_as_surplus": len(rows) - len(kept),
        "target_total": total,
        "declared_mix": {t.value: share for t, share in TYPE_MIX.items()},
        "delivered_mix": {k: round(v / n, 4) for k, v in sorted(delivered.items())},
        "counts": dict(sorted(delivered.items())),
        "shortfall": shortfall,
        "below_min_total": len(kept) < MIN_TOTAL,
        "note": (
            "The published set is subsampled to TYPE_MIX exactly, so the mix is a "
            "property of the artifact rather than a hope about generation. Surplus "
            "is discarded; shortfall is reported rather than silently tolerated."
        ),
    }
    (QA_DIR / "balance_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-name", default="clean.jsonl")
    parser.add_argument("--out-name", default="balanced.jsonl")
    args = parser.parse_args()

    s = run(args.in_name, args.out_name)
    print(
        f"input {s['input_rows']:,} -> kept {s['kept']:,} "
        f"(discarded {s['discarded_as_surplus']:,} surplus)"
    )
    for k, v in s["delivered_mix"].items():
        print(f"  {k:14s} {s['counts'][k]:5d}  {v:6.1%}")
    if s["shortfall"]:
        print(f"\n  SHORTFALL (top these up): {s['shortfall']}")
    if s["below_min_total"]:
        print(f"  WARNING: {s['kept']} rows is below MIN_TOTAL={MIN_TOTAL}")


if __name__ == "__main__":
    main()
