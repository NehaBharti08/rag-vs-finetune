"""Measure retrieval recall@k independently of any arm.

Threat 7 in docs/THREATS_TO_VALIDITY.md: if retrieval is bad, the RAG arms lose
for reasons that have nothing to do with RAG as an approach, and the headline
conclusion is wrong. A benchmark that reports "fine-tuning beat RAG" while
quietly running a broken retriever has measured its own bug.

So recall is measured **before** any arm runs and reported on its own. Every QA
pair records the section it was generated from, which is ground truth for
"did retrieval find the right place?".

If recall@5 is below ~0.8, the RAG arms' failures are retrieval failures and
the report says so rather than letting a reader infer a conclusion about
method.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ragft.retrieval.retriever import Retriever
from ragft.settings import REPO_ROOT

QA_DIR = REPO_ROOT / "data" / "qa"
REPORTS = REPO_ROOT / "reports"


def measure(split: str = "val", k_values: tuple[int, ...] = (1, 3, 5, 10, 20)) -> dict[str, Any]:
    rows = [
        json.loads(line) for line in (QA_DIR / "clean.jsonl").open(encoding="utf-8") if line.strip()
    ]
    # Multi-hop pairs cite two sections; "the right place" is ambiguous for
    # them, so recall is measured on single-source questions only and the
    # exclusion is reported rather than silently applied.
    subset = [r for r in rows if r["split"] == split and len(r["source_section_ids"]) == 1]
    excluded = sum(1 for r in rows if r["split"] == split and len(r["source_section_ids"]) != 1)

    retriever = Retriever()
    max_k = max(k_values)
    retriever.cfg = retriever.cfg.model_copy(update={"top_k_context": max_k})

    hits: dict[int, int] = dict.fromkeys(k_values, 0)
    ranks: list[int | None] = []

    for row in subset:
        want = row["source_section_ids"][0]
        results = retriever.retrieve(row["question"])
        found = next(
            (i for i, r in enumerate(results, 1) if r.section_id == want),
            None,
        )
        ranks.append(found)
        for k in k_values:
            if found is not None and found <= k:
                hits[k] += 1

    n = len(subset) or 1
    found_ranks = [r for r in ranks if r is not None]
    recall = {f"recall@{k}": round(hits[k] / n, 4) for k in k_values}
    verdict = "OK" if recall["recall@5"] >= 0.8 else "WEAK"

    return {
        "split": split,
        "questions": len(subset),
        "excluded_multihop": excluded,
        **recall,
        "mean_rank_when_found": (
            round(sum(found_ranks) / len(found_ranks), 2) if found_ranks else None
        ),
        "never_found": len(ranks) - len(found_ranks),
        "verdict": verdict,
        "interpretation": (
            "recall@5 >= 0.8: retrieval is adequate, so RAG-arm failures are "
            "attributable to generation rather than retrieval."
            if verdict == "OK"
            else "recall@5 < 0.8: retrieval is the bottleneck. RAG-arm failures are "
            "RETRIEVAL failures and must not be read as a verdict on RAG as a method."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    result = measure(args.split)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "retrieval_recall.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    for k, v in result.items():
        if isinstance(v, (int, float, str)) and k != "interpretation":
            print(f"  {k:24s} {v}")
    print(f"\n{result['interpretation']}")


if __name__ == "__main__":
    main()
