"""Build the data file behind the results explorer. Phase 7.

The plan called for "a results-explorer Space over the eval logs plus a local
demo.py, rather than overpromising" a hosted 7B. The demo shipped; this is the
other half.

Verdicts are computed with the project's own `SectionRegistry`, not
re-implemented in JavaScript. A browser-side re-implementation would be a second
definition of the headline metric, free to drift from the one that produced
every number in `reports/` -- which is the same failure the frozen harness
exists to prevent.

Usage::

    uv run python -m ragft.analysis.explorer_data
"""

from __future__ import annotations

import json
from typing import Any

from ragft.corpus.toc import CitationVerdict, registry
from ragft.settings import REPO_ROOT

RESPONSES = REPO_ROOT / "data" / "eval" / "responses"
OUT = REPO_ROOT / "reports" / "explorer_data.json"

ARMS = {
    "A1_base_zeroshot": "A1",
    "A2_base_rag": "A2",
    "A3_ft_zeroshot": "A3",
    "A4_ft_rag": "A4",
}

# Ordered worst-to-best is wrong here; ordered by *how hard the error is to
# catch* is the useful axis, and it is the one the project's finding turns on.
#   correct        - cites the provision the question came from
#   near           - RIGHT act, WRONG section: reads authoritative, needs a lookup
#   wrong_act      - names a real corpus act that is not the right one
#   out_of_corpus  - names an act outside the corpus (usually repealed law)
#   none           - produced no parseable citation
VERDICTS = ("correct", "near", "wrong_act", "out_of_corpus", "none")


def classify(response: str, gold_ids: list[str], reg: Any) -> tuple[str, str | None]:
    check = reg.validate(response)
    if check.verdict is CitationVerdict.UNPARSEABLE:
        return "none", None
    if check.act_slug is None:
        return "out_of_corpus", check.raw
    gold_acts = {g.split(":")[0] for g in gold_ids}
    if check.section_id and check.section_id in gold_ids:
        return "correct", check.raw
    if check.act_slug in gold_acts:
        return "near", check.raw
    return "wrong_act", check.raw


def build() -> dict[str, Any]:
    reg = registry()
    by_id: dict[str, dict[str, Any]] = {}

    for fname, short in ARMS.items():
        path = RESPONSES / f"{fname}.jsonl"
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            item = by_id.setdefault(
                r["gold_id"],
                {
                    "id": r["gold_id"],
                    "q": r["question"],
                    "ref": r["reference"],
                    "stratum": r["stratum"],
                    "gold": r["source_section_ids"],
                    "arms": {},
                },
            )
            verdict, cited = classify(r["response"], r["source_section_ids"], reg)
            item["arms"][short] = {
                "t": r["response"],
                "v": verdict,
                "c": cited,
                # Did retrieval surface the gold section? Only meaningful for
                # the retrieval arms; None elsewhere, and the UI says so.
                "hit": (
                    bool(set(r.get("retrieved_sections") or []) & set(r["source_section_ids"]))
                    if r.get("retrieved_sections")
                    else None
                ),
            }

    items = sorted(by_id.values(), key=lambda i: i["q"])
    counts = {
        short: {
            v: sum(1 for i in items if i["arms"].get(short, {}).get("v") == v) for v in VERDICTS
        }
        for short in ARMS.values()
    }

    # The filter that carries the project's finding: the base model got it
    # right and fine-tuning broke it. Precomputed so the page can offer it
    # without re-deriving the metric client-side.
    broke = [
        i["id"]
        for i in items
        if i["arms"].get("A1", {}).get("v") == "correct"
        and i["arms"].get("A3", {}).get("v") != "correct"
    ]
    fixed = [
        i["id"]
        for i in items
        if i["arms"].get("A2", {}).get("v") != "correct"
        and i["arms"].get("A4", {}).get("v") == "correct"
    ]

    payload = {
        "n": len(items),
        "arms": {
            "A1": "base, no retrieval",
            "A2": "base + RAG",
            "A3": "fine-tuned, no retrieval",
            "A4": "fine-tuned + RAG",
        },
        "counts": counts,
        "ft_broke_it": broke,
        "ft_fixed_it": fixed,
        "items": items,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Trailing newline: the end-of-file-fixer pre-commit hook adds one
    # otherwise, which makes every regeneration dirty the tree and abort a
    # commit that had nothing wrong with it.
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    p = build()
    print(f"{p['n']} items -> {OUT.relative_to(REPO_ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")
    for arm, c in p["counts"].items():
        print(f"  {arm}: " + "  ".join(f"{k}={v}" for k, v in c.items() if v))
    print(f"  fine-tuning BROKE {len(p['ft_broke_it'])} the base model got right")
    print(f"  retrieval FIXED  {len(p['ft_fixed_it'])} that A2 got wrong")


if __name__ == "__main__":
    main()
