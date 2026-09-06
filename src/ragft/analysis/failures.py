"""Failure taxonomy, built from the eval logs rather than from the summary. Phase 6.

An aggregate accuracy number says an arm was wrong; it does not say *where* the
pipeline broke. For the retrieval arms that distinction is the whole analysis,
because retrieval failure and generation failure have opposite fixes: one is a
better index, the other is a better model. A benchmark that cannot separate them
cannot tell you which to buy.

Every item is assigned to exactly one cell of retrieved-the-source x
cited-it-correctly. The `missed source, still correct` cell is the interesting
one: it is the only place parametric knowledge can show up on top of retrieval,
and if it is empty then the model's weights contributed nothing the index did
not already supply.

For the no-retrieval arms the taxonomy is different -- there is no retrieval to
blame -- so failures are graded by how close the citation got: right act, wrong
act but real, or an act outside the corpus entirely.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from ragft.corpus.toc import CitationVerdict, registry
from ragft.settings import REPO_ROOT

RESPONSES = REPO_ROOT / "data" / "eval" / "responses"
REPORTS = REPO_ROOT / "reports"


def _rows(arm: str) -> list[dict[str, Any]]:
    path = RESPONSES / f"{arm}.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def classify_arm(arm: str) -> dict[str, Any]:
    reg = registry()
    rows = _rows(arm)
    n = len(rows) or 1
    has_retrieval = any(r.get("retrieved_sections") for r in rows)

    cells: Counter[str] = Counter()
    examples: dict[str, dict[str, Any]] = {}

    for r in rows:
        gold = set(r["source_section_ids"])
        retrieved = set(r.get("retrieved_sections") or [])
        check = reg.validate(r["response"])
        correct = bool(check.section_id and check.section_id in gold)

        if has_retrieval:
            hit = bool(retrieved & gold)
            if hit and correct:
                cell = "retrieved_and_correct"
            elif hit:
                cell = "generation_failure"  # the section was in context and it still missed
            elif correct:
                cell = "parametric_recovery"  # retrieval missed, weights saved it
            else:
                cell = "retrieval_failure"
        else:
            if correct:
                cell = "correct"
            elif check.verdict is CitationVerdict.UNPARSEABLE:
                cell = "no_citation"
            elif check.act_slug is None:
                cell = "out_of_corpus_act"  # usually repealed pre-2024 law
            elif check.act_slug in {sid.split(":")[0] for sid in gold}:
                cell = "right_act_wrong_section"
            else:
                cell = "wrong_act"

        cells[cell] += 1
        # Keep one verbatim example per cell. Concrete beats summarised.
        if cell not in examples:
            examples[cell] = {
                "question": r["question"],
                "reference": r["reference"],
                "gold_sections": sorted(gold),
                "cited": check.raw,
                "response": r["response"],
            }

    return {
        "arm": arm,
        "n": len(rows),
        "has_retrieval": has_retrieval,
        "counts": dict(cells.most_common()),
        "rates": {k: round(v / n, 4) for k, v in cells.most_common()},
        "examples": examples,
    }


def build() -> dict[str, Any]:
    arms = [p.stem for p in sorted(RESPONSES.glob("*.jsonl"))]
    payload: dict[str, Any] = {"arms": {a: classify_arm(a) for a in arms}}

    rag = [a for a in arms if payload["arms"][a]["has_retrieval"]]
    if len(rag) == 2:
        a, b = rag
        ca, cb = payload["arms"][a]["counts"], payload["arms"][b]["counts"]
        payload["retrieval_ceiling"] = {
            "arms": rag,
            "retrieval_failures": {
                a: ca.get("retrieval_failure", 0),
                b: cb.get("retrieval_failure", 0),
            },
            "identical": ca.get("retrieval_failure", 0) == cb.get("retrieval_failure", 0),
            "generation_failures": {
                a: ca.get("generation_failure", 0),
                b: cb.get("generation_failure", 0),
            },
            "note": (
                "Both retrieval arms share one index, so retrieval failures should be "
                "identical between them. When they are, any difference in accuracy is "
                "attributable to generation alone -- which is what isolates the "
                "adapter's contribution from the retriever's."
            ),
        }

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "failures.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "failures.md").write_text(render(payload), encoding="utf-8")
    return payload


def _example_block(title: str, ex: dict[str, Any]) -> list[str]:
    resp = ex["response"].strip()
    if len(resp) > 700:
        resp = resp[:700] + " …"
    return [
        f"**{title}**",
        "",
        f"> **Q.** {ex['question']}",
        f"> **Gold.** {ex['reference']} — `{', '.join(ex['gold_sections'])}`",
        f"> **Cited.** `{ex['cited']}`",
        "",
        "```",
        resp,
        "```",
        "",
    ]


def render(p: dict[str, Any]) -> str:
    lines = [
        "# Failure taxonomy",
        "",
        "Generated by `ragft.analysis.failures`. **Do not hand-edit.**",
        "",
        "Every eval item is assigned to exactly one failure cell, from the response",
        "logs rather than from aggregate rates. Examples are verbatim.",
        "",
    ]

    ceil = p.get("retrieval_ceiling")
    if ceil:
        a, b = ceil["arms"]
        rf = ceil["retrieval_failures"]
        gf = ceil["generation_failures"]
        lines += [
            "## Retrieval failure vs generation failure",
            "",
            "Both retrieval arms query the same index, so their retrieval failures are",
            "the same items. That makes the decomposition clean:",
            "",
            "| | " + f"{a} | {b} |",
            "|---|---|---|",
            f"| Retrieval failure (source never retrieved) | {rf[a]} | {rf[b]} |",
            f"| Generation failure (source retrieved, still wrong) | {gf[a]} | {gf[b]} |",
            "",
        ]
        if ceil["identical"]:
            lines += [
                f"Retrieval failures are **identical ({rf[a]} items)**, as they must be.",
                "So the entire accuracy difference between the two retrieval arms comes",
                f"from generation: {gf[a]} → {gf[b]} errors. That is the adapter's real",
                "contribution, measured with the retriever held fixed.",
                "",
                "It also fixes a ceiling. Neither arm can exceed the retriever's hit rate,",
                f"so ~{rf[a]} items are unreachable without a better index — no amount of",
                "fine-tuning addresses them.",
                "",
            ]

    for arm, d in p["arms"].items():
        lines += [f"## {arm}", "", f"n = {d['n']}", "", "| Outcome | n | rate |", "|---|---|---|"]
        for k, v in d["counts"].items():
            lines.append(f"| {k.replace('_', ' ')} | {v} | {d['rates'][k]:.1%} |")
        lines.append("")
        for cell in ("generation_failure", "right_act_wrong_section", "retrieval_failure"):
            if cell in d["examples"]:
                lines += _example_block(f"Example — {cell.replace('_', ' ')}", d["examples"][cell])
    return "\n".join(lines)


def main() -> None:
    p = build()
    for arm, d in p["arms"].items():
        print(f"{arm:22s} " + "  ".join(f"{k}={v}" for k, v in d["counts"].items()))
    print(f"\nWrote {REPORTS / 'failures.md'}")


if __name__ == "__main__":
    main()
