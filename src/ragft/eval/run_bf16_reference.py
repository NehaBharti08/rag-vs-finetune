"""bf16 base arm: the sanity reference for threat 3. NOT a cell of the 2x2.

Every arm of the grid runs 4-bit NF4, including the base arms, so that
quantization cannot be confounded with adaptation. That is the defense, and it
holds by construction.

It leaves one question open: **how much does NF4 itself cost the base model?**
If quantization were badly damaging Qwen2.5, A1's weakness would be partly an
artifact of the quantization rather than a fact about its knowledge — and every
statement of the form "the base model does not know this corpus" would be
overstated by that amount.

This arm measures that, and only that. It re-runs A1's exact prompt and gold set
with the model loaded in bf16 instead of NF4. The delta bounds the quantization
penalty.

It is deliberately **not** reported as part of the grid. Including a bf16 cell
would reintroduce precisely the confound threat 3 exists to prevent.

Needs ~15 GB of VRAM (unquantized 7B), against ~5.5 GB for the NF4 arms, so it
requires a card with real headroom rather than a shared slot.

Usage::

    uv run python -m ragft.eval.run_bf16_reference
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from ragft.eval.arms import ArmRunner, arm_specs, build_prompt
from ragft.eval.metrics.citation import aggregate as citation_aggregate
from ragft.eval.metrics.format_check import aggregate as format_aggregate
from ragft.eval.runner import load_gold
from ragft.settings import REPO_ROOT, QuantConfig

REPORTS = REPO_ROOT / "reports"
RESPONSES = REPO_ROOT / "data" / "eval" / "responses"


def run(limit: int | None = None) -> dict[str, Any]:
    free, total = torch.cuda.mem_get_info()
    free_gib = free / 1024**3
    print(f"free VRAM: {free_gib:.1f} GiB of {total / 1024**3:.1f} GiB")
    if free_gib < 16.0:
        raise SystemExit(
            f"Need ~16 GiB free for an unquantized 7B; only {free_gib:.1f} GiB available.\n"
            "This arm requires a card with real headroom. Wait for a free slot."
        )

    items = load_gold()
    if limit:
        items = items[:limit]

    # The ONLY difference from A1. Same prompt, same gold set, same decoding.
    runner = ArmRunner(quant=QuantConfig(load_in_4bit=False))
    arm = next(a for a in arm_specs(None) if a.name == "A1_base_zeroshot")

    rows: list[dict[str, Any]] = []
    for n, item in enumerate(items, 1):
        out = runner.generate(build_prompt(arm, item.question, ""))
        rows.append(
            {
                "gold_id": item.gold_id,
                "question": item.question,
                "response": out["response"],
                "source_section_ids": item.source_section_ids,
                "latency_seconds": out["latency_seconds"],
            }
        )
        if n % 50 == 0 or n == len(items):
            print(f"  [{n}/{len(items)}]", flush=True)

    RESPONSES.mkdir(parents=True, exist_ok=True)
    with (RESPONSES / "A1_base_zeroshot__bf16.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    bf16: dict[str, Any] = {
        "citation": citation_aggregate(
            [r["response"] for r in rows], [r["source_section_ids"] for r in rows]
        ),
        "format": format_aggregate([r["response"] for r in rows]),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
    }

    nf4_rows = [
        json.loads(line)
        for line in (RESPONSES / "A1_base_zeroshot.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    nf4: dict[str, Any] = {
        "citation": citation_aggregate(
            [r["response"] for r in nf4_rows], [r["source_section_ids"] for r in nf4_rows]
        ),
        "format": format_aggregate([r["response"] for r in nf4_rows]),
    }

    payload = {
        "n": len(rows),
        "bf16": bf16,
        "nf4_A1": nf4,
        "delta_correct_section": round(
            bf16["citation"]["section_correct_rate"] - nf4["citation"]["section_correct_rate"], 4
        ),
        "delta_correct_act": round(
            bf16["citation"]["act_correct_rate"] - nf4["citation"]["act_correct_rate"], 4
        ),
        "note": (
            "Reference only, never a cell of the 2x2. Including a bf16 cell would "
            "reintroduce the quantization confound that threat 3 exists to prevent."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "bf16_reference.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (REPORTS / "bf16_reference.md").write_text(render(payload), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k.startswith("delta")}, indent=2))
    return payload


def render(p: dict[str, Any]) -> str:
    b, n = p["bf16"]["citation"], p["nf4_A1"]["citation"]
    ds, da = p["delta_correct_section"], p["delta_correct_act"]
    verdict = (
        "Quantization is **not** what makes the base model weak on this corpus."
        if abs(ds) < 0.05 and abs(da) < 0.10
        else "Quantization has a material effect and every base-arm claim should be "
        "read with that in mind."
    )
    return f"""# bf16 reference arm

**This is not a cell of the 2x2.** Every arm of the grid runs 4-bit NF4 so that
quantization cannot be confounded with adaptation. This arm exists to answer the
one question that defense leaves open: *what does NF4 itself cost the base
model?*

Same prompt, same gold set, same decoding as A1 — the only difference is bf16
weights instead of NF4.

| Metric | A1 (NF4) | A1 (bf16) | delta |
|---|---|---|---|
| Cites the CORRECT section | {n["section_correct_rate"]:.1%} | {b["section_correct_rate"]:.1%} | **{ds * 100:+.1f} pp** |
| Names the CORRECT act | {n["act_correct_rate"]:.1%} | {b["act_correct_rate"]:.1%} | **{da * 100:+.1f} pp** |
| Cites a section that exists | {n["section_exists_rate"]:.1%} | {b["section_exists_rate"]:.1%} | {(b["section_exists_rate"] - n["section_exists_rate"]) * 100:+.1f} pp |
| Fabrication rate | {n["fabrication_rate"]:.1%} | {b["fabrication_rate"]:.1%} | {(b["fabrication_rate"] - n["fabrication_rate"]) * 100:+.1f} pp |
| Format valid | {p["nf4_A1"]["format"]["valid_rate"]:.1%} | {p["bf16"]["format"]["valid_rate"]:.1%} | — |

n = {p["n"]}. Peak reserved in bf16: {p["bf16"]["peak_reserved_gib"]} GiB
(against ~5.5 GiB for the NF4 arms).

## What this bounds

{verdict}

If bf16 scored much higher, A1's weakness would be partly an artifact of
quantization and every "the base model does not know this corpus" claim would be
overstated by that margin. The delta above is that margin, measured rather than
assumed.

_Regenerate: `uv run python -m ragft.eval.run_bf16_reference`_
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.limit)


if __name__ == "__main__":
    main()
