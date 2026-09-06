"""Measure what adaptation cost in general capability. Phase 6.

Runs the same MMLU items twice -- once on the base model, once with the adapter
attached -- and reports the delta. Absolute scores are not the point; the
question is what fine-tuning took away.

Two in-domain subjects (professional_law, jurisprudence) and two far
out-of-domain (college_biology, formal_logic). Reporting only in-domain would
hide the trade this probe exists to expose.

Caveat kept in front of the numbers rather than in a footnote: MMLU has no
Indian-law subject, so `professional_law` and `jurisprudence` are US-centric.
This measures whether general legal reasoning survives adaptation, NOT whether
the model learned Indian statutes. The gold set measures the latter.

Usage::

    uv run python -m ragft.eval.run_mmlu --adapter out/.../checkpoint-354
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ragft.eval.metrics.mmlu import evaluate, forgetting_delta, load_items
from ragft.settings import BASE_MODEL, REPO_ROOT, QuantConfig

REPORTS = REPO_ROOT / "reports"


def build_base() -> tuple[Any, Any]:
    """Base model under the SAME quantization every arm uses."""
    quant = QuantConfig()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    bnb = BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant.compute_dtype),
        bnb_4bit_use_double_quant=quant.double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return model, tokenizer


def run(adapter: str, per_subject: int = 100) -> dict[str, Any]:
    items = load_items(per_subject=per_subject)
    print(f"MMLU items: {len(items)}")

    model, tokenizer = build_base()
    print("scoring base model...")
    base_scores = evaluate(model, tokenizer, items)

    from peft import PeftModel

    print(f"attaching adapter {adapter}...")
    adapted = PeftModel.from_pretrained(model, adapter)
    adapted.eval()
    print("scoring adapted model...")
    adapted_scores = evaluate(adapted, tokenizer, items)

    delta = forgetting_delta(base_scores, adapted_scores)
    payload = {
        "adapter": adapter,
        "n_items": len(items),
        "base": base_scores,
        "adapted": adapted_scores,
        "delta": delta,
        "caveat": (
            "MMLU has no Indian-law subject. professional_law and jurisprudence "
            "are US-centric, so this measures whether general legal reasoning "
            "survives adaptation, not whether Indian statutes were learned."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "mmlu.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "base": base_scores["per_subject"],
                "adapted": adapted_scores["per_subject"],
                "delta": delta,
            },
            indent=2,
        )
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="out/seed42_r16_lr0.0002_e3/checkpoint-354")
    parser.add_argument("--per-subject", type=int, default=100)
    args = parser.parse_args()
    run(args.adapter, args.per_subject)


if __name__ == "__main__":
    main()
