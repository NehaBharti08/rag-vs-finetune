"""MMLU probe for catastrophic forgetting -- no generation, no judge.

Fine-tuning on a narrow domain can cost general capability. This measures how
much, and it is deliberately the cheapest metric in the suite: answers are
scored by comparing the log-probability of the tokens " A", " B", " C", " D" at
a single forward pass. No sampling, no generation, no judge, fully
deterministic.

Subject choice is what makes it a *forgetting* probe rather than a trivia quiz.
Two in-domain subjects establish that adaptation did something; two far
out-of-domain subjects show what it cost. Reporting only in-domain scores would
hide the trade the whole probe exists to expose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

# In-domain: adaptation should hold or improve these.
IN_DOMAIN = ("high_school_biology", "college_biology")
# Far out-of-domain: this is where forgetting shows up.
OUT_OF_DOMAIN = ("formal_logic", "professional_law")
SUBJECTS = IN_DOMAIN + OUT_OF_DOMAIN

CHOICES = ("A", "B", "C", "D")


@dataclass(frozen=True)
class MMLUItem:
    subject: str
    question: str
    choices: list[str]
    answer_index: int


def format_item(item: MMLUItem) -> str:
    lines = [
        f"The following is a multiple choice question about {item.subject.replace('_', ' ')}.",
        "",
    ]
    lines.append(item.question)
    for letter, choice in zip(CHOICES, item.choices, strict=False):
        lines.append(f"{letter}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_items(subjects: tuple[str, ...] = SUBJECTS, per_subject: int = 125) -> list[MMLUItem]:
    """Load a fixed slice of MMLU test items per subject."""
    from datasets import load_dataset

    items: list[MMLUItem] = []
    for subject in subjects:
        ds = load_dataset("cais/mmlu", subject, split="test")
        # Deterministic slice, not a random sample: the probe must compare the
        # same questions across every arm and every seed.
        for row in list(ds)[:per_subject]:
            items.append(
                MMLUItem(
                    subject=subject,
                    question=str(row["question"]),
                    choices=[str(c) for c in row["choices"]],
                    answer_index=int(row["answer"]),
                )
            )
    return items


@torch.inference_mode()
def score_item(model: Any, tokenizer: Any, item: MMLUItem) -> int:
    """Return the predicted choice index by comparing option log-probs."""
    prompt = format_item(item)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1, :]

    option_ids = [tokenizer.encode(f" {c}", add_special_tokens=False)[0] for c in CHOICES]
    return int(torch.tensor([logits[i] for i in option_ids]).argmax().item())


def evaluate(model: Any, tokenizer: Any, items: list[MMLUItem]) -> dict[str, Any]:
    by_subject: dict[str, list[bool]] = {}
    for item in items:
        correct = score_item(model, tokenizer, item) == item.answer_index
        by_subject.setdefault(item.subject, []).append(correct)

    per_subject = {s: round(sum(v) / len(v), 4) for s, v in by_subject.items()}
    in_scores = [v for s, v in per_subject.items() if s in IN_DOMAIN]
    out_scores = [v for s, v in per_subject.items() if s in OUT_OF_DOMAIN]

    return {
        "n": len(items),
        "per_subject": per_subject,
        "in_domain_mean": round(sum(in_scores) / len(in_scores), 4) if in_scores else None,
        "out_of_domain_mean": round(sum(out_scores) / len(out_scores), 4) if out_scores else None,
        "overall": round(sum(sum(v) for v in by_subject.values()) / max(1, len(items)), 4),
        "note": (
            "Forgetting is the DELTA against the base model's scores, not these "
            "absolutes. Out-of-domain drop is the cost of adaptation; reporting "
            "only in-domain would hide the trade this probe exists to expose."
        ),
    }


def forgetting_delta(base: dict[str, Any], adapted: dict[str, Any]) -> dict[str, Any]:
    """What adaptation cost, per subject and per domain group."""
    deltas = {
        s: round(adapted["per_subject"].get(s, 0.0) - v, 4) for s, v in base["per_subject"].items()
    }
    ood = [d for s, d in deltas.items() if s in OUT_OF_DOMAIN]
    return {
        "per_subject_delta": deltas,
        "in_domain_delta": round(
            (adapted["in_domain_mean"] or 0) - (base["in_domain_mean"] or 0), 4
        ),
        "out_of_domain_delta": round(
            (adapted["out_of_domain_mean"] or 0) - (base["out_of_domain_mean"] or 0), 4
        ),
        "catastrophic_forgetting": bool(ood and min(ood) < -0.05),
    }
