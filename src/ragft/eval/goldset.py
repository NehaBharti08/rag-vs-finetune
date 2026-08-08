"""Build the evaluation gold set.

Three properties this has to get right, each defending a specific threat.

**A different generator from the training data (threat 10).** Training pairs
were written by `gemma4:e4b`. If the eval set came from the same model with the
same prompts, it would reward matching that model's idiosyncrasies rather than
being correct. The gold set uses `llama3:8b` -- a third family, neither the
student (Qwen) nor the training generator -- with a different prompt.

**Parametric answerability is a stratification variable, not a filter
(threat 1).** VidyaRAG excludes questions answerable without retrieval, which is
right for evaluating RAG and fatal for a 2x2: it would guarantee the
no-retrieval arms fail and fine-tuning could never win. Here every item is
labelled and results are reported per stratum. The label is measured against the
base model in a later step, not guessed.

**A train-seen slice.** Most items come from eval-unseen sections, but some come
from sections the fine-tuned model trained on. That separates memorisation from
generalisation, and turns a usually-hidden methodological choice into a reported
result.

The unanswerable items are NOT generated here. They are hand-written, because an
LLM asked for unanswerable questions reliably produces obviously out-of-domain
ones, which makes abstention look easy and the metric meaningless -- VidyaRAG's
own evaluation doc makes this point and it is right.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from typing import Any

from ragft.corpus.parse import Section, load_sections
from ragft.corpus.split import Split, load_splits
from ragft.dataset.generate import load_prompt, parse_items
from ragft.dataset.passages import to_passages
from ragft.dataset.schema import chunk_sha256
from ragft.llmclient.ollama import OllamaClient
from ragft.settings import REPO_ROOT, Settings

EVAL_DIR = REPO_ROOT / "data" / "eval"
GOLD_SEED = 20260808

# A third family: not Qwen (the student), not gemma (the training generator).
GOLD_GENERATOR = "llama3:8b"

TARGET_EVAL_UNSEEN = 240
TARGET_TRAIN_SEEN = 60
PER_SECTION = 3


@dataclass
class GoldItem:
    gold_id: str
    question: str
    reference: str
    stratum: str
    source_section_ids: list[str]
    source_chunk_sha256: list[str]
    book_slug: str
    citation: str
    generator_model: str
    unanswerable: bool = False
    # Measured against the base model in a later step, never guessed.
    parametric_answerable: bool | None = None
    human_verified: bool = False


def _gold_id(question: str) -> str:
    return chunk_sha256(question.strip().lower())[:16]


def generate_from(
    sections: list[Section], stratum: str, client: OllamaClient, n_target: int
) -> list[GoldItem]:
    rng = random.Random(GOLD_SEED)
    pool = list(sections)
    rng.shuffle(pool)

    template = load_prompt("gold_eval")
    items: list[GoldItem] = []
    for section in pool:
        if len(items) >= n_target:
            break
        passages = to_passages(section)
        if not passages:
            continue
        passage = passages[0]
        prompt = template.format(
            n=PER_SECTION,
            book=section.book_title,
            label=section.label,
            title=section.title,
            passage=passage,
        )
        raw = client.complete(prompt, temperature=0.7)
        for entry in parse_items(raw):
            question = str(entry.get("question", "")).strip()
            reference = str(entry.get("reference", "")).strip()
            if not question or not reference or not question.endswith("?"):
                continue
            # Same leakage rule as the training set: an eval question that
            # mentions the passage is unanswerable without one.
            if re.search(r"(?i)\b(?:the|this)\s+(?:passage|text|excerpt)\b", question):
                continue
            items.append(
                GoldItem(
                    gold_id=_gold_id(question),
                    question=question,
                    reference=reference,
                    stratum=stratum,
                    source_section_ids=[section.section_id],
                    source_chunk_sha256=[chunk_sha256(passage)],
                    book_slug=section.book_slug,
                    citation=section.citation,
                    generator_model=client.model,
                )
            )
        print(f"  [{len(items)}/{n_target}] {section.section_id}", flush=True)
    return items[:n_target]


def build() -> dict[str, Any]:
    settings = Settings(_env_file=None)
    client = OllamaClient(GOLD_GENERATOR, base_url=settings.ollama_base_url)
    if not client.health():
        raise SystemExit(f"{GOLD_GENERATOR} not available in Ollama")

    assignment = load_splits()
    sections = load_sections()
    unseen = [s for s in sections if assignment.get(s.section_id) == Split.EVAL_UNSEEN.value]
    seen = [s for s in sections if assignment.get(s.section_id) == Split.TRAIN.value]

    print(f"generating with {GOLD_GENERATOR} (training data used a different model)")
    print(f"eval_unseen sections: {len(unseen)} | train sections: {len(seen)}")

    items = generate_from(unseen, "eval_unseen", client, TARGET_EVAL_UNSEEN)
    items += generate_from(seen, "train_seen", client, TARGET_TRAIN_SEEN)

    # Deduplicate on the question itself; the generator repeats across sections.
    seen_ids: set[str] = set()
    unique: list[GoldItem] = []
    for item in items:
        if item.gold_id not in seen_ids:
            seen_ids.add(item.gold_id)
            unique.append(item)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "gold_candidates.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for item in unique:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    summary = {
        "generator": GOLD_GENERATOR,
        "generated": len(items),
        "unique": len(unique),
        "by_stratum": {
            s: sum(i.stratum == s for i in unique) for s in ("eval_unseen", "train_seen")
        },
        "usage": client.usage.summary(),
        "next_steps": [
            "measure parametric_answerable against the base model",
            "hand-write ~60 unanswerable items (an LLM makes these too easy)",
            "human-verify every item via ragft.eval.label",
            "freeze via ragft.eval.frozen --freeze",
        ],
    }
    (EVAL_DIR / "gold_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return summary


def load_candidates() -> list[GoldItem]:
    path = EVAL_DIR / "gold_candidates.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [GoldItem(**json.loads(line)) for line in fh if line.strip()]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()


if __name__ == "__main__":
    main()
