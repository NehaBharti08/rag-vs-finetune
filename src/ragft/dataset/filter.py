"""Quality gates, with every rejection attributed to a named criterion.

A rejection rate is only meaningful if you can say *what* was rejected and
*why*, so each gate reports its own count and the dataset card publishes the
table. A single aggregate "we filtered 30%" tells a reader nothing about
whether the remaining 70% is any good.

One gate does normalisation rather than rejection, and that decision is worth
stating plainly. Local models very often open an explanation with "The passage
states that ...", despite explicit instructions not to. Measured on a sample it
affected ~93% of generations. Rejecting them would discard almost the whole
dataset over a stylistic prefix that carries no content; the reference is a
sentence-initial clause that strips losslessly. So it is stripped, counted, and
reported -- and anything where the reference is *structurally* embedded, and so
cannot be removed without changing meaning, is still rejected.

Why it matters at all: the fine-tuned model never sees a passage at inference.
Training it on text that refers to one teaches it to cite a source it cannot
read, which is the exact failure the citation metric is built to detect.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ragft.corpus.toc import registry
from ragft.dataset.schema import REFUSAL_TEXT, QAType
from ragft.settings import REPO_ROOT

QA_DIR = REPO_ROOT / "data" / "qa"

MIN_QUESTION_CHARS, MAX_QUESTION_CHARS = 20, 400
MIN_ANSWER_CHARS, MAX_ANSWER_CHARS = 10, 900
MAX_WHY_CHARS = 1200

# Sentence-initial meta-reference: "The passage notes that ...". Stripped.
_META_PREFIX = re.compile(
    r"(?:^|(?<=[.!?]\s))\s*(?:The|This)\s+"
    r"(?:passage|text|section|excerpt|source|article|document)\s+"
    r"(?:also\s+)?"
    r"(?:states?|notes?|explains?|indicates?|describes?|defines?|emphasi[sz]es?|"
    r"mentions?|says?|shows?|specifies?|establishes?|makes?|lists?|uses?|"
    r"suggests?|clarifies?|highlights?|points? out|discusses?)\s+"
    r"(?:that\s+)?",
    re.IGNORECASE,
)
# Any surviving reference to the source document. Rejected.
_META_ANY = re.compile(
    r"(?i)\b(?:the|this)\s+(?:passage|excerpt)\b"
    r"|according to the (?:passage|text|excerpt)"
    r"|\bas (?:shown|described|stated|mentioned) (?:above|below|here)\b"
    r"|\bin this (?:text|section|passage)\b"
)


@dataclass
class Rejection:
    qa_id: str
    criterion: str
    detail: str


def strip_meta(text: str) -> tuple[str, bool]:
    """Remove sentence-initial source references; recapitalise what follows."""
    cleaned = _META_PREFIX.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    changed = cleaned != text.strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
    return cleaned, changed


def check(row: dict[str, Any], seen_questions: set[str]) -> str | None:
    """Return the name of the first failing criterion, or None if the row passes."""
    q, a, why = row["question"], row["answer"], row["why"]
    qa_type = row["qa_type"]

    if not q or not a:
        return "empty_field"
    if not (MIN_QUESTION_CHARS <= len(q) <= MAX_QUESTION_CHARS):
        return "question_length"
    if not (MIN_ANSWER_CHARS <= len(a) <= MAX_ANSWER_CHARS):
        return "answer_length"
    if len(why) > MAX_WHY_CHARS:
        return "why_length"
    if not q.rstrip().endswith("?"):
        return "not_a_question"
    # A truncated generation ends mid-word or mid-clause.
    if a.rstrip().endswith((",", ";", "and", "the", "of", "to")):
        return "truncated_answer"

    # Unanswerable items must actually refuse, or they teach the opposite of
    # what they exist to teach.
    if qa_type == QAType.UNANSWERABLE.value:
        if REFUSAL_TEXT.lower()[:40] not in a.lower():
            return "refusal_text_missing"
    elif REFUSAL_TEXT.lower()[:40] in a.lower():
        # An answerable item that refuses is a generation failure.
        return "unexpected_refusal"

    if _META_ANY.search(f"{q} {a} {why}"):
        return "passage_reference"
    if _META_ANY.search(q):
        return "passage_reference_in_question"

    # Citations are constructed from verified metadata, so a failure here means
    # the corpus registry and the generated data have gone out of sync.
    if not registry().validate(row["citation"]).is_valid:
        return "invalid_citation"

    key = re.sub(r"\W+", " ", q.lower()).strip()
    if key in seen_questions:
        return "duplicate_question"
    seen_questions.add(key)
    return None


def run(in_name: str = "raw.jsonl", out_name: str = "filtered.jsonl") -> dict[str, Any]:
    src = QA_DIR / in_name
    if not src.exists():
        raise FileNotFoundError(f"{src} missing - run ragft.dataset.generate first")

    rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
    counts: Counter[str] = Counter()
    normalised = 0
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    rejections: list[Rejection] = []

    for row in rows:
        for field in ("answer", "why"):
            cleaned, changed = strip_meta(row[field])
            row[field] = cleaned
            normalised += int(changed)

        reason = check(row, seen)
        if reason:
            counts[reason] += 1
            rejections.append(Rejection(row["qa_id"], reason, row["question"][:90]))
            continue

        # Rebuild the formatted answer from the normalised fields.
        row["formatted_answer"] = (
            f"**Answer.** {row['answer']}\n\n"
            f"**Why.** {row['why']}\n\n"
            f"**Source.** {row['citation']}"
        )
        kept.append(row)
        counts["kept"] += 1

    out = QA_DIR / out_name
    with out.open("w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    total = len(rows) or 1
    summary = {
        "input_rows": len(rows),
        "kept": len(kept),
        "rejected": len(rows) - len(kept),
        "rejection_rate": round((len(rows) - len(kept)) / total, 4),
        "meta_prefix_normalised_fields": normalised,
        "by_criterion": dict(counts.most_common()),
        "by_type_kept": dict(Counter(r["qa_type"] for r in kept)),
        "by_split_kept": dict(Counter(r["split"] for r in kept)),
    }
    (QA_DIR / "filter_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (QA_DIR / "rejections.jsonl").write_text(
        "\n".join(json.dumps(r.__dict__) for r in rejections) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-name", default="raw.jsonl")
    parser.add_argument("--out-name", default="filtered.jsonl")
    args = parser.parse_args()

    s = run(args.in_name, args.out_name)
    print(
        f"input {s['input_rows']:,} -> kept {s['kept']:,} "
        f"(rejected {s['rejected']:,} = {s['rejection_rate']:.1%})"
    )
    print(f"meta-prefix normalised: {s['meta_prefix_normalised_fields']:,} fields")
    print("\nby criterion:")
    for k, v in s["by_criterion"].items():
        print(f"  {k:32s} {v:6,}")


if __name__ == "__main__":
    main()
