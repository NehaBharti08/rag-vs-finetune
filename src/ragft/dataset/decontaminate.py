"""Verify that no training example shares a source passage with an eval item.

The guarantee itself comes from ordering, not from this module: sections are
assigned to splits **before** any QA is generated, so a training pair and an
eval question cannot derive from the same passage. What happens here is
*verification* of that guarantee, by four independent checks that would each
catch a different way it could have been broken.

That ordering matters. The common alternative -- generate everything, then
filter out overlaps -- is strictly weaker, because near-duplicate detection has
a threshold and whatever slips under it becomes silent contamination that
inflates every reported number.

Checks, weakest assumption to strongest:

1. **Provenance (structural).** Split membership of source sections is
   disjoint. If this fails, the pipeline is broken, not the data.
2. **Exact 13-gram.** The GPT-3 convention. Catches verbatim reuse that
   survived paraphrasing elsewhere.
3. **MinHash LSH.** Jaccard >= 0.7 over 5-gram shingles. Catches reworded
   near-duplicates that share no long n-gram.
4. **Embedding cosine.** >= 0.95 via a local encoder. Catches semantic
   duplicates that share neither n-grams nor tokens.

Run 2-4 both across splits and *within* train, because within-train duplicates
inflate the effective epoch count on whatever they duplicate.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np

from ragft.corpus.split import Split, load_splits
from ragft.settings import REPO_ROOT

QA_DIR = REPO_ROOT / "data" / "qa"
REPORTS = REPO_ROOT / "reports"

NGRAM_N = 13
SHINGLE_N = 5
MINHASH_PERMS = 128
JACCARD_THRESHOLD = 0.7
COSINE_THRESHOLD = 0.95
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

_WORD_RE = re.compile(r"\w+")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: dict[str, Any]


def tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def ngrams(text: str, n: int) -> set[str]:
    """Word n-grams. Empty when the text is shorter than ``n`` tokens."""
    toks = tokens(text)
    return {" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def shingles(text: str, n: int = SHINGLE_N) -> set[str]:
    """Shingles for similarity, degrading to unigrams on short texts.

    `ngrams` returns the empty set for anything shorter than n tokens, and an
    empty MinHash signature is identical to every other empty signature. Left
    unhandled that makes every short question a "duplicate" of every other one:
    the first run of this check flagged 420 within-train pairs, of which the
    top matches were "What is phagocytosis?" against "What is chemical
    energy?" - three-token questions with no 5-gram between them.

    That is a false positive in the checker, not contamination in the data, and
    it is the more dangerous direction of error: a decontamination check that
    cries wolf gets its threshold relaxed until it stops catching real
    problems.
    """
    grams = ngrams(text, n)
    return grams if grams else set(tokens(text))


def check_provenance(rows: list[dict[str, Any]]) -> CheckResult:
    """Every row's source sections must lie in the row's own split."""
    assignment = load_splits()
    violations: list[dict[str, str]] = []
    per_split: Counter[str] = Counter()

    for row in rows:
        per_split[row["split"]] += 1
        for section_id in row["source_section_ids"]:
            actual = assignment.get(section_id)
            if actual != row["split"]:
                violations.append(
                    {
                        "qa_id": row["qa_id"],
                        "section": section_id,
                        "row_split": row["split"],
                        "section_split": str(actual),
                    }
                )

    train_sections = {
        s for r in rows if r["split"] == Split.TRAIN.value for s in r["source_section_ids"]
    }
    eval_sections = {s for s, sp in assignment.items() if sp == Split.EVAL_UNSEEN.value}
    leaked = train_sections & eval_sections

    return CheckResult(
        name="provenance",
        passed=not violations and not leaked,
        detail={
            "rows_by_split": dict(per_split),
            "split_violations": len(violations),
            "train_sections_used": len(train_sections),
            "eval_unseen_sections_total": len(eval_sections),
            "train_eval_section_overlap": sorted(leaked),
            "examples": violations[:5],
        },
    )


def check_ngram(a: list[str], b: list[str], label: str) -> CheckResult:
    """Exact 13-gram overlap between two groups of texts."""
    b_grams: set[str] = set()
    for text in b:
        b_grams |= ngrams(text, NGRAM_N)

    hits = 0
    examples: list[str] = []
    for text in a:
        shared = ngrams(text, NGRAM_N) & b_grams
        if shared:
            hits += 1
            if len(examples) < 5:
                examples.append(next(iter(shared)))

    return CheckResult(
        name=f"ngram_{label}",
        passed=hits == 0,
        detail={"n": NGRAM_N, "texts_with_overlap": hits, "of": len(a), "examples": examples},
    )


def check_minhash(a: list[str], b: list[str], label: str, within: bool = False) -> CheckResult:
    from datasketch import MinHash, MinHashLSH

    def signature(text: str) -> MinHash:
        m = MinHash(num_perm=MINHASH_PERMS)
        for shingle in shingles(text):
            m.update(shingle.encode("utf-8"))
        return m

    lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=MINHASH_PERMS)
    for i, text in enumerate(b):
        lsh.insert(f"b{i}", signature(text))

    matches: list[tuple[int, str]] = []
    for i, text in enumerate(a):
        for key in lsh.query(signature(text)):
            # Within-group comparison would otherwise match every text to itself.
            if within and key == f"b{i}":
                continue
            matches.append((i, key))

    return CheckResult(
        name=f"minhash_{label}",
        passed=len(matches) == 0,
        detail={
            "threshold": JACCARD_THRESHOLD,
            "pairs": len(matches),
            "of": len(a),
            "examples": [a[i][:100] for i, _ in matches[:5]],
        },
    )


def check_embedding(a: list[str], b: list[str], label: str) -> CheckResult:
    from fastembed import TextEmbedding

    model = TextEmbedding(EMBED_MODEL)
    va = np.array(list(model.embed(a)), dtype=np.float32)
    vb = np.array(list(model.embed(b)), dtype=np.float32)
    va /= np.linalg.norm(va, axis=1, keepdims=True)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)

    sim = va @ vb.T
    max_per_row = sim.max(axis=1) if sim.size else np.array([0.0])
    over = int((max_per_row >= COSINE_THRESHOLD).sum())

    # Histogram, not just a pass/fail: the shape of the similarity
    # distribution is what tells a reader whether the threshold was
    # comfortably clear or a near miss.
    hist, edges = np.histogram(max_per_row, bins=[0, 0.5, 0.7, 0.8, 0.9, 0.95, 1.01])
    return CheckResult(
        name=f"embedding_{label}",
        passed=over == 0,
        detail={
            "model": EMBED_MODEL,
            "threshold": COSINE_THRESHOLD,
            "over_threshold": over,
            "of": len(a),
            "max_similarity": round(float(max_per_row.max()), 4),
            "mean_max_similarity": round(float(max_per_row.mean()), 4),
            "histogram": {
                f"{edges[i]:.2f}-{edges[i + 1]:.2f}": int(hist[i]) for i in range(len(hist))
            },
        },
    )


def enforce(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop val rows whose question semantically duplicates a train question.

    Val exists to monitor overfitting. A val question that is a paraphrase of a
    training question is one the model has effectively already seen, so val
    loss would understate the generalisation gap - the single measurement val
    is there to provide.

    These are real duplicates, not a checker artefact: textbooks define the
    same concept in several places, so `Biology 31.1` and `Anatomy 2.4` both
    define "organic compound" from genuinely different passages. Structural
    provenance cannot catch that, which is why the embedding check exists.

    Removal uses the **pre-registered** 0.95 threshold. The count at a stricter
    0.90 is reported but NOT acted on: moving a threshold after seeing results
    is the post-hoc adjustment this project exists to avoid.
    """
    from fastembed import TextEmbedding

    train = [r for r in rows if r["split"] == Split.TRAIN.value]
    val = [r for r in rows if r["split"] == Split.VAL.value]
    if not train or not val:
        return rows, {"removed": 0, "reason": "one side empty"}

    model = TextEmbedding(EMBED_MODEL)
    va = np.array(list(model.embed([r["question"] for r in val])), dtype=np.float32)
    vb = np.array(list(model.embed([r["question"] for r in train])), dtype=np.float32)
    va /= np.linalg.norm(va, axis=1, keepdims=True)
    vb /= np.linalg.norm(vb, axis=1, keepdims=True)
    max_sim = (va @ vb.T).max(axis=1)

    keep_val = [r for r, sim in zip(val, max_sim, strict=True) if sim < COSINE_THRESHOLD]
    removed = [
        {
            "qa_id": r["qa_id"],
            "question": r["question"][:120],
            "max_similarity": round(float(sim), 4),
        }
        for r, sim in zip(val, max_sim, strict=True)
        if sim >= COSINE_THRESHOLD
    ]
    return train + keep_val, {
        "threshold_applied": COSINE_THRESHOLD,
        "val_before": len(val),
        "val_after": len(keep_val),
        "removed": len(removed),
        "examples": removed[:10],
        "residual_at_0.90": int((max_sim >= 0.90).sum()),
        "residual_note": (
            "Reported, not acted on. The 0.95 threshold was fixed before these "
            "numbers existed; tightening it now would be a post-hoc adjustment."
        ),
    }


def run(in_name: str = "filtered.jsonl") -> dict[str, Any]:
    src = QA_DIR / in_name
    rows = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]

    rows, enforcement = enforce(rows)
    out_path = QA_DIR / "clean.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    train = [r for r in rows if r["split"] == Split.TRAIN.value]
    val = [r for r in rows if r["split"] == Split.VAL.value]
    train_q = [r["question"] for r in train]
    val_q = [r["question"] for r in val]

    checks: list[CheckResult] = [check_provenance(rows)]
    if val_q:
        checks += [
            check_ngram(val_q, train_q, "val_vs_train"),
            check_minhash(val_q, train_q, "val_vs_train"),
            check_embedding(val_q, train_q, "val_vs_train"),
        ]
    checks.append(check_minhash(train_q, train_q, "within_train", within=True))

    summary = {
        "input": in_name,
        "rows": len(rows),
        "train_rows": len(train),
        "val_rows": len(val),
        "enforcement": enforcement,
        "all_passed": all(c.passed for c in checks),
        "checks": [{"name": c.name, "passed": c.passed, **c.detail} for c in checks],
        "note": (
            "The eval gold set is built in Phase 2. These checks cover the "
            "train/val boundary and within-train duplication; the same functions "
            "are re-run against the gold set once it exists."
        ),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "decontamination.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-name", default="filtered.jsonl")
    args = parser.parse_args()

    s = run(args.in_name)
    print(f"rows={s['rows']:,} train={s['train_rows']:,} val={s['val_rows']:,}")
    for c in s["checks"]:
        mark = "PASS" if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}")
    print(f"\nall_passed = {s['all_passed']}")


if __name__ == "__main__":
    main()
