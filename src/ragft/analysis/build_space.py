"""Build the deployable artifacts for the Hugging Face demo Space. Phase 7.

Two Spaces come out of one pipeline, because Hugging Face prices them
differently:

* **Static** (free) -- `deploy/space/page.template.html` with the data inlined.
  The citation checker runs the benchmark's own regex client-side; the answers
  and retrieval are real measurements, precomputed.
* **Gradio** (needs HF PRO) -- `deploy/space/app.py`, which adds live retrieval
  for arbitrary questions. Free `cpu-basic` returns 402 for Gradio Spaces.

**Verdicts are computed by the repo's own validator, never re-implemented.**
The first version of this builder hand-ported the citation regex into the demo
and it drifted: A2 came out at 254 correct / 2 out-of-corpus against the
published 251 / 5, because the hand-port missed typographic apostrophes, the
optional year group and the `Code of ...` prefix form. The pattern itself is now
shipped to the browser verbatim, so the demo cannot disagree with `reports/`.

Usage::

    uv run python -m ragft.analysis.build_space
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ragft.corpus.acts import ACTS
from ragft.corpus.toc import CITATION_RE, CitationVerdict, registry
from ragft.settings import REPO_ROOT

RESPONSES = REPO_ROOT / "data" / "eval" / "responses"
DEPLOY = REPO_ROOT / "deploy"
OUT = DEPLOY / "build"

ARMS = {
    "A1_base_zeroshot": "A1 base, no retrieval",
    "A2_base_rag": "A2 base + RAG",
    "A3_ft_zeroshot": "A3 fine-tuned, no retrieval",
    "A4_ft_rag": "A4 fine-tuned + RAG",
}
TOP_K_RETRIEVE, TOP_K_CONTEXT = 20, 5
EMBED_MODEL = "BAAI/bge-large-en-v1.5"


def classify(text: str, gold: list[str]) -> tuple[str, str | None, str]:
    """The repo's five-state ladder, using the repo's own registry."""
    reg = registry()
    c = reg.validate(text)
    if c.verdict is CitationVerdict.UNPARSEABLE:
        return "none", None, "No parseable citation."
    if c.act_slug is None:
        return (
            "out_of_corpus",
            c.raw,
            (f"`{c.raw}` names an act outside the corpus — usually repealed pre-2024 law."),
        )
    act = next(a.exact_name for a in ACTS if a.slug == c.act_slug)
    gold_acts = {g.split(":")[0] for g in gold}
    if c.section_id and c.section_id in gold:
        return (
            "correct",
            c.raw,
            f"Correct — {act}, §{c.cited_section} is the provision this answer lives in.",
        )
    if c.section_id is None:
        key = "wrong_act" if c.act_slug not in gold_acts else "near"
        return key, c.raw, f"§{c.cited_section} does not exist in {act}."
    title = reg.by_id[c.section_id].title
    if c.act_slug not in gold_acts:
        return "wrong_act", c.raw, f"Wrong act. {act}, §{c.cited_section} is “{title}”."
    return (
        "near",
        c.raw,
        (
            f"Right act, wrong section. Cited {act}, §{c.cited_section} — “{title}” — "
            "not the provision the answer is about."
        ),
    )


def _retrieval(questions: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Real retrieval, same embedder and top_k as the benchmark.

    Embeddings are computed here rather than in the browser on purpose: a
    smaller in-browser embedder would change what the demo retrieves relative to
    what the benchmark measured, which is the substitution this project exists
    to warn about.
    """
    from fastembed import TextEmbedding

    chunks = [
        json.loads(line)
        for line in (REPO_ROOT / "data" / "corpus" / "chunks.jsonl").open(encoding="utf-8")
        if line.strip()
    ]
    model = TextEmbedding(EMBED_MODEL)
    mat = np.array(list(model.embed([c["text"] for c in chunks])), dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    qv = np.array(list(model.embed(questions)), dtype=np.float32)
    qv /= np.linalg.norm(qv, axis=1, keepdims=True)

    out: dict[str, list[dict[str, Any]]] = {}
    for question, vec in zip(questions, qv, strict=True):
        scores = mat @ vec
        top = np.argsort(-scores)[:TOP_K_RETRIEVE][:TOP_K_CONTEXT]
        out[question] = [
            {
                "cit": chunks[int(i)]["citation"],
                "sid": chunks[int(i)]["section_id"],
                "title": chunks[int(i)]["section_title"],
                "sim": round(float(scores[int(i)]), 3),
                "text": chunks[int(i)]["text"][:420],
            }
            for i in top
        ]
    return out


def build(with_retrieval: bool = True) -> Path:
    reg = registry()
    by_id: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        for line in (RESPONSES / f"{arm}.jsonl").open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            entry = by_id.setdefault(
                r["gold_id"],
                {
                    "q": r["question"],
                    "ref": r["reference"],
                    "gold": r["source_section_ids"],
                    "arms": {},
                },
            )
            verdict, raw, message = classify(r["response"], r["source_section_ids"])
            entry["arms"][arm] = {"t": r["response"], "v": verdict, "c": raw, "m": message}

    items = sorted(by_id.values(), key=lambda i: i["q"])
    retrieval = _retrieval([i["q"] for i in items]) if with_retrieval else {}
    for item in items:
        item["ret"] = retrieval.get(item["q"], [])

    payload = {
        "arms": ARMS,
        "items": items,
        "acts": {a.slug: a.exact_name for a in ACTS},
        "shortnames": {a.slug: a.short_name for a in ACTS},
        "sections": {sid: s.title for sid, s in reg.by_id.items()},
        # Shipped verbatim. The browser translates (?P<x>) to (?<x>); everything
        # else is identical, so the client-side checker cannot drift.
        "citation_re": CITATION_RE.pattern,
    }

    template = (DEPLOY / "space" / "page.template.html").read_text(encoding="utf-8")
    fragment = template.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    head, body = fragment.split('\n<div class="wrap">', 1)
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<style>body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n"
        + head
        + '\n</head>\n<body>\n<div class="wrap">'
        + body
        + "\n</body>\n</html>\n"
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "index.html"
    out.write_text(html, encoding="utf-8")

    from collections import Counter

    print(
        f"{out.relative_to(REPO_ROOT)}  {out.stat().st_size // 1024} KB  ({len(items)} questions)"
    )
    for arm in ARMS:
        counts = Counter(i["arms"][arm]["v"] for i in items)
        print(f"  {arm:20s} " + "  ".join(f"{k}={v}" for k, v in counts.most_common()))
    # A JS regex must accept the pattern; a silent failure here would leave the
    # client-side checker reporting "no citation" for everything.
    assert "(?P<" in CITATION_RE.pattern
    assert re.sub(r"\(\?P<(\w+)>", r"(?<\1>", CITATION_RE.pattern).count("(?<") == 3
    return out


def main() -> None:
    build()


if __name__ == "__main__":
    main()
