"""Live demo for the RAG vs QLoRA benchmark on Indian statutes.

WHAT IS LIVE HERE, AND WHAT IS NOT -- stated up front because the distinction is
the whole point of the project.

LIVE, running on this CPU Space right now:
  * Retrieval. Your question is embedded with BAAI/bge-large-en-v1.5 and searched
    against 1,354 chunks of four Indian statutes. Same embedder, same chunking
    (512/64), same top_k (20 -> 5) as the benchmark.
  * Citation validation. Any text you paste is resolved against a registry of
    1,237 real sections. This is the project's headline metric and it needs no
    model at all.

NOT LIVE: the 7B generation. A free Space is CPU with 16 GB of RAM and cannot
serve a 4-bit 7B. A Space that quietly swapped in a smaller model would be
demonstrating something this project never measured, so instead it serves the
REAL measured answers -- all 300 gold questions x 4 arms, 1,200 responses from
the actual benchmark run.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
from citation import ACTS, grade, registry, validate

HERE = Path(__file__).parent
ANSWERS = json.loads((HERE / "answers.json").read_text(encoding="utf-8"))
ITEMS = ANSWERS["items"]
ARMS = ANSWERS["arms"]
CHUNKS = json.loads((HERE / "chunks.json").read_text(encoding="utf-8"))
VECTORS = np.load(HERE / "chunk_vectors.npy")

TOP_K_RETRIEVE, TOP_K_CONTEXT = 20, 5


@lru_cache(maxsize=1)
def embedder():
    from fastembed import TextEmbedding

    return TextEmbedding("BAAI/bge-large-en-v1.5")


def search(question: str) -> str:
    """LIVE retrieval over the real corpus. No model, no API, runs on CPU."""
    question = (question or "").strip()
    if len(question) < 8:
        return "_Type a question about Indian statutory law._"

    q = np.array(next(iter(embedder().embed([question]))), dtype=np.float32)
    q /= np.linalg.norm(q)
    scores = VECTORS @ q  # cosine: both sides unit-normalised
    top = np.argsort(-scores)[:TOP_K_RETRIEVE][:TOP_K_CONTEXT]

    lines = [
        f"**{TOP_K_CONTEXT} passages retrieved** from 1,354 chunks "
        f"(of {TOP_K_RETRIEVE} candidates), embedded live on CPU.\n"
    ]
    for rank, i in enumerate(top, 1):
        c = CHUNKS[int(i)]
        body = c["text"].strip().replace("\n", " ")
        if len(body) > 420:
            body = body[:420] + " …"
        lines.append(
            f"**{rank}. {c['citation']}** — *{c['section_title']}*  \n"
            f"<sub>similarity {scores[int(i)]:.3f} · split `{c['split']}`</sub>\n\n"
            f"> {body}\n"
        )
    lines.append(
        "\n---\n*This is the exact context the RAG arms (A2, A4) were given. "
        "Retrieval hit the right section for 94.0% of gold questions — so when a "
        "RAG arm cites wrongly, the failure is generation, not retrieval.*"
    )
    return "\n".join(lines)


VERDICT_STYLE = {
    "correct": "🟢",
    "near": "🟠",
    "wrong_act": "🔴",
    "out_of_corpus": "🔴",
    "none": "⚪",
    "exists": "🔵",
}


def compare(question: str) -> str:
    """The real measured answers from all four arms, with citations graded."""
    item = next((i for i in ITEMS if i["q"] == question), None)
    if item is None:
        return "_Pick a question from the dropdown._"

    gold = item["gold"]
    gold_txt = ", ".join(f"`{g}`" for g in gold)
    out = [
        f"### {item['q']}",
        f"**Gold answer.** {item['ref']}",
        f"**Correct provision.** {gold_txt}\n",
        "---\n",
    ]
    for key, label in ARMS.items():
        text = item["arms"].get(key, "")
        verdict, explanation = grade(text, gold)
        out.append(f"#### {VERDICT_STYLE.get(verdict,'')} {label}\n")
        out.append(text.strip() + "\n")
        out.append(f"**Citation check.** {explanation}\n")
        out.append("---\n")
    out.append(
        "*These are real responses from the benchmark run, not generated now. "
        "Across 3 seeds the fine-tuned arm names the correct **act** 92.1% ± 1.7 "
        "of the time and the correct **section** 0.9% ± 1.0 — it learned which "
        "statute governs a question and almost nothing about which section.*"
    )
    return "\n".join(out)


def check_citation(text: str) -> str:
    """LIVE validation of anything you paste. No model needed."""
    text = (text or "").strip()
    if not text:
        return "_Paste an answer containing a citation, e.g. “The Bharatiya Nyaya Sanhita, 2023, §103”._"
    c = validate(text)
    if c.verdict == "unparseable":
        return (
            "⚪ **No parseable citation found.**\n\nThe validator looks for an act "
            "name followed by a section marker — `The Bharatiya Nyaya Sanhita, 2023, §103`, "
            "`BNS section 103`, `Indian Contract Act, 1872, s. 11`."
        )
    if c.act_slug is None:
        return (
            f"🔴 **Outside the corpus.** Found `{c.raw}`.\n\nThat act is not one of the four "
            "indexed statutes. On this corpus that usually means **repealed pre-2024 law** — "
            "the IPC, CrPC and Indian Evidence Act were replaced in 2023, and a model trained "
            "before then cites them by reflex."
        )
    act = ACTS[c.act_slug]
    if c.section_id is None:
        return (
            f"🔴 **Fabricated section.** `{c.raw}` names a real act ({act}) but "
            f"**§{c.cited_section} does not exist** in it."
        )
    title = registry()[c.section_id]["title"]
    return (
        f"🔵 **Real section.** {act}, §{c.cited_section} exists — *{title}*.\n\n"
        "Note this only proves the section is **real**, not that it is the **right** one. "
        "Separating those two questions is the whole finding: the fine-tuned model names a "
        "real section 99% of the time and the correct one ~1% of the time."
    )


CSS = """
.gradio-container {max-width: 1080px !important}
footer {display:none !important}
"""

with gr.Blocks(title="RAG vs QLoRA — Indian Statutes", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# ⚖️ RAG vs QLoRA fine-tuning — live demo

A 2x2 benchmark over four Indian statutes: {base, QLoRA} x {no retrieval, retrieval}.

**Fine-tuning taught the model which _statute_ governs a question (47.7% → 92.1%)
and almost nothing about which _section_ (0.3% → 0.9%).** It names a real section
99% of the time and the *correct* one about 1% of the time.

> **What runs live here:** retrieval and citation validation, on CPU, using the
> benchmark's own embedder and registry. **What doesn't:** the 7B generation — a
> free Space cannot serve a 4-bit 7B, and quietly swapping in a smaller model
> would demo something this project never measured. Tab 2 serves the *real*
> measured answers instead.
"""
    )

    with gr.Tab("🔎 Live retrieval"):
        gr.Markdown(
            "Ask anything about Indian statutory law. Your question is embedded "
            "**right now** and searched against 1,354 chunks — the same pipeline the "
            "RAG arms used."
        )
        q1 = gr.Textbox(
            label="Your question",
            placeholder="When may a police officer arrest a person without a warrant?",
            lines=2,
        )
        b1 = gr.Button("Retrieve passages", variant="primary")
        o1 = gr.Markdown()
        gr.Examples(
            [
                "When may a police officer arrest a person without a warrant?",
                "What punishment does the law prescribe for murder?",
                "When is an agreement without consideration void?",
                "How is the admissibility of electronic records established?",
            ],
            inputs=q1,
        )
        b1.click(search, q1, o1)
        q1.submit(search, q1, o1)

    with gr.Tab("⚖️ All four arms"):
        gr.Markdown(
            "The **real measured answers** for each of the 300 gold questions, with "
            "every citation graded against the corpus. Look for 🟠 — *right act, "
            "wrong section* — which is 271 of 300 for the fine-tuned arm."
        )
        q2 = gr.Dropdown(
            choices=[i["q"] for i in ITEMS],
            label="Gold question (300 available)",
            value=ITEMS[0]["q"],
        )
        o2 = gr.Markdown()
        q2.change(compare, q2, o2)
        demo.load(compare, q2, o2)

    with gr.Tab("✅ Citation checker"):
        gr.Markdown(
            "Paste any answer. The validator resolves its citation against 1,237 real "
            "sections — **no model involved.** This is the judge-free metric the whole "
            "benchmark rests on."
        )
        q3 = gr.Textbox(
            label="Text containing a citation",
            placeholder="**Source.** The Bharatiya Nyaya Sanhita, 2023, §103",
            lines=4,
        )
        b3 = gr.Button("Validate", variant="primary")
        o3 = gr.Markdown()
        gr.Examples(
            [
                "**Source.** The Bharatiya Nyaya Sanhita, 2023, §103",
                "**Source.** The Bharatiya Nyaya Sanhita, 2023, §9999",
                "**Source.** The Indian Penal Code, 1860, §302",
                "**Source.** The Bharatiya Sakshya Adhiniyam, 2023, §145",
            ],
            inputs=q3,
        )
        b3.click(check_citation, q3, o3)
        q3.submit(check_citation, q3, o3)

    gr.Markdown(
        """
---
**Nothing here is legal advice.** The fine-tuned model fabricates section numbers in
~97% of its unaided answers — that is the finding, and it is also the warning.

[Code & full results](https://github.com/NehaBharti08/rag-vs-finetune) ·
[Adapter](https://huggingface.co/nehabharti0802/rag-vs-finetune-legal-qlora) ·
[Results explorer](https://huggingface.co/spaces/nehabharti0802/statute-citation-explorer)
"""
    )

if __name__ == "__main__":
    demo.launch()
