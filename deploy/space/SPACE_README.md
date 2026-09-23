---
title: RAG vs QLoRA — Live Demo
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: static
app_file: index.html
pinned: false
license: mit
short_description: Live citation validation over 1,200 measured answers
---

# RAG vs QLoRA fine-tuning — live demo

A 2×2 factorial benchmark over four Indian statutes: {base, QLoRA} × {no retrieval, retrieval}.

**Fine-tuning taught the model which _statute_ governs a question (47.7% → 92.1% ± 1.7) and
almost nothing about which _section_ (0.3% → 0.9% ± 1.0).** It names a real section 99% of the
time and the *correct* one about 1% — 271 of 300 answers cite the right act and the wrong
section, the error a non-lawyer cannot catch.

## What is live

**The citation checker runs the benchmark's own regex and section registry in your browser.**
Paste anything and it resolves against 1,237 real sections — no model, no server, no API. This
is the project's headline judge-free metric, executing client-side.

Try a repealed citation (`The Indian Penal Code, 1860, §302`) to see the failure mode this
corpus was chosen to expose.

## What is not live, and why

The 7B generation. A free Space cannot serve a 4-bit 7B, and a Space that quietly swapped in a
smaller model would be demonstrating something this project never measured.

So the other two tabs serve real measurements rather than fakes:

- **All four arms** — 1,200 responses from the actual benchmark run, each citation graded by
  the repo's own validator. The verdict counts here match `reports/failures.md` exactly.
- **Retrieved context** — the real passages the RAG arms received, computed with the same
  `BAAI/bge-large-en-v1.5` embedder, same chunking (512/64), same `top_k` (20 → 5).

The full interactive version — live retrieval for *any* question you type — is in the repo as
`app.py` and runs locally, or on a Gradio Space with an HF PRO account.

## Links

- [Code and full results](https://github.com/NehaBharti08/rag-vs-finetune)
- [Adapter](https://huggingface.co/nehabharti0802/rag-vs-finetune-legal-qlora)
- [Results explorer](https://huggingface.co/spaces/nehabharti0802/statute-citation-explorer)

Every finding reproduced from a clean clone on an independently regenerated dataset.

**Nothing here is legal advice.** The fine-tuned model fabricates section numbers in ~97% of
its unaided answers; that is the finding, and it is also the warning.
