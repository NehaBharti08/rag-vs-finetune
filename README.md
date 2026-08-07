<div align="center">

# rag-vs-finetune

**Does fine-tuning beat retrieval? I built both and measured it, instead of guessing.**

A 2×2 factorial benchmark — {base model, QLoRA fine-tuned} × {no retrieval, retrieval} —
over one corpus, one evaluation set, one base checkpoint.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Corpus: CC BY 4.0](https://img.shields.io/badge/corpus-CC%20BY%204.0-lightgrey.svg)](ATTRIBUTION.md)

</div>

> **Status: Phase 0 — foundation.** This README is built up phase by phase.
> Sections marked _pending_ are filled in as the work behind them lands.
> No number appears here before it has been measured.

---

## The question

Most "RAG vs fine-tuning" writing online is an opinion with a graph attached.
The comparison is usually two arms — a base model with retrieval against a
fine-tuned model without it — which confounds the two variables it claims to
separate.

This repo runs the full grid:

|  | No retrieval | With retrieval |
|---|---|---|
| **Base model** | zero-shot control | standard RAG |
| **QLoRA fine-tuned** | pure parametric adaptation | fine-tune + RAG |

The fourth cell is the one most write-ups omit, and it is usually the most
interesting. Any conclusion of the form "X beats Y" is only credible with all
four.

## Results

_Pending (Phase 3 onward)._

**Any result that contradicts my expectations goes in this section first.** If
fine-tuning loses on everything except latency, that sentence will be the first
thing you read. A suspiciously clean win is a bug report, not a finding.

## Relationship to VidyaRAG

The retrieval arms are not a reimplementation-from-scratch of "some RAG
baseline". They mirror the frozen `baseline` profile of
**[VidyaRAG](https://github.com/NehaBharti08/VidyaRAG)** — same corpus, same
chunking (512/64), same `top_k` (20 → 5), same citation format — so the
comparison is genuinely apples-to-apples rather than a strawman built to lose.

`configs/retrieval.yaml` records what it mirrors and
`tests/test_retrieval_mirror.py` fails if the two drift apart.

The two repos stay separate on purpose: VidyaRAG is deliberately torch-free
(~400 MB deployable image), and this project needs torch, bitsandbytes and
CUDA. Merging them would cost VidyaRAG the property it was designed around.

### One inherited decision that had to be changed

VidyaRAG verifies that every gold question is **not** answerable from
parametric knowledge without retrieval. For a RAG evaluation that check is
exactly right — it is what stops the evaluation from measuring nothing.

For this 2×2 it would be fatal. If every question requires retrieval *by
construction*, the no-retrieval arms are guaranteed to fail, fine-tuning can
never win, and the grid produces a rigged result that looks rigorous.

So here, parametric answerability is a **stratification variable, not a
filter**. Every eval item is labelled and results are reported per stratum.
VidyaRAG's 60 questions become one stratum of a 300-item superset.

## Experimental design

_Full detail in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); threats and their
defenses in [docs/THREATS_TO_VALIDITY.md](docs/THREATS_TO_VALIDITY.md)._

| Concern | Choice | Why |
|---|---|---|
| Base model | `Qwen2.5-7B-Instruct` | Same checkpoint in **every** arm, so the comparison isolates the adaptation method rather than confounding it with a model swap |
| Quantization | 4-bit NF4, **all arms** | If fine-tuned arms ran 4-bit and base arms ran bf16, quantization would be confounded with adaptation |
| Adaptation | QLoRA, all 7 projections | Attention *and* MLP: adapting every linear layer is what closes the gap to full fine-tuning |
| Generator / judge | never Qwen-family | A same-family generator distils its own style into the fine-tuned arm; a same-family judge favours it. Enforced in `settings.py`, not in a footnote |
| Seeds | 3, mean ± std | Single-run numbers are this genre's most common credibility failure |
| Primary metric | pre-registered | Factual accuracy on eval-unseen, split by parametric answerability. Everything else is explicitly secondary |

## Quickstart

Requires Python 3.11 and an NVIDIA GPU with ≥16 GB VRAM for the training path.
No API key needed — the default provider is local Ollama.

```bash
git clone https://github.com/NehaBharti08/rag-vs-finetune.git
cd rag-vs-finetune

make install          # pinned venv via uv + pre-commit hooks
cp .env.example .env

make smoke            # Phase 0 gate: 20 real QLoRA steps, measures your GPU
make check            # lint + types + fast tests
```

`make smoke` writes [`reports/env_matrix.md`](reports/env_matrix.md). Every
GPU-hour estimate in this project is calibrated from it rather than assumed.

_Dataset, evaluation and training commands land in Phases 1–4._

## Environment note

The dependency matrix here is not a preference, it is a derived constraint:

```
bitsandbytes 0.50  requires  transformers<5
trl          1.9.x requires  transformers>=4.56.2
=> only satisfiable window:  transformers>=4.56.2,<5
```

Verified working versions are in
[`reports/env_matrix.md`](reports/env_matrix.md), produced by an actual GPU run
rather than by reading changelogs.

There is no SLURM on the target machine, so long jobs run under tmux via
`scripts/run.sh` and every runner resumes from its own checkpoints — the GPU is
shared, so a run can lose its slot at any time.

## Corpus & licensing

Code is MIT. The textbooks are **not** — they are OpenStax titles under
CC BY 4.0, and that license travels with every retrieved passage, generated
citation, the derived QA dataset, and the trained adapter.

**Edition matters more than title.** Most OpenStax second editions are CC
BY-**NC-SA**, not CC BY. Only the *first* editions of *Biology* and *Anatomy and
Physiology* are usable here. See [ATTRIBUTION.md](ATTRIBUTION.md).

> Download Biology for free at https://openstax.org/details/books/biology
> Download Anatomy and Physiology for free at https://openstax.org/details/books/anatomy-and-physiology

This is an independent student project, not affiliated with or endorsed by
OpenStax or Rice University.
