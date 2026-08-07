# Methodology

How every number in this repository is produced. Written as decisions are made
rather than reconstructed afterwards, so the reasoning reflects what was
actually known at the time.

Threats to these choices, and their defenses, live in
[THREATS_TO_VALIDITY.md](THREATS_TO_VALIDITY.md).

---

## The design

A 2×2 factorial over two binary factors — adaptation (base / QLoRA) and
retrieval (off / on) — holding everything else constant:

- **One base checkpoint** (`Qwen2.5-7B-Instruct`) in every arm, so the
  comparison isolates the adaptation method rather than confounding it with a
  model swap.
- **One quantization configuration** (4-bit NF4) in every arm, including the
  base-model arms.
- **One evaluation set**, frozen before training begins.
- **One retrieval configuration**, mirrored from VidyaRAG's frozen `baseline`.

The fourth cell — fine-tuned *and* retrieval-augmented — is the one most
write-ups omit. It is included because "X beats Y" is not a credible claim
without it.

## Pre-registered primary metric

Declared before any number existed:

> **Factual accuracy on the eval-unseen split, reported separately for
> parametrically-answerable and parametrically-unanswerable questions.**

Everything else — citation validity, format adherence, abstention behaviour,
latency, cost, MMLU — is **secondary and exploratory**, and labelled as such in
`RESULTS.md`. This is stated up front because four arms × eight metrics × four
strata offers a great many chances for something to look significant by
accident.

Bootstrap confidence intervals accompany the primary metric. Three training
seeds give mean ± std.

## Phase order, and why it is that order

| Phase | What | Why here |
|---|---|---|
| 0 | Environment pinned and **measured** | Every compute estimate is arithmetic until something runs. `reports/env_matrix.md` replaces the planning figures |
| 1 | Corpus + synthetic dataset | Section-level split happens *before* generation, so decontamination is guaranteed by construction rather than filtered after the fact |
| 2 | Retrieval + eval harness, **frozen** | Built before training. A metric invented after seeing results is not a metric |
| 3 | Baseline arms + **go/no-go gate** | If the base model already knows the corpus, training compute buys nothing. Find out before spending it |
| 4 | QLoRA training | Sweep, then 3 seeds at the winner |
| 5 | Fine-tuned arms, full grid | |
| 6 | Analysis | Cost/quality/latency frontier, failure taxonomy, forgetting |
| 7 | Ship | Model card, reproduction script |

The gate at Phase 3 is real. If the zero-shot base arm scores near ceiling, the
recommendation will be to cut Phase 4 rather than spend ~13 GPU-hours
confirming a ceiling effect.

## Evaluation set construction

300 items, human-verified in full, frozen by hash.

| Stratum | n | Source |
|---|---|---|
| `parametric_answerable=no`, eval-unseen | 60 | Inherited from VidyaRAG unchanged |
| `parametric_answerable=yes`, eval-unseen | 120 | New — the only stratum where fine-tuning can plausibly win |
| Train-seen absorption slice | 60 | New — memorization vs generalization |
| Unanswerable / false-presupposition | 60 | New, **hand-written** |

The unanswerable items are hand-written on purpose. An LLM asked to produce
unanswerable questions reliably produces obviously out-of-domain ones, which
makes abstention look easy and the resulting metric meaningless. They have to
be biology-shaped and plausible.

## Metrics, and which need a judge

Judge-free metrics are preferred wherever a mechanical check is possible,
because they cost nothing, cannot be biased, and reproduce exactly.

| Metric | Judge needed? | How |
|---|---|---|
| Citation validity | **No** | Resolves against the canonical corpus TOC. A fabricated `§`/page either exists or does not |
| Format adherence | **No** | Structural regex over the three-part answer contract |
| Abstention precision / recall / false-abstention | **No** | Refusal detection against known-unanswerable labels |
| MMLU (forgetting probe) | **No** | Log-probability over A/B/C/D — no generation at all |
| Latency p50/p95 | **No** | Measured, exclusive GPU window, n ≥ 5 |
| Factual accuracy | Yes | LLM judge, never Qwen-family, κ reported against 100 human labels |
| Groundedness | Yes | Same |

That the *hallucination* signal is judge-free is the main reason the answer
format carries a page-level citation: it converts a subjective property into a
lookup.

## Answer format contract

```
**Answer.** <direct answer>
**Why.** <explanation>
**Source.** Biology, §7.3, p.214 (OpenStax, CC BY 4.0)
```

Adopted verbatim from VidyaRAG, including the license suffix — which is a CC BY
obligation rather than decoration, and conveniently a fixed string, so one
regex catches both a format failure and a license violation.

## What is deliberately *not* done

- **No merged weights.** Adapters only, pushed to the Hub. Merged 7B weights
  are 15 GB and add nothing reproducible.
- **No tuned RAG stack.** The retrieval arm is VidyaRAG's frozen `baseline`,
  not its best configuration. A fully-tuned RAG pipeline against an untuned
  fine-tune would be its own confound.
- **No vLLM.** Faster, but adds a heavy dependency and LoRA-swap complexity for
  an evaluation that takes ~3 GPU-hours with plain `generate()`. Boring wins.
- **No epoch sweep.** Checkpointing every epoch reads epochs 1/2/3 out of a
  single run, collapsing a 12-run grid to 4.
