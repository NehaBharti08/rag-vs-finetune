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

300 items, human-verified in full, frozen by hash at digest `6cb89a80fb6ff6c2`
before any training ran.

### What was planned

| Stratum | n | Source |
|---|---|---|
| `parametric_answerable=no`, eval-unseen | 60 | Inherited from VidyaRAG |
| `parametric_answerable=yes`, eval-unseen | 120 | The only stratum where fine-tuning can plausibly win |
| Train-seen absorption slice | 60 | Memorization vs generalization |
| Unanswerable / false-presupposition | 60 | Hand-written |

### What was actually built

The plan is recorded above and the delivered set below, because the difference
matters and hiding it would make this document a description of intentions
rather than of work.

| Stratum | n | Status |
|---|---|---|
| eval-unseen | 240 | delivered |
| train-seen absorption slice | 60 | delivered |
| **unanswerable** | **0** | **NOT delivered** |

Two deviations:

**Parametric answerability is measured, not designed.** It could not be a
construction parameter, because whether the base model knows an answer is an
empirical fact about the model, not a property you can assign to a question when
you write it. The delivered split is **80 answerable / 220 unanswerable**,
measured against the base model after the set was built. This is the correct
handling — assigning the label at authoring time would have meant guessing it —
but it means the stratum sizes were not controllable.

**The unanswerable stratum does not exist.** All 300 items are answerable, so
**abstention is not measured** in any result in this repo. It is reported as
absent rather than estimated. The tooling to close this is built
(`ragft.eval.label write`, `ragft.eval.run_abstention`) and the items will land
in a **separate file with its own freeze** — appending to `gold.jsonl` would
change its digest and invalidate every measured result, which is exactly what
the freeze exists to prevent.

The unanswerable items must be hand-written. An LLM asked to produce
unanswerable questions reliably produces obviously out-of-domain ones, which
makes abstention look easy and the resulting metric meaningless. They have to be
plausible near-misses: questions a competent lawyer might actually ask.

## Metrics, and which need a judge

Judge-free metrics are preferred wherever a mechanical check is possible,
because they cost nothing, cannot be biased, and reproduce exactly.

| Metric | Judge needed? | How |
|---|---|---|
| Citation validity | **No** | Resolves against a registry of every section in the corpus. A cited `§` either exists, is the right one, or is neither |
| Format adherence | **No** | Structural regex over the three-part answer contract |
| Abstention precision / recall / false-abstention | **No** | Refusal detection against known-unanswerable labels |
| MMLU (forgetting probe) | **No** | Log-probability over A/B/C/D — no generation at all |
| Latency p50/p95 | **No** | Measured, exclusive GPU window, n ≥ 5 |
| Factual accuracy | Yes | LLM judge, never Qwen-family, κ reported against 100 human labels |
| Groundedness | Yes | Same |

That the *hallucination* signal is judge-free is the main reason the answer
format carries a section-level citation: it converts a subjective property into
a lookup against a finite registry.

**The judge failed its threshold, and the rule held.** κ = 0.452 against 100
human labels, below the 0.60 pre-registered here before the labels were
collected. So both judge-dependent rows above are demoted to secondary, and
every headline number in this project comes from the judge-free rows. The judge
is also 2.7× more generous than the human, which is why the base model's judged
26.7% answerability is reported as overstated.

Citation validity is scored as a **ladder** — parseable → act exists → act
correct → section exists → section correct — because a single rate reported at
either end is misleading in opposite directions. `act_correct` was added
**post-hoc** (see threat 16); the pre-registered primary metric is the
section-correct rung and it did not change.

## Answer format contract

```
**Answer.** <direct answer, one or two sentences>
**Why.** <two to four sentences stating the rule>
**Source.** The Bharatiya Sakshya Adhiniyam, 2023, §54
```

The three-part shape is inherited from VidyaRAG; the citation is **not**. The
original was page-level (`Biology, §7.3, p.214 (OpenStax, CC BY 4.0)`) and
carried a CC BY attribution suffix. Statutes have neither pages nor a CC BY
obligation — reuse rests on a statutory exemption that requires no attribution
string — so the citation is act + section and the suffix is gone.

Section numbers are a **harsher** test than page numbers were. A page number is
correct if it falls in a range; a section number is a single discrete value out
of ~1,090, and a model that has read a lot of pre-2024 Indian law will
confidently supply IPC numbers that are not in this corpus at all.

## What is deliberately *not* done

- **No merged weights.** Adapters only. Merged 7B weights are 15 GB and add
  nothing reproducible. This has a measured cost: the adapter runs **unmerged**
  at inference, so every forward pass pays extra LoRA matmuls and the fine-tuned
  arms are ~1.7× slower than base. All latency and cost results are therefore
  scoped to an unmerged deployment, and say so.
- **No tuned retriever.** Retrieval config is held fixed across all four arms so
  the grid isolates adaptation. It is not the best retriever obtainable.
- **No vLLM.** Faster, but adds a heavy dependency and LoRA-swap complexity for
  an evaluation that takes ~3 GPU-hours with plain `generate()`. Boring wins.
- **No epoch sweep.** Checkpointing every epoch reads epochs 1/2/3 out of a
  single run, collapsing a 12-run grid to 4.
