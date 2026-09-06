# Results

Every number here is measured. Numbers that need an LLM judge are labelled as
such and are never used for a headline, because the judge failed its
pre-registered agreement threshold (§6).

The eval set was frozen before any training ran, over 300 human-verified items,
at digest `6cb89a80fb6ff6c2`. It has not changed since.

---

## 1. The headline, and why it is not the whole story

**Fine-tuning alone made the model worse at the pre-registered metric, and
confidently so.**

|  | No retrieval | With retrieval |
|---|---|---|
| **Base model** | 0.3% | 83.7% |
| **QLoRA fine-tuned** | **0.9% ± 1.0** | **87.5% ± 1.4** |

_Correct-section rate: cites the statutory provision the question came from.
Pre-registered primary metric. Judge-free — a string match against a registry of
real acts and section numbers._

Read at that rung alone, the adapter learned nothing and cost something. That
reading is incomplete, and §2 is why.

## 2. What fine-tuning actually learned

Citation validity is scored as a **ladder**, each rung strictly harder than the
last. The gap between rungs 2 and 5 is where the result lives.

| | A1 base | A2 base+RAG | A3 fine-tuned | A4 FT+RAG |
|---|---|---|---|---|
| 1. Produced a parseable citation | 98.7% | 100.0% | 100.0% | 100.0% |
| 2. Named an act in the corpus | 98.7% | 98.3% | 100.0% | 99.7% |
| 3. **Named the CORRECT act** | **47.7%** | **96.0%** | **92.1% ± 1.7** | **98.2% ± 0.5** |
| 4. Cited a section that exists | 78.3% | 98.3% | 99.0% | 99.7% |
| 5. **Cited the CORRECT section** | **0.3%** | **83.7%** | **0.9% ± 1.0** | **87.5% ± 1.4** |
| Right act, wrong section | 47.3% | 12.3% | **~92%** | 10.3% |

**Fine-tuning took statute routing from a coin flip to near-ceiling: 47.7% →
92.1% ± 1.7, +44 points.** Section-level accuracy stayed indistinguishable from
zero (0.9% ± 1.0, against the base model's 0.3%).

The gap is roughly 26× the seed-to-seed spread, making this the most robust
result in the project — and it is the one found by reading raw responses rather
than by planning to measure it.

So the adapter did learn real, verifiable content — *which of four statutes
governs a question* — and none of the content that requires memorising which of
~1,090 sections it is. A low-cardinality mapping is learnable from 2,830
examples; a high-cardinality one is not.

**The prompts make this sharper, not weaker.** A1's prompt explicitly names all
four statutes and warns against citing the repealed Acts they replaced — and
still gets 47.7%. A3's prompt is literally `{question}`, 52 tokens, with no
statute list and no format specification, because that is what it was trained
on. The fine-tuned model routes to the correct statute **unaided, from
parameters**, and beats a base model that was handed the answer list.

> ⚠️ `act_correct_rate` was added **post-hoc**, after reading A3's responses. The
> frozen-harness pre-commit hook blocked the commit, correctly. It was kept
> because the pre-registered primary metric is unchanged, the eval set did not
> move, and no result reversed — it explains the headline rather than rescuing
> it. See threat 16 in [THREATS_TO_VALIDITY.md](THREATS_TO_VALIDITY.md).

### Why this is the dangerous failure mode, not the reassuring one

**~92% of A3's answers name the correct statute with the wrong section** — 271 of
300 on seed 42, and the pattern replicates on all three.

That is the most authoritative-looking error a legal model can make. A citation
to the wrong Act is visible to any lawyer at a glance. A citation to the right
Act with a plausible section number is not — it has to be looked up. Fine-tuning
did not reduce the error rate here; it made the errors harder to catch.

## 3. Where each arm actually fails

Assigned per item from the response logs, not inferred from aggregate rates.

| Outcome | A2 base+RAG | A4 FT+RAG |
|---|---|---|
| Retrieved the source, cited it correctly | 251 | 265 |
| Retrieved the source, **still cited wrong** (generation failure) | 31 | **17** |
| Missed the source, still correct (parametric recovery) | **0** | **0** |
| Missed the source, wrong (retrieval failure) | 18 | 18 |

Three things fall out of this table.

**Retrieval failure is identical — exactly 18 items in both arms.** They share
one index, so this is what it must be, and it confirms the decomposition is
sound. It also fixes a ceiling: 18 items are unreachable without a better
retriever, and no amount of fine-tuning touches them.

**The entire A4-over-A2 gain is generation.** 31 → 17 errors on identical
retrieved context. That is the adapter's real contribution, measured with the
retriever held fixed — a 45% reduction in generation-stage errors.

**Parametric recovery is zero in both arms.** Neither model — base or fine-tuned
— *ever* answered correctly when retrieval missed. Not once in 36 opportunities.
This is the sharpest evidence in the project that the adapter holds no usable
section-level knowledge: when the index failed, the weights had nothing to add.

For the no-retrieval arms the taxonomy is different, because there is no
retriever to blame:

| Outcome | A1 base | A3 fine-tuned |
|---|---|---|
| Wrong act entirely | 153 | **29** |
| Right act, wrong section | 142 | **271** |
| No parseable citation | 4 | 0 |
| Correct | 1 | 0 |

Fine-tuning converted wrong-act errors into right-act errors — 153 → 29 — and
converted nothing into correct answers.

## 4. The null result on this benchmark's central hypothesis

Parametric answerability was the **stratification variable this whole benchmark
was designed around**. VidyaRAG filters such questions out; this project turned
that filter into a stratification precisely so fine-tuning would have somewhere
it could win. It was the headline the design predicted.

Correct-section rate by stratum:

| Stratum | A1 base | A2 base+RAG | A3 fine-tuned | A4 FT+RAG |
|---|---|---|---|---|
| parametric answerable (n=80) | 1.2% | 86.2% | 0.0% | 87.5% |
| parametric unanswerable (n=220) | 0.0% | 82.7% | 0.0% | **88.6%** |

**It did not separate the arms.** The strata differ by at most a few points, and
in A4 — the best arm — the gap runs the *wrong way*: the model does slightly
better on questions it supposedly could not answer parametrically.

This is a null result on the design's central hypothesis and it is reported
here, prominently, rather than dropped. The methodological correction to
VidyaRAG's filter was still right — filtering would have *guaranteed* the
no-retrieval arms failed and rigged the grid. But the stratification it enabled
turned out to measure nothing.

Two candidate explanations, neither tested here:
- The 26.7% answerability labels come from the judge that failed its agreement
  threshold (§6). Human-calibrated the true rate is nearer 10%, so the
  "answerable" stratum is probably substantially mislabelled.
- Answerability was measured as *content* recall, but the task requires *section
  number* recall. A model can know what the law says without knowing where it
  is written, and this corpus scores only the latter.

## 5. Fine-tuning is not free at inference

Measured in a GPU window **verified exclusive at all six sampling points**
(0 MiB held by any other process throughout). n = 25 items × 3 repeats per arm.

| Arm | Correct section | p50 | p95 | GPU-s/query | Prompt tokens |
|---|---|---|---|---|---|
| A1 base | 0.3% | 3.14s | 3.73s | 3.11 | 487 |
| A2 base + RAG | 83.7% | 3.26s | 4.68s | 3.28 | 1640 |
| A3 fine-tuned | 0.0% | **5.38s** | 7.23s | 5.38 | **54** |
| A4 FT + RAG | 88.3% | **6.06s** | 9.95s | 6.45 | 1176 |

**A3 is 1.7× slower than A1 while sending 9× fewer prompt tokens.** The adapter
is attached **unmerged**, so every forward pass pays extra LoRA matmuls, and that
cost exceeds everything saved by dropping the prompt.

### There is no payback point, at any query volume

Fine-tuning costs GPU-hours once; retrieval costs prefill tokens forever. Which
wins depends on volume, so the honest question is where they cross:

```
N* = training_gpu_seconds / (rag_gpu_seconds − ft_gpu_seconds)
```

The denominator is **negative** here — 3.28 − 6.45 — so there is no crossover.
Training the adapter cost 0.248 GPU-hours (the epoch-1 checkpoint actually
evaluated; 0.743 for the full 3-epoch search), and it never amortises because
the fine-tuned arm is more expensive *per query* as well as up front.

| Comparison | Quality change | Crossover |
|---|---|---|
| FT replaces RAG (A2→A3) | **−83.7 points** | never |
| FT added to RAG (A2→A4) | +4.7 points | never |

Scope limit, stated plainly: this measures an **unmerged** deployment. Merging
the adapter into the base weights would remove most of the overhead and could
flip the sign of the denominator. It was not done because this project's
artifact is a ~160 MB adapter, not a 15 GB merged checkpoint. A merged
deployment is a different measurement and this benchmark did not make it.

Full detail: [`reports/frontier.md`](../reports/frontier.md).

## 6. The judge failed its pre-registered threshold

| | |
|---|---|
| Cohen's κ vs 100 human labels | **0.452** |
| Pre-registered threshold | 0.60 |
| **Verdict** | **FAIL** |

So LLM-judged accuracy is demoted to a secondary metric and **every headline
number in this document is judge-free**. That rule was fixed in
[METHODOLOGY.md](METHODOLOGY.md) before the labels were collected.

The bias matters more than the agreement. The judge calls an answer fully
correct **2.7×** as often as the human does (27% vs 10%), and the confusion
matrix shows why: it never scored a human-2 below 2. All of its error is false
positives — it is a perfect-recall, low-precision correctness detector.

This is why the base model's **26.7%** parametric-answerability rate is reported
as **overstated**, with a human-calibrated estimate nearer **10%**.

Full detail: [`reports/judge_agreement.md`](../reports/judge_agreement.md).

## 7. Catastrophic forgetting: not catastrophic, but not free

MMLU, logprob-scored (no generation, no judge), 100 items per subject.

| Subject | Base | Adapted | Δ |
|---|---|---|---|
| professional_law (in-domain) | 56.0% | 49.0% | **−7.0** |
| jurisprudence (in-domain) | 82.0% | 78.0% | −4.0 |
| college_biology (out-of-domain) | 84.0% | 79.0% | −5.0 |
| formal_logic (out-of-domain) | 54.0% | 54.0% | **0.0** |
| **in-domain mean** | 69.0% | 63.5% | **−5.5** |
| **out-of-domain mean** | 69.0% | 66.5% | **−2.5** |

**Fine-tuning cost more in-domain legal capability than out-of-domain
capability** — the opposite of the usual framing, in which adaptation trades
general ability for domain ability. Here it degraded the domain it was adapted
to, twice as much as the domains it ignored.

That is consistent with everything above: the adapter overwrote legal reasoning
with legal *formatting*.

Caveat that limits this: MMLU has no Indian-law subject. `professional_law` and
`jurisprudence` are US-centric, so this measures whether general legal reasoning
survived adaptation — not whether Indian statutes were learned.

## 8. Results that contradicted my expectations

Collected here because the brief required them reported prominently rather than
buried.

1. **Fine-tuning taught act routing, which I did not predict and initially did
   not measure.** The metric suite had no rung for it. I found it by reading raw
   responses, and it required a post-hoc metric addition that the project's own
   freeze mechanism blocked.
2. **The stratification variable the benchmark was designed around measured
   nothing** (§4), and in the best arm ran backwards.
3. **Forgetting was worse in-domain than out-of-domain** (§7).
4. **`out_of_corpus_act_rate` came back at ~0%**, not the IPC-302 reflex I
   predicted. The inference prompt lists the four statutes, so act choice is
   constrained and failure lands entirely on section numbers. The metric
   measured what the prompt left free — a lesson about metric design, not about
   the model.
5. **Fine-tuning made the model slower** — 1.7× — despite a 9× shorter prompt,
   so there is no query volume at which it pays back (§5).
6. **Two bugs in my own validity defenses**, both found in Phase 6, both
   recorded in [THREATS_TO_VALIDITY.md](THREATS_TO_VALIDITY.md): latency
   contention sampled at the wrong time, and an exclusivity check that was
   detecting its own model.
7. **The variance run deleted one of my own headlines.** "Fine-tuning alone
   scores 0.0%, below the untuned base model" read well and was a single-seed
   artifact. Across three seeds it is 0.9% ± 1.0, which contains the base
   model's 0.3%. The finding is now weaker and correct: indistinguishable, not
   worse. The same run left the act-routing result essentially untouched, which
   is what makes the contrast informative rather than merely deflating.

## 9. What this does not establish

- **Three seeds, not more.** Enough for mean ± std; not enough for a
  significance test. A4's +3.9 over A2 exceeds one standard deviation and holds
  on all three seeds, which is the strongest claim three runs support.

  The variance run earned its cost by **removing** a claim: "fine-tuning alone
  scores 0.0%, below the untuned base model's 0.3%" was a seed-42 artifact. The
  true value is 0.9% ± 1.0, which contains 0.3%. Fine-tuning alone is not
  measurably *worse* than the base model at citing sections — it is
  indistinguishable from it, and both are indistinguishable from zero.
  Full detail: [`reports/seeds.md`](../reports/seeds.md).
- **Abstention** is measured on a 60-item hand-written stratum frozen separately
  from the main gold set — see §10.
- **One corpus, one model, one adapter configuration.** No hyperparameter sweep
  was run, so "QLoRA fails at this" is really "this QLoRA configuration failed
  at this".
- **A public corpus.** Indian statutes are public text. Conclusions may not
  transfer to a genuinely private domain, which is the case fine-tuning is
  usually argued for.
