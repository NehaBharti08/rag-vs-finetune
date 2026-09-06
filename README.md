<div align="center">

# rag-vs-finetune

**Does fine-tuning beat retrieval? I built both and measured it, instead of guessing.**

A 2×2 factorial benchmark — {base model, QLoRA fine-tuned} × {no retrieval, retrieval} —
over one corpus, one evaluation set, one base checkpoint.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Adapter on HF](https://img.shields.io/badge/%F0%9F%A4%97%20adapter-Hub-yellow.svg)](https://huggingface.co/nehabharti0802/rag-vs-finetune-legal-qlora)
[![Corpus: Indian statutes](https://img.shields.io/badge/corpus-Indian%20statutes-lightgrey.svg)](ATTRIBUTION.md)

</div>

> **Status: Phase 5 — the full 2×2 is measured.** This README is built up phase
> by phase.
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

**Fine-tuning taught the model which *statute* governs a question, and nothing
about which *section* of it.** Statute routing went from a coin flip to
near-ceiling — **47.7% → 92.1% ± 1.7** — while correct-section accuracy stayed
at **0.9% ± 1.0**, statistically indistinguishable from the untuned base model's
0.3%.

That is one adapter learning a 4-way mapping and failing a ~1,090-way one from
2,830 training examples. It is not "fine-tuning doesn't work"; it is a claim
about what a thin adapter can absorb, and about which half of a citation is
cheap to learn.

All fine-tuned figures are **mean ± std over 3 seeds**. The base arms carry no
adapter and run greedy, so they are deterministic and quoted as single values.

### The full 2×2 (judge-free metrics)

|  | No retrieval | With retrieval |
|---|---|---|
| **Base model** | 0.3% | 83.7% |
| **QLoRA fine-tuned** | **0.9% ± 1.0** | **87.5% ± 1.4** |

*Correct-section rate — cites the statutory provision the question came from.
Fine-tuned cells are mean ± std over 3 seeds.*

Scored as a ladder, because a single "citation validity" number hides the result:

| | A1 base | A2 +RAG | A3 fine-tuned | A4 FT+RAG |
|---|---|---|---|---|
| Named an act in the corpus | 98.7% | 98.3% | 100.0% | 99.7% |
| **Named the CORRECT act** | **47.7%** | 96.0% | **92.1% ± 1.7** | 98.2% ± 0.5 |
| Cited a section that exists | 78.3% | 98.3% | 99.0% | 99.7% |
| **Cited the CORRECT section** | **0.3%** | **83.7%** | **0.9% ± 1.0** | **87.5% ± 1.4** |
| Fabrication rate | 78.0% | 14.7% | **96.9% ± 2.0** | 12.0% ± 1.1 |
| Format valid | 99.7% | 100.0% | 100.0% | 100.0% |
| Mean prompt tokens | 486 | 1603 | **52** | 1139 |
| Latency p50 | 3.14s | 3.26s | **5.38s** | 6.06s |

The prompts make the routing result stronger, not weaker. A1's prompt
**explicitly lists all four statutes** and warns against citing the repealed Acts
they replaced — and still scores 47.7%. A3's prompt is literally `{question}`,
52 tokens, no statute list, because that is what it was trained on. The
fine-tuned model routes correctly **unaided**, beating a base model that was
handed the answer list.

### This is the dangerous failure mode

**~92% of A3's answers name the correct statute with the wrong section** (271 of
300 on seed 42, and the pattern holds on all three).

A citation to the wrong Act is obvious to any lawyer. A citation to the right Act
with a plausible section number has to be looked up. Fine-tuning did not reduce
the error rate — it made the errors harder to catch.

### Fine-tuning is not free at inference, and never pays back

A3 is **1.7× slower** than A1 while sending **9× fewer prompt tokens** — the
adapter is attached unmerged, so every forward pass pays extra LoRA matmuls.

That kills the usual economic argument. Fine-tuning is a one-off cost that has to
amortise against a per-query saving, and here the fine-tuned arms are *more*
expensive per query (5.38 and 6.45 GPU-s) than RAG (3.28). The crossover volume
is not large — it does not exist.

Measured in a GPU window verified exclusive at all six sampling points.
Scope limit: this is an **unmerged** deployment; merging would remove most of the
overhead, and this benchmark did not measure that.

### The fourth cell, and where its gain comes from

A4 beats A2 by **+3.9 ± 1.4 points** — and every one of the three seeds beats it,
worst-seed gap +2.3. The failure taxonomy says exactly why:

| | A2 base+RAG | A4 FT+RAG |
|---|---|---|
| Retrieved source, cited correctly | 251 | 265 |
| Retrieved source, **still wrong** | 31 | **17** |
| Missed source, still correct | **0** | **0** |
| Missed source, wrong | 18 | 18 |

Retrieval failures are **identical** — both arms share one index — so the entire
gain is generation: 31 → 17 errors on the same retrieved context (seed 42; the
per-item taxonomy is reported for one seed, the rates for three).

And **parametric recovery is zero in both arms**. Neither model ever answered
correctly when retrieval missed, in 36 opportunities. When the index failed, the
fine-tuned weights had nothing to add.

### Fine-tuning destroyed the ability to refuse

| Arm | Refused | Recall | Precision |
|---|---|---|---|
| A1 base | 25/60 | **41.7%** | 92.6% |
| A2 base + RAG | 31/60 | **51.7%** | 100% |
| A3 fine-tuned | 9/60 | **15.0%** | 100% |
| A4 FT + RAG | 10/60 | **16.7%** | 100% |

Measured on 60 hand-written unanswerable questions plus 60 answerable controls.

**Fine-tuning cut refusal by ~3×**, and the training set contained **317
refusal examples** — 10% of the data — put there specifically to prevent this.
It did not. "You just didn't teach it to refuse" is ruled out by construction,
which makes this the mechanism behind the 96.9% fabrication rate rather than a
guess at it.

Worst cell in the table: **A4 refuses 0/12 repealed-law questions.** Asked about
IPC §302, the fine-tuned+RAG model answers every time — and the corpus exists
*because* the IPC was replaced in 2024. The questions a real user is most likely
to ask from memory are the ones it is least likely to decline.

Every arm but A1 has 100% abstention precision and 0% false abstention. These
models are not confused about what they cannot answer; they are unwilling to say
so.

### The null result on this benchmark's own hypothesis

Parametric answerability was the stratification variable this project was
*designed around* — the one place fine-tuning was expected to win.

| Stratum | A1 | A2 | A3 | A4 |
|---|---|---|---|---|
| answerable (n=80) | 1.2% | 86.2% | 0.0% | 87.5% |
| unanswerable (n=220) | 0.0% | 82.7% | 0.0% | **88.6%** |

_Per-stratum figures are seed 42; the stratum split is computed per-item and is
not re-derived per seed._

It did not separate the arms, and in the best arm the gap runs the **wrong way**.
Reported here rather than dropped.

### Forgetting was worse in-domain than out

| | base | adapted | Δ |
|---|---|---|---|
| in-domain (professional_law, jurisprudence) | 69.0% | 63.5% | **−5.5** |
| out-of-domain (college_biology, formal_logic) | 69.0% | 66.5% | −2.5 |

The opposite of the usual framing. Adaptation degraded the domain it was adapted
*to*, twice as much as the domains it ignored — consistent with an adapter that
overwrote legal reasoning with legal formatting.

### Where the base model actually stands

**26.7%** of gold questions are answered correctly by the base model with no
retrieval — and that number is **overstated**. The local judge was measured
against 100 human labels at Cohen's κ = **0.452**, and calls 2.7× as many answers
fully correct as a human does. Human-calibrated, the true rate is nearer **10%**.

κ is below the pre-registered 0.60 threshold, so **LLM-judged accuracy is a
secondary metric only** and every headline number above is judge-free. The rule
was fixed before the measurement, not after.

_Full analysis: [`docs/RESULTS.md`](docs/RESULTS.md). Evidence:
[`reports/arms_comparison.md`](reports/arms_comparison.md),
[`reports/failures.md`](reports/failures.md),
[`reports/judge_agreement.md`](reports/judge_agreement.md)._

## Relationship to VidyaRAG

This project began as the empirical counterpart to
**[VidyaRAG](https://github.com/NehaBharti08/VidyaRAG)** and inherited its
retrieval design. One inherited claim had to be **retired**, and saying so
matters more than the claim did.

**The two repos no longer share a corpus.** VidyaRAG indexes OpenStax biology;
this project moved to Indian statutes (see below). So the retrieval arms here are
*not* "the same pipeline on the same data" and the apples-to-apples claim that
justification originally rested on is void. It is removed rather than quietly
left standing.

What survives is narrower and still worth stating: the retrieval configuration
(chunk 512 / overlap 64, `top_k` 20 → 5, dense-only, no reranker) is held fixed
at VidyaRAG's frozen `baseline` values. Not for cross-repo comparability, which
no longer exists, but so that **retrieval is a constant across all four arms of
this grid** — the 2×2 isolates adaptation, and a retriever tuned per-arm would
destroy that. `configs/retrieval.yaml` records the provenance and
`tests/test_retrieval_mirror.py` fails if the values drift.

The repos stay separate on purpose: VidyaRAG is deliberately torch-free (~400 MB
deployable image) and this project needs torch, bitsandbytes and CUDA.

### One inherited decision that had to be changed

VidyaRAG verifies that every gold question is **not** answerable from parametric
knowledge without retrieval. For a RAG evaluation that is exactly right — it is
what stops the evaluation from measuring nothing.

For this 2×2 it would be fatal. If every question requires retrieval *by
construction*, the no-retrieval arms are guaranteed to fail, fine-tuning can
never win, and the grid produces a rigged result that looks rigorous.

So here, parametric answerability is a **stratification variable, not a filter**.
Every eval item is labelled and results are reported per stratum.

That correction was right, and it still **did not work** — the stratification
separated nothing (see Results). Both halves are reported: the reasoning was
sound, the hypothesis it enabled was wrong.

## Experimental design

_Full detail in [docs/METHODOLOGY.md](docs/METHODOLOGY.md); threats and their
defenses in [docs/THREATS_TO_VALIDITY.md](docs/THREATS_TO_VALIDITY.md)._

| Concern | Choice | Why |
|---|---|---|
| Base model | `Qwen2.5-7B-Instruct` | Same checkpoint in **every** arm, so the comparison isolates the adaptation method rather than confounding it with a model swap |
| Quantization | 4-bit NF4, **all arms** | If fine-tuned arms ran 4-bit and base arms ran bf16, quantization would be confounded with adaptation |
| Adaptation | QLoRA, all 7 projections | Attention *and* MLP: adapting every linear layer is what closes the gap to full fine-tuning |
| Generator / judge | never Qwen-family | A same-family generator distils its own style into the fine-tuned arm; a same-family judge favours it. Enforced in `settings.py`, not in a footnote |
| API cost | **$0.00** | Runs entirely on local Ollama, so anyone with a GPU and no budget reproduces it end to end. The cost is a weaker generator and judge — measured and reported, not hidden: see [threat 4b](docs/THREATS_TO_VALIDITY.md) |
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

Then reproduce everything:

```bash
./scripts/reproduce_all.sh            # ~10h wall clock, ~4h of it GPU
./scripts/reproduce_all.sh --from 3   # resume from any phase
```

Every phase is idempotent and separately runnable, because they have very
different costs and failure modes:

| | Phase | Cost |
|---|---|---|
| `scripts/01_dataset.sh` | corpus → QA, split before generation | ~5h CPU (local Ollama, $0) |
| `scripts/02_build_index.sh` | embed + index | ~20 min CPU |
| `scripts/03_baselines.sh` | A1, A2 + the go/no-go gate | ~40 min GPU |
| `scripts/04_train.sh` | QLoRA | ~45 min GPU |
| `scripts/05_full_grid.sh` | A3, A4 | ~40 min GPU |
| `scripts/06_analysis.sh` | forgetting, latency, frontier | ~40 min GPU |

The GPU here is shared, so long jobs go through `scripts/run.sh <name> <cmd>`,
which detaches into tmux with a timestamped log and records the GPU state at
launch. Every runner resumes from its own checkpoints.

Two human-in-the-loop steps cannot be automated and have their own CLI:

```bash
uv run python -m ragft.eval.label verify   # check gold candidates
uv run python -m ragft.eval.label judge    # grade responses, for Cohen's kappa
uv run python -m ragft.eval.label write    # author the unanswerable stratum
```

The last one is why abstention is currently unmeasured — see
[Limitations](#what-this-does-not-establish).

## Dataset (Phase 1)

**3,171 QA pairs** (2,830 train / 341 val) grounded in **1,090 sections** of four
Indian statutes. Full detail in
[`reports/dataset_card.md`](reports/dataset_card.md); decontamination evidence in
[`reports/decontamination.md`](reports/decontamination.md).

Decontamination is **structural, not best-effort**: sections are assigned to
splits *before* any QA is generated, so a training pair and an eval question
cannot share a source passage. Four checks verify that guarantee rather than
trying to establish it after the fact. All pass.

Two things Phase 1 measured that the plan had guessed wrong:

- **Token length.** Planned ~600 tokens/example and `max_seq_length` 2048;
  measured p99 **231**, mean **137**. A training example is a question plus a
  formatted answer and contains *no passage* — the no-retrieval arm has to recall
  parametrically, so there is no long context to hold. `max_seq_length` is now
  512, and padding to 2048 would have wasted ~11× the compute for nothing.
- **A false positive in my own decontamination check.** The first run flagged 420
  within-train duplicates; the top matches were *"What is phagocytosis?"* against
  *"What is chemical energy?"*. Three-token questions form no 5-gram shingle, so
  their MinHash signatures were empty — and empty signatures are identical to one
  another. Fixed, with regression tests. A check that cries wolf is more
  dangerous than one that is merely absent, because its threshold gets relaxed
  until it stops catching anything.

**The thin-signal limitation, and what fixed it.** Phase 1 flagged ~78 optimizer
steps per epoch as too weak a training signal. Turning **packing off** in Phase 4
raised that to **353 steps/epoch** — same data, roughly the same wall clock,
3.7× more gradient updates. So "the adapter didn't have enough steps" is not
available as an excuse for the results above.

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

## Try it

The adapter is on the Hub: **[`nehabharti0802/rag-vs-finetune-legal-qlora`](https://huggingface.co/nehabharti0802/rag-vs-finetune-legal-qlora)** — ~77 MB of LoRA weights over
`Qwen2.5-7B-Instruct`. Read its card before using it; it leads with why you
should not answer legal questions with it.

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = PeftModel.from_pretrained(base, "nehabharti0802/rag-vs-finetune-legal-qlora")
```

```bash
uv run python demo.py --list-failures    # no GPU needed — reads the eval logs
```

Prints real failures from the committed logs: the question, the gold section,
and what the fine-tuned model cited instead. It is the fastest way to see the
central finding rather than read a table about it.

With a GPU, ask anything and watch all four arms answer side by side, each
citation mechanically checked against the corpus:

```bash
uv run python demo.py "What punishment does the law prescribe for murder?"
uv run python demo.py --interactive
```

**No hosted Space, deliberately.** A free HF Space gets a CPU and 16 GB of RAM
and cannot serve a 4-bit 7B. One that quietly fell back to a smaller model would
demo something this project never measured.

## What this does not establish

Stated here rather than left for a reader to discover:

- **Three seeds, not more.** Enough for mean ± std, not enough for a
  significance test. A4's +3.9 over A2 exceeds one standard deviation and holds
  on every seed, which is the strongest claim three runs can carry — no more.
  The variance run also **killed** a headline: "0.0%, below the base model" was
  a single-seed artifact. See [`reports/seeds.md`](reports/seeds.md).
- **Abstention rests on 60 items and one seed.** Enough to show a 3× effect;
  not enough to put a tight interval on it.
- **One hyperparameter configuration.** No sweep, so "QLoRA fails at this" is
  really "this QLoRA configuration failed at this".
- **A public corpus.** Indian statutes are public text; results may not transfer
  to a private domain, which is the case fine-tuning is usually argued for.

Full list in [docs/RESULTS.md](docs/RESULTS.md) §9. The adapter's own
limitations are in [MODEL_CARD.md](MODEL_CARD.md), which leads with why you
should not use it to answer legal questions.

## Corpus & licensing

Code is MIT. **The corpus is not**, and the basis for reusing it is narrower than
a permissive licence — which is worth stating precisely rather than waving at.

The corpus is the bare text of four Indian statutes, retrieved from
**[India Code](https://indiacode.gov.in)**, the Government of India's official
repository of central legislation:

| Act | Year | Replaces |
|---|---|---|
| The Bharatiya Nyaya Sanhita | 2023 | The Indian Penal Code, 1860 |
| The Bharatiya Nagarik Suraksha Sanhita | 2023 | The Code of Criminal Procedure, 1973 |
| The Bharatiya Sakshya Adhiniyam | 2023 | The Indian Evidence Act, 1872 |
| The Indian Contract Act | 1872 | — |

Reuse rests on **§52(1)(q)(ii) of the Indian Copyright Act, 1957**, which exempts
the reproduction of bare legislative text. That is a **statutory exemption, not a
licence** — it is narrower than CC BY, it does not carry a grant to sublicense,
and it covers the bare text only, not commentary or headnotes. Full reasoning in
[ATTRIBUTION.md](ATTRIBUTION.md).

**Why statutes and not textbooks.** This project originally used OpenStax
biology. The Phase 3 gate measured **75.7%** parametric answerability there — the
base model already knew the corpus, so every cell of the 2×2 compressed toward
ceiling and the training compute would have bought nothing. The same gate on
statutes in force since July 2024 returned **26.7%**. The domain switch was a
measurement, not a preference.

This is an independent student project. It is not affiliated with or endorsed by
the Government of India, and **nothing it produces is legal advice.** The
fine-tuned model fabricates section numbers in 99% of its unaided answers; that
is the finding, and it is also the warning.
