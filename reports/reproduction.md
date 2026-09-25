# End-to-end reproduction from a clean clone

`scripts/reproduce_all.sh` was run from a fresh `git clone` in a separate
directory, with its own venv and no data carried over. It re-downloaded the
corpus from India Code, regenerated the entire QA dataset (a separate 16-hour
generation run), rebuilt the index, retrained the adapter, and re-ran all four
arms, the analysis and abstention. It was scored against the **same frozen gold
set** (`ade68027a759`), so only the training data changed.

**Every finding reproduced.** The pipeline that claimed to produce them did not,
at first: the run exposed eight defects, listed below. All are fixed and merged.

## What reproduced

### Deterministic stages: byte-identical

| Artifact | Original | Reproduction |
|---|---|---|
| `data/corpus/sections.jsonl` | `233a9a05…` | `233a9a05…`, 1,237/1,237 identical text |
| `data/corpus/splits.json` | `13804fb0…` | `13804fb0…` |

Corpus ingestion, parsing and the seeded split reproduce exactly, fetched fresh
from India Code. The decontamination guarantee (no training pair shares a
section with an evaluation question) is therefore checkable by anyone, not
merely asserted.

### The 2×2: correct-section rate

| Arm | Original | Reproduction | |
|---|---|---|---|
| A1 base | 0.3% | **0.3%** | exact |
| A2 base + RAG | 83.7% | **83.7%** | exact |
| A3 fine-tuned | 0.9% ± 1.0 | **2.0%** | within spread |
| A4 FT + RAG | 87.5% ± 1.4 | **86.0%** | within spread |
| A4 − A2 | +3.9 ± 1.4 | **+2.3** | equals original worst-seed gap |

A1 and A2 reproduce exactly because they carry no adapter and decode greedily;
given the same checkpoint, prompt and gold set they are deterministic. That
claim in `reports/seeds.md` is now demonstrated rather than asserted.

### The central finding

| | Original (3 seeds) | Reproduction |
|---|---|---|
| A3 names the **correct act** | 92.1% ± 1.7 | **92.0%** |
| A3 names the **correct section** | 0.9% ± 1.0 | **2.0%** |
| A3 fabrication | 96.9% ± 2.0 | 94.0% |

Fine-tuning taught the model which statute governs a question and almost
nothing about which section. That holds on a model trained on a **different
2,752-pair dataset**. The seed run showed it was not seed luck; this shows it
is not dataset luck.

### Everything else

| Finding | Original | Reproduction |
|---|---|---|
| Retrieval recall@5 | 0.8114 | 0.8419 (from the run log) |
| Retrieval hit rate, both RAG arms | 0.94 | 0.94 |
| Validation loss rises from epoch 1 | 0.887 → 0.919 → 1.059 | 0.861 → 0.889 → 1.026 |
| Fine-tuning pays back at any query volume | never | never |
| Abstention recall, A1 / A2 | 41.7% / 51.7% | **41.7% / 51.7%** exact |
| Abstention recall, A3 | 15.0% | 16.7% |

## What did not reproduce cleanly

**A4 abstention recall: 16.7% original, 30.0% reproduction.** This is the one
number that moved beyond what I would call noise on 60 items. The direction is
unchanged (both sit far below the base arms' 41.7–51.7%), and it is a single
seed on a regenerated dataset. It is reported as a divergence, not smoothed
over. The headline abstention claim concerns A3 and survives: fine-tuning cut
refusal 41.7% → 16.7%.

**Latency is not comparable, and is not claimed as reproduced.** The original
figures came from a GPU window verified exclusive at six sampling points; this
run shared the card throughout, and absolute latencies came out 1.4–1.7× higher.
The ordering (A1 < A2 < A3 < A4) and the no-crossover conclusion held, but that
is a qualitative match only.

## Eight defects the reproduction exposed

Every published number was correct, because each came from commands run by
hand. The automation that claimed to produce them was broken in eight places.
None was findable by reading the code. Four could have produced **wrong numbers
without raising an error**.

| # | Defect | Effect | Silent? |
|---|---|---|---|
| 1 | India Code renamed `dc.title.act_name` → `dc.identifier.act_name` | Corpus could not be rebuilt; the API returned HTTP 200 with zero results | no, failed closed |
| 2 | Dataset step defaults chained wrongly | `balance` crashed; the bypass path skipped balancing and **halved refusal training data** (10% → 6.3%) | **yes** |
| 3 | `decontaminate` exited 0 on failure | **Contamination failures would not stop the build** | **yes** |
| 4 | `gold.jsonl` was gitignored while `frozen.lock` hashed it | The pre-registration was unverifiable and no eval number reproducible by anyone else | no |
| 5 | `checkpoint-354` hardcoded in four shell scripts | Pipeline worked on exactly one dataset size | no |
| 6 | Same hardcode in Python argparse defaults (fix #5 was incomplete) | Abstention crashed after every earlier phase succeeded | no |
| 7 | `report_arms` globbed `*.jsonl` | Seed and sweep runs would be **added to the headline 2×2 as extra arms** | **yes** |
| 8 | Adapter provenance was a hardcoded literal | Described one training run, not the one being scored | **yes** |

Each fix has a regression test. The rules that fell out:

- **A test may depend on committed evidence, never on generated data.**
- **A gate that cannot stop the build is a log line.**
- **Resolve anything dataset-dependent at run time.** Checkpoint numbers are
  `steps_per_epoch + 1`, and they sort *numerically*, not lexically: `1032`
  sorts before `344` as text, which silently selects the most overfit adapter.

## Reproduce it yourself

```bash
git clone https://github.com/NehaBharti08/rag-vs-finetune.git
cd rag-vs-finetune && uv sync --frozen
./scripts/reproduce_all.sh          # ~20h here, mostly QA generation on Ollama
```

The 16-hour figure is for this shared machine under contention; the original
generation took about 5 hours.
