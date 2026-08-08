# Threats to validity

Every way this benchmark could produce a misleading result, and what the design
does about it. Written before any number was measured, so it cannot be a
post-hoc rationalization of whatever came out.

Each threat is stated as harshly as a reviewer would state it. A defense that
is only a caveat in a README is marked as such rather than dressed up.

---

## 1. The inherited gold-set filter would rig the entire grid

**The threat.** VidyaRAG's gold set verifies that every answerable question is
*not* answerable from parametric knowledge without retrieval. For a RAG
evaluation that is exactly right — it is what stops the evaluation measuring
nothing. Inherited unchanged here, it is fatal: if every question requires
retrieval *by construction*, arms 1 and 3 are guaranteed to fail, fine-tuning
can never win on accuracy, and the 2×2 produces a rigged result that looks
rigorous.

**Defense.** Parametric answerability is a **stratification variable, not a
filter**. Every eval item carries `parametric_answerable ∈ {yes, no}`, measured
empirically against the base model, and results are reported per stratum.
VidyaRAG's 60 questions become the `no` stratum of a 300-item superset.

Reporting the interaction between adaptation method and parametric
answerability is a more interesting result than the headline 2×2, and it exists
only because the two projects were designed together.

## 2. Prompt asymmetry

**The threat.** Giving the fine-tuned arm its trained format prompt and the
base arm a naive one. This is the most common cheat in this genre and it
silently manufactures the result — the fine-tune "wins on format" because the
baseline was never asked to comply.

**Defense.** Base arms get a strong, few-shot, format-specified prompt
developed under the *same* effort budget as the fine-tuning data. Both prompts
are committed under `prompts/inference/`, and the number of prompt-engineering
iterations spent on each is logged. If the base prompt got less effort, the
comparison is not honest and the log will show it.

## 3. Quantization confound

**The threat.** Fine-tuned arms run 4-bit NF4 plus an adapter. If the base arms
run bf16, then quantization is confounded with adaptation and no cell of the
grid is interpretable.

**Defense.** All four arms run an identical NF4 configuration, enforced by a
single shared `QuantConfig`. A bf16 base arm is reported separately as a sanity
reference only, never as a cell of the 2×2.

## 4. The base model may already know the corpus

**The threat.** Qwen2.5 was near-certainly pretrained on OpenStax — it is
public, CC BY, and heavily crawled. If the base model already knows the corpus,
every cell compresses toward ceiling and the grid says nothing while appearing
to say something.

**Defense.** Measured explicitly in Phase 3 and used as a **go/no-go gate
before any training compute is spent**. If the zero-shot base arm already
scores near ceiling on the primary metric, that *is* the finding and it is
reported prominently rather than buried.

**Residual limitation, stated plainly:** this benchmark measures adaptation on
a **public** corpus. Results may not transfer to a genuinely private domain,
where the base model has no prior exposure and fine-tuning starts from a very
different place. No design choice here fixes that; it is a scope limit.

## 4b. The generator and judge are small local models

**The threat.** Generation and judging both run on `gemma4:e4b` (9.6 GB,
local). It is materially weaker than a hosted model at both jobs. A weak
generator caps the quality and variety of what the fine-tuned arm can learn; a
weak judge adds noise, and possibly bias, to the primary metric.

**This is a deliberate constraint, not an oversight.** The project runs at zero
API cost by choice, so the repository reproduces end to end for anyone with a
GPU and no budget. The provider layer is swappable by configuration, so
upgrading is a one-line change in `.env` rather than a code change.

**Defense, such as it is.** Partial, and worth being blunt about:

- Judge quality is *measured*, not assumed: Cohen's kappa against 100
  human-labelled responses is reported. If it comes out below 0.6 the judge is
  not trustworthy and the analysis says so, leaning on the judge-free metrics
  instead.
- The headline metrics that need no judge at all -- citation validity, format
  adherence, abstention, MMLU -- are unaffected by this threat entirely.
- Phase 1 measured a concrete cost of the weak generator: 2,583 training
  examples averaging 114 tokens, giving ~215 optimizer steps over three epochs.
  That is a thin signal, and `reports/dataset_card.md` names it as the first
  thing to suspect if Phase 4 shows little movement -- ahead of any conclusion
  about fine-tuning as a method.

**Residual risk.** A null result in Phase 4 is genuinely ambiguous between "the
method does not help here" and "the training data was too weak to show it".
That ambiguity cannot be resolved from inside this configuration, so any null
result will be reported with both readings stated.

## 5. Judge family bias

**The threat.** An LLM judge that shares a model family with one of the arms
scores that arm favourably.

**Defense.** The judge is never Qwen-family — enforced in `settings.py`, which
refuses to construct a configuration violating it, rather than being left as a
note. Judge quality is reported as Cohen's κ against 100 human-labelled
responses, and a second judge from a third family gives inter-judge agreement.
If κ < 0.6 the judge is not trustworthy, and the analysis leans on the
mechanical metrics instead — a fact that will be stated, not hidden.

## 6. Generator/student family match

**The threat.** If the synthetic training data is generated by a Qwen model,
the fine-tuned arm is partly distilling its own family's style, which inflates
its apparent gain.

**Defense.** Same mechanism as #5: the generator is never Qwen-family, enforced
in code.

## 7. A weak retriever would strawman RAG

**The threat.** If retrieval is bad, the RAG arms lose for reasons that have
nothing to do with RAG as an approach, and the headline conclusion is wrong.

**Defense.** `reports/retrieval_recall.md` reports recall@k on the gold set
**independently of the arms**. If recall@5 is below ~0.8, the RAG arms' failures
are retrieval failures and the report says so explicitly rather than letting
the reader infer a conclusion about method.

## 8. The format metric is circular

**The threat.** The fine-tuned model is trained on exactly the format the
metric rewards. It "winning" on format adherence is tautological, not a
finding. Reporting it as a win would be dishonest.

**Defense.** Stated as such. Format adherence is reported as a **manipulation
check** — evidence that training did what it was supposed to — not as a result.
The non-trivial question is reframed: *does hitting the format cost accuracy?*

## 9. RAGAS context metrics are undefined for half the grid

**The threat.** Faithfulness, context precision and context recall are defined
relative to retrieved context. Arms 1 and 3 have no context. Reporting these
across all four arms would be nonsense dressed as rigor.

**Defense.** Cross-arm metrics are restricted to those defined for all four
arms: factual accuracy, citation validity, format adherence, abstention
behaviour, latency, and MMLU. RAGAS is reported **within the retrieval arms
only**, for continuity with VidyaRAG.

## 10. Evaluation generated by the same model as training data

**The threat.** If both come from the same generator with the same prompts, the
evaluation rewards matching that generator's idiosyncrasies rather than being
correct.

**Defense.** The eval set uses a different model and different prompts from the
training data, and all 300 items are human-verified. 300 is small enough to
actually do this rather than claim it.

## 11. Single-seed results

**The threat.** One training run's number is indistinguishable from noise, and
reporting it as a result is this genre's most common credibility failure.

**Defense.** Three seeds on the final configuration, reported as mean ± std.
The judge runs at temperature 0 and its self-consistency is reported.

## 12. Latency measured on a contended GPU

**The threat.** This box is shared. Latency measured while another job runs is
noise, and cost-per-query derived from it is wrong.

**Defense.** Latency is measured in an exclusive GPU window with n ≥ 5 repeats,
and the contention state at measurement time is recorded. Cost is *also*
reported in GPU-seconds, which is robust to contention, alongside dollars.

## 13. Multiple comparisons

**The threat.** Four arms × ~8 metrics × 4 strata is a great many chances for
something to look significant by accident, and a reader has no way to know how
many comparisons were run before one was reported.

**Defense.** The primary metric is **pre-registered** here, before any number
exists: *factual accuracy on the eval-unseen split, reported separately by
parametric answerability*. Every other metric is explicitly secondary and
exploratory, and labelled that way in `RESULTS.md`. Bootstrap confidence
intervals on the primary.

## 14. Silent configuration drift between the two repos

**The threat.** The apples-to-apples claim rests on this project's retrieval
arm matching VidyaRAG's frozen baseline. Six weeks from now someone tunes a
chunk size in one repo, and every reported delta quietly stops being
comparable — with nothing failing.

**Defense.** `configs/retrieval.yaml` records the upstream repo, profile and
commit it mirrors. `tests/test_retrieval_mirror.py` hard-codes the expected
values rather than reading the file it checks, so drift fails a test instead of
passing silently.

## 15. The task mix could be rigged toward one arm

**The threat.** Choosing question types after seeing preliminary results, or
choosing a mix that structurally favours one approach.

**Defense.** The mix is declared before generation (`configs/generation.yaml`)
and deliberately contains items favouring *both* sides: multi-hop synthesis,
where chunk retrieval is structurally weak, and evaluation on unseen sections,
where retrieval has the structural advantage. Per-type results are reported so
any reader can re-weight the aggregate themselves.

## 16. A metric invented after seeing results

**The threat.** The most insidious failure, because it leaves no trace: run the
grid, notice one arm looks good on some axis, add that axis to the metric
suite.

**Defense.** The harness is frozen before training begins.
`configs/eval/frozen.lock` hashes the eval set, judge prompts, and metric
source; the runner asserts the hash before producing numbers and a pre-commit
hook fails any commit that changes one without re-freezing. Re-freezing is
possible — it just cannot be silent.
