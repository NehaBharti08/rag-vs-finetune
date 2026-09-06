# Judge agreement

`gemma4:e4b` (temp 0) against **100** human labels on A1 responses.
Scale: 0 = wrong, 1 = partially correct, 2 = fully correct.

## The pre-registered gate

| | |
|---|---|
| Cohen's kappa (unweighted) | **0.4518** |
| Pre-registered threshold | 0.6 |
| **Verdict** | **FAIL** |

kappa is BELOW the pre-registered 0.60 threshold, so LLM-judged accuracy is reported as a SECONDARY metric only and every headline number in this project is judge-free (citation validity, format, MMLU, latency -- all mechanical). The threshold was fixed before the labels were collected.

The threshold was fixed in `docs/METHODOLOGY.md` before these labels were
collected. A threshold chosen after seeing the number is not a threshold.

## Secondary kappas

| Variant | kappa | What it answers |
|---|---|---|
| Unweighted, 3-way | 0.4518 | The pre-registered gate |
| Linear-weighted, 3-way | 0.5112 | Fairer on an ordinal scale -- credits near-misses |
| Binary (fully-correct) | 0.462 | The decision the headline metric depends on |
| Exact agreement rate | 0.67 | Raw, uncorrected for chance -- shown to make the correction visible |

## Bias, which kappa does not measure

| Rater | Calls an answer fully correct |
|---|---|
| Human | **10.0%** |
| Judge | **27.0%** |

The judge is **2.7x** as generous as the human.

This is the more consequential finding. Kappa measures whether two raters move
together; it says nothing about whether one is systematically lenient. A judge
could agree at chance level and still be unbiased, or agree well and inflate
every number it touches. Here it does the latter, which is why the base model's
judged 26.7% parametric-answerable rate should be read as nearer **10%** and why
it is not used as a headline anywhere.

## Confusion matrix

| | judge=0 | judge=1 | judge=2 |
|---|---|---|---|
| human=0 | 44 | 11 | 7 |
| human=1 | 5 | 13 | 10 |
| human=2 | 0 | 0 | 10 |

_Regenerate: `uv run python -m ragft.eval.judge_agreement`_
