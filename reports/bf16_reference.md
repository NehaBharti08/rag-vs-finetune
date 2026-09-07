# bf16 reference arm

**This is not a cell of the 2x2.** Every arm of the grid runs 4-bit NF4 so that
quantization cannot be confounded with adaptation. This arm exists to answer the
one question that defense leaves open: *what does NF4 itself cost the base
model?*

Same prompt, same gold set, same decoding as A1 — the only difference is bf16
weights instead of NF4.

| Metric | A1 (NF4) | A1 (bf16) | delta |
|---|---|---|---|
| Cites the CORRECT section | 0.3% | 1.3% | **+1.0 pp** |
| Names the CORRECT act | 47.7% | 51.0% | **+3.3 pp** |
| Cites a section that exists | 78.3% | 86.7% | +8.3 pp |
| Fabrication rate | 78.0% | 85.3% | +7.3 pp |
| Format valid | 99.7% | 98.0% | — |

n = 300. Peak reserved in bf16: 14.38 GiB
(against ~5.5 GiB for the NF4 arms).

## What this bounds

Quantization is **not** what makes the base model weak on this corpus.

If bf16 scored much higher, A1's weakness would be partly an artifact of
quantization and every "the base model does not know this corpus" claim would be
overstated by that margin. The delta above is that margin, measured rather than
assumed.

_Regenerate: `uv run python -m ragft.eval.run_bf16_reference`_
