# Compute log

Every GPU-hour spent, and the decisions that spent or saved it. This box has
one shared RTX A5000, so waste is other people's queue time.

## Budget

Planning estimates are superseded by measured throughput in
[`../reports/env_matrix.md`](../reports/env_matrix.md) as soon as Phase 0 runs.

| Item | Planned | Actual |
|---|---|---|
| Phase 0 smoke | ~0.5 h | _see env_matrix_ |
| Sweep (4 configs x 1 epoch) | ~6 h | _pending_ |
| Final (3 seeds x 3 epochs) | ~13 h | _pending_ |
| All evaluation | ~3 h | _pending_ |
| Latency measurement | ~1 h | _pending_ |
| **Total** | **~24 h** | _pending_ |

## Savings taken

- **Epoch axis collapsed.** Checkpointing every epoch reads epochs 1/2/3 out of
  one run instead of one run per epoch count: 12 runs -> 4.
- **Sweep shortened.** Configs ranked at 1 epoch, not 3: ~5 h -> ~1.5 h each.

Together these are the difference between a ~75 GPU-hour plan and a ~24 GPU-hour
one.

## Savings deliberately NOT taken

- **3 seeds kept.** ~13 h is real on a shared box, but variance reporting is the
  cheapest credibility available and single-run numbers are this genre's most
  common failure.

## Gate decisions

### Phase 3 go/no-go

_Pending._ If the zero-shot base arm scores near ceiling on the primary metric,
the corpus is not novel to the model, every cell compresses, and Phase 4 buys
nothing. Decision and numbers recorded here before any training compute is spent.
