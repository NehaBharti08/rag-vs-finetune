#!/usr/bin/env bash
# Phase 5: the fine-tuned arms, completing the 2x2.
#
# Uses the EPOCH-1 checkpoint, not the final adapter. Validation loss rose
# monotonically (0.887 -> 0.919 -> 1.059) while train loss fell, so the model
# overfits from epoch 1 onward on 2,830 short examples. Checkpointing every
# epoch is what makes selecting epoch 1 free rather than another training run.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER="${ADAPTER:-out/seed42_r16_lr0.0002_e3/checkpoint-354}"
echo "adapter: $ADAPTER"
uv run python -m ragft.eval.runner --arms A3_ft_zeroshot --adapter "$ADAPTER"
uv run python -m ragft.eval.runner --arms A4_ft_rag --adapter "$ADAPTER"
uv run python -m ragft.eval.report_arms
