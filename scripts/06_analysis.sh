#!/usr/bin/env bash
# Phase 6: analysis over results already measured. Mostly CPU.
#
# run_latency is the exception and it wants an EXCLUSIVE GPU window: it samples
# contention at every arm boundary and records whether the window was clean.
# Run it when `nvidia-smi` shows the card idle, or its numbers are noise.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADAPTER="${ADAPTER:-out/seed42_r16_lr0.0002_e3/checkpoint-354}"

uv run python -m ragft.eval.judge_agreement   # CPU
uv run python -m ragft.analysis.failures      # CPU
uv run python -m ragft.eval.run_mmlu     --adapter "$ADAPTER"
uv run python -m ragft.eval.run_latency  --adapter "$ADAPTER"   # wants an idle GPU
uv run python -m ragft.analysis.frontier      # CPU, reads the two above
