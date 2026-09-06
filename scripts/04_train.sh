#!/usr/bin/env bash
# Phase 4: QLoRA training. ~45 min on one A5000.
#
# Checkpoints every epoch, which is what makes the epoch axis free: epochs
# 1/2/3 are read out of one run rather than launching a run per epoch count.
# It is also what makes the run resumable on a shared GPU that can lose its
# slot mid-run.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run python -m ragft.train.sft --seed "${SEED:-42}"
