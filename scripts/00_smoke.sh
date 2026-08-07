#!/usr/bin/env bash
# Phase 0 gate.
#
# Loads Qwen2.5-7B in 4-bit NF4, attaches LoRA to all seven projections, and
# takes 20 real optimizer steps. Proves the pinned dependency matrix works and
# measures throughput and peak VRAM on this specific GPU.
#
# Nothing else in the project is built on top of these versions until this
# passes, because the conflict it resolves is real and already present:
#
#   bitsandbytes 0.50  requires  transformers<5
#   trl          1.9.x requires  transformers>=4.56.2
#   base env on this machine:    transformers 5.14.1 on Python 3.13  <- outside
#
# Results land in reports/smoke_results.json and reports/env_matrix.md, and
# every GPU-hour estimate in the project plan is recalibrated against them.

set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

STEPS="${STEPS:-20}"
SEQ_LEN="${SEQ_LEN:-2048}"

echo "=== Phase 0 smoke: ${STEPS} steps @ seq_len ${SEQ_LEN} ==="
uv run python -m ragft.train.smoke --steps "$STEPS" --seq-len "$SEQ_LEN"

echo
echo "=== rendering reports/env_matrix.md ==="
uv run python -m ragft.train.env_report
