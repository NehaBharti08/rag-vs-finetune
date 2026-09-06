#!/usr/bin/env bash
# Three-seed variance run. ~3h on one A5000.
#
# Every number in this repo is currently a single seed, which the plan calls
# "this genre's most common credibility failure". A4 beats A2 by 4.6 points and
# that difference is NOT established without a variance estimate.
#
# Seed 42 is already trained and evaluated, so only 1 and 2 are run here. Their
# responses are namespaced with --tag: without it the resume check finds all 300
# gold_ids already present from seed 42, runs nothing, and reports success while
# the numbers stay seed 42's.
#
# Only the FINE-TUNED arms vary with seed. A1 and A2 use no adapter, so
# re-running them per seed would burn ~40 GPU-minutes each to reproduce
# identical numbers under greedy decoding.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="${SEEDS:-1 2}"

for seed in $SEEDS; do
    RUN="seed${seed}_r16_lr0.0002_e3"
    ADAPTER="out/${RUN}/checkpoint-354"   # epoch 1, matching seed 42

    echo
    echo "=============================================================="
    echo "== seed ${seed}: train"
    echo "=============================================================="
    if [[ -d "out/${RUN}/checkpoint-354" ]]; then
        echo "already trained, skipping"
    else
        uv run python -m ragft.train.sft --seed "$seed"
    fi

    echo
    echo "== seed ${seed}: evaluate A3, A4"
    uv run python -m ragft.eval.runner --arms A3_ft_zeroshot --adapter "$ADAPTER" --tag "seed${seed}"
    uv run python -m ragft.eval.runner --arms A4_ft_rag      --adapter "$ADAPTER" --tag "seed${seed}"
done

uv run python -m ragft.analysis.seeds
