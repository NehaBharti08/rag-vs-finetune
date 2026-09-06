#!/usr/bin/env bash
# Phase 4 sweep, run late and deliberately. ~2.5h.
#
# Waits for the bf16 reference arm to finish first: both want the card, and two
# waiters racing for one slot is how you get an OOM that kills both. Then waits
# for VRAM headroom of its own.
#
# Four configs x 1 epoch, evaluated on a 100-item subset of the frozen gold set.
# The sweep is a RANKING problem, not a measurement one - it needs to tell four
# configs apart, and every headline number comes from the full 300 regardless.
#
# What it can settle: whether "THIS QLoRA configuration failed" can be shortened
# to "QLoRA failed here". See configs/train/sweep.yaml and reports/sweep.md.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NEED_MIB="${NEED_MIB:-12500}"
EVAL_ITEMS="${EVAL_ITEMS:-100}"

# 1. Do not compete with the bf16 arm.
if tmux has-session -t ragft-bf16ref 2>/dev/null; then
    echo "waiting for ragft-bf16ref to finish first..."
    while tmux has-session -t ragft-bf16ref 2>/dev/null; do sleep 60; done
    echo "bf16ref done"
fi

wait_for_vram() {
    for i in $(seq 1 360); do   # up to 12h
        free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
        [ "$free_mib" -ge "$NEED_MIB" ] && return 0
        [ $((i % 15)) -eq 0 ] && echo "[$(date -Is)] waiting for ${NEED_MIB} MiB, free=${free_mib}"
        sleep 120
    done
    echo "TIMED OUT waiting for VRAM"; return 1
}

run_config() {
    local name="$1" rank="$2" lr="$3"
    local run="sweep_${name}"
    echo
    echo "=============================================================="
    echo "== ${name}: rank=${rank} lr=${lr}"
    echo "=============================================================="
    wait_for_vram || exit 1

    if [ -d "out/${run}/checkpoint-354" ] || [ -d "out/${run}/adapter" ]; then
        echo "already trained, skipping"
    else
        uv run python -m ragft.train.sft --seed 42 --epochs 1 \
            --rank "$rank" --lr "$lr" --run-name "$run" || exit 1
    fi

    local adapter
    adapter=$(ls -d "out/${run}"/checkpoint-* 2>/dev/null | head -1)
    [ -z "$adapter" ] && adapter="out/${run}/adapter"

    uv run python -m ragft.eval.runner --arms A3_ft_zeroshot \
        --adapter "$adapter" --tag "${run}" --limit "$EVAL_ITEMS" || exit 1
}

run_config r16_lr2e-4 16 2e-4
run_config r16_lr1e-4 16 1e-4
run_config r64_lr2e-4 64 2e-4
run_config r64_lr1e-4 64 1e-4

uv run python -m ragft.analysis.sweep
