#!/usr/bin/env bash
# Resolve the EPOCH-1 checkpoint of a training run.
#
# Every arm, sweep and analysis script needs this adapter, and they all used to
# hardcode `checkpoint-354`. That number is steps_per_epoch + 1, which is a
# function of the DATASET SIZE: 2,830 train rows at effective batch 8 gives 354,
# but a regenerated dataset of 2,752 rows gives 344. The hardcode worked on
# exactly one dataset and broke on any other, which an end-to-end reproduction
# hit immediately:
#
#     ValueError: Can't find 'adapter_config.json' at '.../checkpoint-354'
#
# Epoch 1 is wanted because validation loss rises from epoch 1 on every seed
# measured, so the first checkpoint is the best one. Sorting numerically and
# taking the lowest step count is exactly "end of the first epoch" whatever the
# dataset size.
epoch1_checkpoint() {
    local run_dir="$1"
    local ckpt
    ckpt=$(find "$run_dir" -maxdepth 1 -name 'checkpoint-*' -type d 2>/dev/null \
           | sed 's/.*checkpoint-//' | sort -n | head -1)
    if [[ -n "$ckpt" ]]; then
        echo "${run_dir}/checkpoint-${ckpt}"
        return 0
    fi
    # A completed run also saves a final adapter/ dir; fall back to it so a
    # checkpoint-less run is still usable rather than a hard failure.
    if [[ -d "${run_dir}/adapter" ]]; then
        echo "${run_dir}/adapter"
        return 0
    fi
    echo "no checkpoint or adapter under ${run_dir}" >&2
    return 1
}
