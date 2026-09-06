#!/usr/bin/env bash
# Wait for a real GPU slot, then run the bf16 reference arm.
#
# An unquantized 7B needs ~16 GiB against ~5.5 for the NF4 arms, so this arm
# cannot share the card the way the others can. Waiting is the correct
# behaviour on a shared box: it takes a slot when one frees rather than
# competing for one that is in use.
#
# Runs under tmux via scripts/run.sh so it survives SSH drops and agent
# sessions ending - an earlier attempt used a plain background job and was
# killed with its session without ever getting a slot.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NEED_MIB="${NEED_MIB:-16800}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-720}"   # 12h

echo "waiting for ${NEED_MIB} MiB free (checking every 2 min, up to ${MAX_WAIT_MIN} min)"
for i in $(seq 1 $((MAX_WAIT_MIN / 2))); do
    free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$free_mib" -ge "$NEED_MIB" ]; then
        echo "[$(date -Is)] GPU free: ${free_mib} MiB - starting bf16 reference arm"
        uv run python -m ragft.eval.run_bf16_reference
        exit $?
    fi
    [ $((i % 15)) -eq 0 ] && echo "[$(date -Is)] still waiting, free=${free_mib} MiB"
    sleep 120
done
echo "TIMED OUT after ${MAX_WAIT_MIN} min; last free=${free_mib} MiB"
exit 1
