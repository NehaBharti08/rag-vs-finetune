#!/usr/bin/env bash
# Launch a long job in a detached tmux session with a timestamped log.
#
# There is no SLURM on this machine — verified: no sbatch, no squeue, no module
# system, no cluster in ~/.ssh/config. What exists is one shared RTX A5000. So
# this is the job runner: tmux keeps a run alive across SSH disconnects, and
# every runner it launches must be able to resume from its own checkpoints,
# because a shared GPU means a run can lose its slot at any time.
#
# Usage:
#   scripts/run.sh <name> <command...>
#
# Examples:
#   scripts/run.sh smoke uv run python -m ragft.train.smoke --steps 20
#   scripts/run.sh sft   uv run python -m ragft.train.sft --config configs/train/base.yaml
#
# Then:
#   tmux attach -t ragft-smoke      # watch it
#   tail -f logs/smoke-<stamp>.log  # or just read the log
#   scripts/run.sh --list           # what is running

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"

if [[ "${1:-}" == "--list" ]]; then
    tmux ls 2>/dev/null | grep '^ragft-' || echo "no ragft sessions running"
    exit 0
fi

if [[ $# -lt 2 ]]; then
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
fi

NAME="$1"; shift
SESSION="ragft-${NAME}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/${NAME}-${STAMP}.log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: session '$SESSION' already exists." >&2
    echo "  attach: tmux attach -t $SESSION" >&2
    echo "  kill:   tmux kill-session -t $SESSION" >&2
    # Refusing rather than clobbering: silently starting a second training run
    # on a 24GB shared GPU would OOM both of them.
    exit 1
fi

# GPU contention is a fact of life on this box, not an error. Report it so the
# log records what the card looked like when the run started — latency numbers
# are uninterpretable without it.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU state at launch:"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu \
               --format=csv,noheader | sed 's/^/  /'
fi

# `exec` is deliberately absent: the shell stays alive after the command exits
# so a crashed run leaves its tmux pane readable instead of vanishing.
tmux new-session -d -s "$SESSION" -c "$REPO_ROOT" \
    "set -o pipefail; { echo '=== $SESSION started '\$(date -Is)' ==='; $*; echo \"=== exit=\$? at \$(date -Is) ===\"; } 2>&1 | tee '$LOG'; echo; echo '[session idle — ctrl-b d to detach, exit to close]'; bash"

echo "started : $SESSION"
echo "log     : $LOG"
echo "attach  : tmux attach -t $SESSION"
