#!/usr/bin/env bash
# Regenerate every number in the README, from raw sources.
#
# Ordered by dependency, and deliberately NOT a single `make all`: the phases
# have wildly different costs (minutes of CPU vs hours of GPU) and different
# failure modes, so each stays separately runnable and separately resumable.
#
# Total: ~10 hours wall clock, of which ~4 are GPU. Most of the rest is local
# QA generation on Ollama.
#
#   ./scripts/reproduce_all.sh          # everything
#   ./scripts/reproduce_all.sh --from 3 # resume from phase 3
#
# Every phase is idempotent and safe to re-run. Long phases should go through
# scripts/run.sh so they survive an SSH drop:
#
#   scripts/run.sh dataset ./scripts/01_dataset.sh
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FROM=0
[[ "${1:-}" == "--from" ]] && FROM="${2:-0}"

run_phase() {
    local n="$1" name="$2" script="$3"
    if (( n < FROM )); then
        echo "== phase $n ($name): skipped =="
        return
    fi
    echo
    echo "=============================================================="
    echo "== phase $n: $name"
    echo "=============================================================="
    bash "$script"
}

run_phase 0 "environment smoke test"        scripts/00_smoke.sh
run_phase 1 "corpus + dataset"              scripts/01_dataset.sh
run_phase 2 "retrieval index"               scripts/02_build_index.sh

# The eval set and metric code are frozen here, BEFORE any training. Every
# number downstream is computed against this hash; the runner refuses to
# produce results if the harness no longer matches it.
if (( FROM <= 2 )); then
    uv run python -m ragft.eval.frozen --check
fi

run_phase 3 "baseline arms + go/no-go gate" scripts/03_baselines.sh
run_phase 4 "QLoRA training"                scripts/04_train.sh
run_phase 5 "fine-tuned arms (full 2x2)"    scripts/05_full_grid.sh
run_phase 6 "analysis"                      scripts/06_analysis.sh

# Abstention is only measurable once the hand-written unanswerable stratum
# exists. It cannot be generated -- see ragft.eval.label.write -- so this is
# skipped rather than faked when the file is absent.
if [[ -s data/eval/gold_unanswerable.jsonl ]]; then
    echo "== phase 7: abstention =="
    uv run python -m ragft.eval.run_abstention
else
    echo
    echo "== abstention: SKIPPED -- data/eval/gold_unanswerable.jsonl is empty."
    echo "   Write it with: uv run python -m ragft.eval.label write"
    echo "   Until then abstention is reported as absent, not estimated."
fi

echo
echo "Done. Regenerated reports:"
ls -1 reports/*.md reports/*.json | sed 's/^/  /'
