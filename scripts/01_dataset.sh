#!/usr/bin/env bash
# Phase 1: corpus -> QA dataset.
#
# The split runs BEFORE generation, and that order is the decontamination
# guarantee. Sections are assigned to train/val/eval first, so a training pair
# and an evaluation question cannot share a source passage by construction.
# The four checks in `decontaminate` verify the guarantee held; they do not
# establish it after the fact, which is a much weaker thing to do.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run python -m ragft.corpus.download        # licence-verified, fails closed
uv run python -m ragft.corpus.parse
uv run python -m ragft.corpus.split           # seeded, BEFORE generation
uv run python -m ragft.dataset.generate       # ~5h on local Ollama, $0
uv run python -m ragft.dataset.filter
uv run python -m ragft.dataset.balance
uv run python -m ragft.dataset.decontaminate
uv run python -m ragft.dataset.stats
