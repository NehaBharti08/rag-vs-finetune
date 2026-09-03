#!/usr/bin/env bash
# Phase 3: the two baseline arms, then the go/no-go gate.
#
# A1 (base, no retrieval) is run first because it does three jobs at once:
# it is one cell of the 2x2, it is the only way to measure
# `parametric_answerable`, and its responses are the sample a human grades
# for judge agreement. A2 (base + RAG) follows.
#
# The gate itself is the answerability rate. On the biology corpus it came
# back 75.7% - the base model already knew the material, every cell
# compressed, and the grid had little room to move. That measurement is what
# this corpus was chosen to change.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

uv run python -m ragft.eval.runner --arms A1_base_zeroshot
uv run python -m ragft.eval.runner --arms A2_base_rag
uv run python -m ragft.eval.answerability
uv run python -m ragft.eval.report_baseline
uv run python -m ragft.eval.report_arms
