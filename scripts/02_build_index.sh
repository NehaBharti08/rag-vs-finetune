#!/usr/bin/env bash
# Build the vector index and immediately measure retrieval recall.
#
# Recall is measured in the same job on purpose: an index that exists but was
# never checked is how a broken retriever reaches a results table. Threat 7.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
uv run python -m ragft.retrieval.index --force
uv run python -m ragft.retrieval.recall --split val
