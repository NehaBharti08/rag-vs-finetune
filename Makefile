.PHONY: help install lint fmt type test test-fast smoke check check-clean clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## Create the pinned venv and install pre-commit hooks
	uv sync --frozen
	uv run pre-commit install

lint:  ## ruff
	uv run ruff check src tests scripts

fmt:  ## black
	uv run black src tests

type:  ## mypy (strict, src only)
	uv run mypy

test-fast:  ## Tests that need no GPU and no network
	uv run pytest -m "not gpu and not slow and not costly"

test:  ## Full suite except paid API calls
	uv run pytest -m "not costly"

smoke:  ## Phase 0 gate: 20 QLoRA steps, measures tok/s and peak VRAM
	bash scripts/00_smoke.sh

check: lint type test-fast  ## Everything CI runs

check-clean:  ## Dry-run what git would commit. Run before the FIRST push.
	@echo "=== files git would add (nothing below may be a weight, PDF, dataset, or .env) ==="
	@git add -An --dry-run . | sed 's/^add //' || true
	@echo
	@echo "=== anything over 5MB staged? ==="
	@git add -An --dry-run . 2>/dev/null | sed "s/^add '//;s/'$$//" | \
		while read -r f; do [ -f "$$f" ] && \
		  find "$$f" -size +5M -printf '  !! %s bytes  %p\n' 2>/dev/null; done; \
		echo "  (no output above = clean)"

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
