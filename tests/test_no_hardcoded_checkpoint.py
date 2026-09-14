"""No module may hardcode a checkpoint step number.

`checkpoint-354` is `steps_per_epoch + 1`, and `steps_per_epoch` is a function of
DATASET SIZE: 2,830 train rows at effective batch 8 gives 354, a regenerated
2,752 gives 344. Anything that hardcodes it works on exactly one dataset.

This test exists because the first fix was **incomplete**. The four shell
scripts were corrected and the Python argparse defaults were missed, so
`reproduce_all.sh` -- which calls `run_abstention` with no `--adapter` -- fell
straight back to the stale constant and died after everything upstream had
already succeeded.

A grep is the right shape of test here: the failure is a literal string in a
default, not a behaviour reachable from a unit test.
"""

from __future__ import annotations

import re
from pathlib import Path

from ragft.settings import REPO_ROOT

# checkpoint-<digits> appearing anywhere executable.
PATTERN = re.compile(r"checkpoint-\d+")

# Prose may name it; only code may not depend on it.
ALLOWED = {
    "src/ragft/train/checkpoints.py",  # the resolver, which explains the bug
    "scripts/lib_checkpoint.sh",  # the shell resolver, same
}


def _candidate_files() -> list[Path]:
    files: list[Path] = []
    for pattern in ("src/**/*.py", "scripts/*.sh", "demo.py"):
        files.extend(REPO_ROOT.glob(pattern))
    return sorted(f for f in files if f.is_file())


def test_no_module_hardcodes_a_checkpoint_step() -> None:
    offenders: list[str] = []
    for path in _candidate_files():
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            # Comments and docstring prose may reference it.
            if stripped.startswith("#") or stripped.startswith('"'):
                continue
            if PATTERN.search(line):
                offenders.append(f"{rel}:{lineno}: {stripped[:80]}")

    assert not offenders, (
        "checkpoint step numbers are dataset-dependent and must be resolved at "
        "run time via epoch1_checkpoint(). Hardcoded in:\n  " + "\n  ".join(offenders)
    )
