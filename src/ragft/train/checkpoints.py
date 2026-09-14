"""Resolve a training run's epoch-1 checkpoint.

The Python counterpart of `scripts/lib_checkpoint.sh`, and it exists for the
same reason: `checkpoint-354` was hardcoded as an argparse default in four eval
modules. That number is `steps_per_epoch + 1`, a function of DATASET SIZE --
2,830 train rows at effective batch 8 gives 354, a regenerated 2,752 gives 344.

The shell scripts were fixed first and that fix was INCOMPLETE: the modules keep
their own defaults, and `reproduce_all.sh` invokes `run_abstention` with no
`--adapter`, so it fell straight back to the stale constant and died:

    ValueError: Can't find 'adapter_config.json' at '.../checkpoint-354'

Epoch 1 is the target because validation loss rises from epoch 1 on every seed
and every dataset measured, so the first checkpoint is the best one.
"""

from __future__ import annotations

import re
from pathlib import Path

from ragft.settings import REPO_ROOT

DEFAULT_RUN = REPO_ROOT / "out" / "seed42_r16_lr0.0002_e3"


def epoch1_checkpoint(run_dir: Path | str = DEFAULT_RUN) -> str:
    """Path to the end-of-first-epoch checkpoint, as a string for PEFT.

    Sorts checkpoints NUMERICALLY. Lexically "checkpoint-1032" precedes
    "checkpoint-344", so a naive sort returns the most overfit adapter instead
    of the least -- and does so silently.
    """
    run = Path(run_dir)
    steps = sorted(
        int(m.group(1))
        for d in run.glob("checkpoint-*")
        if d.is_dir() and (m := re.fullmatch(r"checkpoint-(\d+)", d.name))
    )
    if steps:
        return str(run / f"checkpoint-{steps[0]}")
    # A finished run also writes adapter/; prefer a usable path over a hard fail.
    if (run / "adapter").is_dir():
        return str(run / "adapter")
    raise SystemExit(
        f"No checkpoint-* or adapter/ under {run}. Train first:\n"
        f"    uv run python -m ragft.train.sft --seed 42"
    )
