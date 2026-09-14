"""No module may hardcode a checkpoint step number.

`checkpoint-354` is `steps_per_epoch + 1`, and `steps_per_epoch` is a function of
DATASET SIZE: 2,830 train rows at effective batch 8 gives 354, a regenerated
2,752 gives 344. Anything that hardcodes it works on exactly one dataset.

This exists because the first fix was **incomplete**. Four shell scripts were
corrected and the Python argparse defaults were missed, so `reproduce_all.sh` --
which calls `run_abstention` with no `--adapter` -- fell back to the stale
constant and died after everything upstream had already succeeded.

Python is checked by AST rather than by grep, deliberately. A line-based grep
gets this wrong in both directions: it flags prose inside docstrings, and it
MISSES `"path": ".../checkpoint-354"` because that line begins with a quote.
The first version of this test did exactly that, passing over a real hardcode in
`report_arms.py` while failing on a sentence in `frontier.py`. Only string
constants that are not docstrings can carry the bug.
"""

from __future__ import annotations

import ast
import re

from ragft.settings import REPO_ROOT

PATTERN = re.compile(r"checkpoint-\d+")

ALLOWED = {
    "src/ragft/train/checkpoints.py",  # the resolver, which explains the bug
    "scripts/lib_checkpoint.sh",  # the shell resolver, same
}


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids() of string constants that are docstrings, which may name it."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                out.add(id(body[0].value))
    return out


def test_python_modules_do_not_hardcode_a_checkpoint_step() -> None:
    offenders: list[str] = []
    for path in sorted([*REPO_ROOT.glob("src/**/*.py"), REPO_ROOT / "demo.py"]):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWED or not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=rel)
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and PATTERN.search(node.value)
            ):
                offenders.append(f"{rel}:{node.lineno}: {node.value[:70]!r}")

    assert not offenders, (
        "checkpoint step numbers are dataset-dependent and must be resolved at "
        "run time via epoch1_checkpoint(). Hardcoded in:\n  " + "\n  ".join(offenders)
    )


def test_shell_scripts_do_not_hardcode_a_checkpoint_step() -> None:
    offenders: list[str] = []
    for path in sorted(REPO_ROOT.glob("scripts/*.sh")):
        rel = str(path.relative_to(REPO_ROOT))
        if rel in ALLOWED:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if PATTERN.search(code):
                offenders.append(f"{rel}:{lineno}: {line.strip()[:70]}")
    assert not offenders, "hardcoded checkpoint in shell:\n  " + "\n  ".join(offenders)
