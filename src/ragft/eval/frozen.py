"""Freeze the evaluation harness so results cannot drift under it.

A metric invented after seeing results is not a metric. This module makes that
rule mechanical rather than aspirational: it hashes everything that determines
an evaluation number — the eval set, the judge prompts, and the metric source —
into ``configs/eval/frozen.lock``.

Two consumers:

* ``ragft.eval.runner`` asserts the hash before running, and refuses to produce
  numbers against a harness that no longer matches the one that was frozen.
* A pre-commit hook runs ``--check``, so editing a judge prompt without
  re-freezing fails the commit instead of silently invalidating every result
  already reported.

Phase 0 ships the mechanism with nothing yet to freeze. Phase 2 populates it
and flips ``frozen`` to true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ragft.settings import CONFIG_DIR, REPO_ROOT

LOCK_PATH = CONFIG_DIR / "eval" / "frozen.lock"

# Everything whose content changes what a metric reports. Ordered, so the
# digest is stable across filesystems.
FROZEN_PATHS: tuple[str, ...] = (
    "data/eval/gold.jsonl",
    "prompts/judge",
    "src/ragft/eval/metrics",
)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_digests(root: Path | None = None) -> dict[str, str]:
    """Map every frozen file to its SHA-256, skipping what does not exist yet."""
    root = root or REPO_ROOT
    digests: dict[str, str] = {}
    for entry in FROZEN_PATHS:
        target = root / entry
        if target.is_file():
            digests[entry] = _hash_file(target)
        elif target.is_dir():
            for f in sorted(target.rglob("*")):
                if f.is_file() and f.suffix in {".py", ".md", ".jsonl", ".yaml"}:
                    digests[str(f.relative_to(root))] = _hash_file(f)
    return digests


def compute_digest(root: Path | None = None) -> str:
    """One digest over the whole harness."""
    digests = collect_digests(root)
    payload = json.dumps(digests, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_lock() -> dict[str, Any]:
    if not LOCK_PATH.exists():
        return {}
    lock: dict[str, Any] = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    return lock


def write_lock(frozen: bool = False) -> dict[str, Any]:
    lock = {
        "frozen": frozen,
        "digest": compute_digest(),
        "files": collect_digests(),
        "note": (
            "Written by ragft.eval.frozen. Once `frozen` is true, any change to "
            "the eval set, judge prompts, or metric code must be a deliberate "
            "re-freeze recorded in a PR - not an edit."
        ),
    }
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    return lock


def check() -> int:
    """Return 0 if the harness matches its lock, 1 otherwise."""
    lock = read_lock()
    if not lock:
        print("frozen.lock absent - nothing frozen yet (expected before Phase 2).")
        return 0
    if not lock.get("frozen", False):
        print("frozen.lock present but not yet frozen (expected before Phase 2).")
        return 0

    current = compute_digest()
    if current == lock["digest"]:
        print(f"eval harness matches frozen.lock ({current[:12]}).")
        return 0

    print("EVAL HARNESS CHANGED after freezing.")
    print(f"  expected {lock['digest'][:12]}")
    print(f"  actual   {current[:12]}")
    was, now = lock.get("files", {}), collect_digests()
    for path in sorted(set(was) | set(now)):
        if was.get(path) != now.get(path):
            state = "added" if path not in was else "removed" if path not in now else "modified"
            print(f"  {state:9s} {path}")
    print(
        "\nA metric invented after seeing results is not a metric. If this change is\n"
        "deliberate, re-freeze it in its own commit and say why in the PR."
    )
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="verify against frozen.lock")
    group.add_argument("--write", action="store_true", help="record current state, unfrozen")
    group.add_argument("--freeze", action="store_true", help="record and FREEZE (Phase 2)")
    args = parser.parse_args()

    if args.check:
        raise SystemExit(check())
    lock = write_lock(frozen=args.freeze)
    print(f"Wrote {LOCK_PATH} (frozen={lock['frozen']}, digest={lock['digest'][:12]})")


if __name__ == "__main__":
    main()
