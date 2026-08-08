"""Keyboard-only, resumable labelling CLI.

Two jobs need a human, and neither can be honestly automated:

1. **Verifying gold items.** Each candidate is checked for whether it reads
   naturally, whether the reference answer is actually right, and whether it is
   answerable at all.
2. **Judge agreement.** Grading model responses by hand so Cohen's kappa
   against the judge can be reported. Without this, "the judge is trustworthy"
   is an assumption doing real work in the headline metric -- and on the free
   path the judge is a local 9.6 GB model.

Design constraints, because ~160 decisions is enough that friction matters:

* One keystroke per decision, no Enter.
* Saves after every single answer. Stop with `q` at any point and resume
  exactly where you left off; nothing is lost if the terminal dies.
* Shows progress and lets you skip anything you are unsure about, rather than
  forcing a guess -- a forced guess is worse data than an absent label.

Usage::

    uv run python -m ragft.eval.label verify    # check gold candidates
    uv run python -m ragft.eval.label judge     # grade responses for kappa
    uv run python -m ragft.eval.label status
"""

from __future__ import annotations

import argparse
import json
import sys
import termios
import tty
from pathlib import Path
from typing import Any

from ragft.settings import REPO_ROOT

EVAL_DIR = REPO_ROOT / "data" / "eval"
VERIFY_PATH = EVAL_DIR / "human_verification.jsonl"
JUDGE_PATH = EVAL_DIR / "human_judge_labels.jsonl"

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[32m",
    "\033[31m",
    "\033[33m",
    "\033[0m",
)


def read_key() -> str:
    """Read a single keypress without waiting for Enter."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch.lower()


def load_done(path: Path, key: str) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                done.add(str(json.loads(line)[key]))
    return done


def append(path: Path, record: dict[str, Any]) -> None:
    """Append and flush immediately, so a crash never costs a decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        fh.flush()


def wrap(text: str, width: int = 88, indent: str = "  ") -> str:
    import textwrap

    return "\n".join(
        textwrap.fill(p, width, initial_indent=indent, subsequent_indent=indent)
        for p in text.split("\n")
        if p.strip()
    )


def verify() -> None:
    """Check gold candidates: keep, drop, or flag."""
    from ragft.eval.goldset import load_candidates

    items = load_candidates()
    done = load_done(VERIFY_PATH, "gold_id")
    todo = [i for i in items if i.gold_id not in done]

    print(f"\n{BOLD}Verify gold candidates{RESET}")
    print(f"{len(done)} done, {len(todo)} remaining of {len(items)}\n")
    print(f"{DIM}[k]eep  [d]rop  [f]lag unsure  [q]uit (progress is saved){RESET}\n")

    for n, item in enumerate(todo, 1):
        print(f"{BOLD}{'─' * 90}{RESET}")
        print(f"{DIM}{n}/{len(todo)}  {item.stratum}  {item.citation}{RESET}\n")
        print(f"{BOLD}Q:{RESET}\n{wrap(item.question)}\n")
        print(f"{BOLD}Reference:{RESET}\n{wrap(item.reference)}\n")
        print(f"{DIM}k=keep  d=drop  f=flag  q=quit{RESET}  ", end="", flush=True)

        while (key := read_key()) not in "kdfq":
            pass
        print(key)
        if key == "q":
            print(f"\n{YELLOW}Stopped. {len(done) + n - 1} labelled. Rerun to resume.{RESET}\n")
            return

        append(
            VERIFY_PATH,
            {
                "gold_id": item.gold_id,
                "verdict": {"k": "keep", "d": "drop", "f": "flag"}[key],
                "question": item.question,
                "stratum": item.stratum,
            },
        )
        print()

    print(f"\n{GREEN}All {len(items)} candidates verified.{RESET}\n")


def judge() -> None:
    """Grade model responses by hand, for Cohen's kappa against the judge."""
    path = EVAL_DIR / "judge_sample.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} missing. It is written by the eval runner once the baseline "
            "arms have produced responses."
        )
    with path.open(encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]

    done = load_done(JUDGE_PATH, "item_id")
    todo = [i for i in items if str(i["item_id"]) not in done]

    print(f"\n{BOLD}Grade responses (for judge agreement){RESET}")
    print(f"{len(done)} done, {len(todo)} remaining of {len(items)}\n")
    print(f"{DIM}[2] correct  [1] partial  [0] incorrect  [s]kip  [q]uit{RESET}")
    print(f"{DIM}Judge CONTENT, not style or length. A fluent wrong answer is 0.{RESET}\n")

    for n, item in enumerate(todo, 1):
        print(f"{BOLD}{'─' * 90}{RESET}")
        print(f"{DIM}{n}/{len(todo)}{RESET}\n")
        print(f"{BOLD}Q:{RESET}\n{wrap(item['question'])}\n")
        print(f"{GREEN}Reference:{RESET}\n{wrap(item['reference'])}\n")
        print(f"{YELLOW}Model answer:{RESET}\n{wrap(item['response'])}\n")
        print(f"{DIM}2=correct 1=partial 0=incorrect s=skip q=quit{RESET}  ", end="", flush=True)

        while (key := read_key()) not in "210sq":
            pass
        print(key)
        if key == "q":
            print(f"\n{YELLOW}Stopped. {len(done) + n - 1} graded. Rerun to resume.{RESET}\n")
            return
        if key == "s":
            print(f"{DIM}skipped{RESET}\n")
            continue

        append(
            JUDGE_PATH,
            {
                "item_id": item["item_id"],
                "human_score": int(key),
                "arm": item.get("arm"),
            },
        )
        print()

    print(f"\n{GREEN}All {len(items)} responses graded.{RESET}\n")


def status() -> None:
    print(f"\n{BOLD}Labelling status{RESET}\n")
    for name, path, key in (
        ("gold verification", VERIFY_PATH, "gold_id"),
        ("judge grading", JUDGE_PATH, "item_id"),
    ):
        n = len(load_done(path, key))
        print(f"  {name:20s} {n:4d} labelled   {DIM}{path.relative_to(REPO_ROOT)}{RESET}")

    if VERIFY_PATH.exists():
        from collections import Counter

        with VERIFY_PATH.open(encoding="utf-8") as fh:
            verdicts = Counter(json.loads(line)["verdict"] for line in fh if line.strip())
        print(f"\n  verdicts: {dict(verdicts)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["verify", "judge", "status"])
    args = parser.parse_args()
    {"verify": verify, "judge": judge, "status": status}[args.mode]()


if __name__ == "__main__":
    main()
