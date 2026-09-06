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
    uv run python -m ragft.eval.label write     # author unanswerable questions
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
# A SEPARATE file, deliberately not appended to the frozen gold set. Adding
# items to data/eval/gold.jsonl would change its digest and invalidate every
# result already measured against it -- which is exactly what the freeze exists
# to prevent. Abstention is scored by a different metric from the citation
# ladder and shares nothing with it, so it is cleanly a separate evaluation with
# its own freeze.
UNANSWERABLE_PATH = EVAL_DIR / "gold_unanswerable.jsonl"

TARGET_UNANSWERABLE = 60

# The design note says an LLM asked for unanswerable questions produces
# obviously out-of-domain ones, which makes abstention look easy. The same is
# true of a tired human at question 40. These categories exist to keep the set
# HARD: every one of them is a question a competent lawyer might actually ask,
# which is the only kind worth testing refusal on.
UNANSWERABLE_KINDS: tuple[tuple[str, str, str], ...] = (
    (
        "false_presupposition",
        "Presumes a provision that does not exist",
        "Under BNS \u00a7420, what is the punishment for cheating?"
        "   (\u00a7420 is the OLD IPC number; BNS numbers it differently)",
    ),
    (
        "out_of_corpus_act",
        "Real Indian law, but an Act not in this corpus",
        "What is the penalty for driving without a valid licence?"
        "   (Motor Vehicles Act -- not one of the four)",
    ),
    (
        "repealed_law",
        "Asks about IPC / CrPC / Evidence Act by their old section numbers",
        "What does Section 302 IPC prescribe?"
        "   (repealed; the corpus holds only the 2023 successors)",
    ),
    (
        "beyond_the_text",
        "Something bare statute text simply does not contain",
        "How have the High Courts interpreted the new evidence provisions?"
        "   (case law, not statute)",
    ),
    (
        "underspecified",
        "Cannot be answered as posed, even in principle",
        "What is the punishment under the Sanhita?" "   (which offence? the question names none)",
    ),
)

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


def _read_line(prompt: str) -> str:
    """Line input. Authoring needs full editing, unlike the one-key modes."""
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def write_unanswerable() -> None:
    """Author the hand-written unanswerable stratum.

    This stratum cannot be generated. An LLM asked for unanswerable questions
    produces obviously out-of-domain ones, so a model scores near-perfect
    abstention on them and the metric reports nothing. The questions have to be
    near-misses -- plausible, on-topic, and answerable-looking -- and that is a
    human judgement.
    """
    done = (
        [json.loads(line) for line in UNANSWERABLE_PATH.open(encoding="utf-8") if line.strip()]
        if UNANSWERABLE_PATH.exists()
        else []
    )
    seen = {r["question"].strip().lower() for r in done}
    by_kind: dict[str, int] = {}
    for r in done:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1

    per_kind = TARGET_UNANSWERABLE // len(UNANSWERABLE_KINDS)

    print(f"\n{BOLD}Writing unanswerable eval questions{RESET}")
    print(
        wrap(
            "These are questions the model SHOULD refuse. They must look "
            "answerable and stay on-topic -- an obviously off-topic question "
            "makes refusal trivial and measures nothing. Aim for near-misses.",
            indent="",
        )
    )
    print(f"\n  target {TARGET_UNANSWERABLE} ({per_kind} per category), have {len(done)}")
    print(f"  saved to {UNANSWERABLE_PATH.relative_to(REPO_ROOT)} after every entry")
    print(f"  {DIM}blank line or 'q' to stop -- resume any time{RESET}\n")

    while len(done) < TARGET_UNANSWERABLE:
        # Always offer the category furthest from its quota, so the mix stays
        # balanced even if the session is stopped early.
        kind, desc, example = min(UNANSWERABLE_KINDS, key=lambda k: by_kind.get(k[0], 0))
        have = by_kind.get(kind, 0)

        print(f"{DIM}{'-' * 78}{RESET}")
        print(
            f"{BOLD}[{len(done) + 1}/{TARGET_UNANSWERABLE}]  {kind}{RESET}  {DIM}({have}/{per_kind}){RESET}"
        )
        print(f"  {desc}")
        print(f"  {DIM}e.g. {example}{RESET}\n")

        question = _read_line("  question> ")
        if question.lower() in {"q", "quit", ""}:
            break
        if not question.endswith("?"):
            print(f"  {YELLOW}must end with '?'{RESET}\n")
            continue
        if len(question) < 25:
            print(f"  {YELLOW}too short to be a plausible near-miss{RESET}\n")
            continue
        if question.strip().lower() in seen:
            print(f"  {YELLOW}already written{RESET}\n")
            continue

        why = _read_line("  why unanswerable> ")
        if why.lower() in {"q", "quit"}:
            break
        if len(why) < 10:
            print(f"  {YELLOW}give a real reason -- it is the audit trail{RESET}\n")
            continue

        # CONFIRM the category rather than assuming the offered one was used.
        #
        # The first version recorded whichever kind it had *offered*, on the
        # assumption that a writer answers the prompt in front of them. In
        # practice the first 60-item set was written in the author's own order,
        # cycling through all five types, and 48 of 60 ended up mislabelled --
        # a balanced set with scrambled labels, which silently makes the
        # by-kind breakdown meaningless. Ask; never infer.
        print(f"  {DIM}which kind is this really?{RESET}")
        for n, (k, d, _) in enumerate(UNANSWERABLE_KINDS, 1):
            mark = " <- offered" if k == kind else ""
            print(f"    {n}. {k:22s} {DIM}{d}{mark}{RESET}")
        choice = _read_line(f"  kind [1-{len(UNANSWERABLE_KINDS)}, Enter = {kind}]> ")
        if choice.lower() in {"q", "quit"}:
            break
        if choice.isdigit() and 1 <= int(choice) <= len(UNANSWERABLE_KINDS):
            kind = UNANSWERABLE_KINDS[int(choice) - 1][0]

        record: dict[str, Any] = {
            "gold_id": _gold_id_for(question),
            "question": question,
            # No reference answer exists, by construction. The correct
            # behaviour is refusal, so `reference` holds the refusal string
            # rather than an answer, and metrics never score it for content.
            "reference": None,
            "stratum": "unanswerable",
            "kind": kind,
            "why_unanswerable": why,
            "source_section_ids": [],
            "source_chunk_sha256": [],
            "act_slug": None,
            "citation": None,
            "generator_model": "human",
            "unanswerable": True,
            "parametric_answerable": False,
            "human_verified": True,
        }
        append(UNANSWERABLE_PATH, record)
        done.append(record)
        seen.add(question.strip().lower())
        by_kind[kind] = have + 1
        print(f"  {GREEN}saved{RESET}\n")

    print(f"\n{BOLD}{len(done)}/{TARGET_UNANSWERABLE}{RESET} written.")
    if len(done) >= TARGET_UNANSWERABLE:
        print(f"{GREEN}Complete.{RESET} Next: uv run python -m ragft.eval.run_abstention")
    else:
        print(f"Resume with {DIM}uv run python -m ragft.eval.label write{RESET}")


def _gold_id_for(question: str) -> str:
    import hashlib

    return hashlib.sha256(question.strip().lower().encode("utf-8")).hexdigest()[:16]


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
    parser.add_argument("mode", choices=["verify", "judge", "write", "status"])
    args = parser.parse_args()
    {"verify": verify, "judge": judge, "write": write_unanswerable, "status": status}[args.mode]()


if __name__ == "__main__":
    main()
