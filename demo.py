"""Ask one question and see all four arms answer it, side by side.

This is the demo, and it is deliberately a local script rather than a hosted
Space. A free HF Space gets a CPU and 16 GB of RAM, which cannot serve a 4-bit
7B model. Shipping a Space that quietly falls back to something smaller would
demo a different model than the one this project measured, so the honest option
is a local script that runs the real thing plus a results explorer over the
committed eval logs.

The point of showing four answers at once is that the interesting failure is
invisible in any single one. All four arms produce a confident, correctly
formatted citation. Only by comparing them against the corpus -- which the
script does for you -- can you see that the fine-tuned no-retrieval arm is
almost always naming the right statute and the wrong section.

Usage::

    uv run python demo.py "What punishment does the law prescribe for murder?"
    uv run python demo.py --interactive
    uv run python demo.py --list-failures      # no GPU needed, reads eval logs
"""

from __future__ import annotations

import argparse
import json
import sys

from ragft.settings import REPO_ROOT

BOLD, DIM, GREEN, RED, YELLOW, RESET = (
    "\033[1m",
    "\033[2m",
    "\033[31;1m".replace("31", "32"),
    "\033[31m",
    "\033[33m",
    "\033[0m",
)

DEFAULT_ADAPTER = "out/seed42_r16_lr0.0002_e3/checkpoint-354"


def show_failures(n: int = 5) -> None:
    """Browse real A3 failures from the committed logs. No GPU, no model load."""
    path = REPO_ROOT / "data" / "eval" / "responses" / "A3_ft_zeroshot.jsonl"
    if not path.exists():
        sys.exit(f"{path} not found -- run the eval first, or clone with logs.")

    from ragft.corpus.toc import registry

    reg = registry()
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]

    shown = 0
    for r in rows:
        gold = set(r["source_section_ids"])
        check = reg.validate(r["response"])
        if not check.act_slug or check.section_id in gold:
            continue
        right_act = check.act_slug in {s.split(":")[0] for s in gold}
        if not right_act:
            continue

        print(f"\n{DIM}{'-' * 76}{RESET}")
        print(f"{BOLD}Q.{RESET} {r['question']}")
        print(f"{BOLD}Gold.{RESET} {', '.join(sorted(gold))}")
        print(
            f"{BOLD}Cited.{RESET} {RED}{check.raw}{RESET}  "
            f"{DIM}<- right statute, wrong section{RESET}"
        )
        shown += 1
        if shown >= n:
            break

    print(
        f"\n{YELLOW}This is the project's central finding.{RESET} The fine-tuned model "
        f"names the correct\nstatute and the wrong section in 90.3% of its answers. "
        f"Every citation above is\nwell-formed, confident, and wrong in the way that is "
        f"hardest to catch.\n"
    )


def ask(question: str, adapter: str | None) -> None:
    from ragft.corpus.toc import registry
    from ragft.eval.arms import ArmRunner, arm_specs, build_prompt

    reg = registry()
    runner = ArmRunner()
    specs = arm_specs(adapter)
    ret = None

    print(f"\n{BOLD}Q.{RESET} {question}\n")

    for arm in specs:
        runner.set_adapter(arm.adapter_path)
        context = ""
        if arm.use_retrieval:
            if ret is None:
                from ragft.retrieval.retriever import retriever

                ret = retriever()
            context = ret.format_context(ret.retrieve(question))

        out = runner.generate(build_prompt(arm, question, context))
        check = reg.validate(out["response"])

        print(f"{DIM}{'=' * 76}{RESET}")
        print(f"{BOLD}{arm.name}{RESET}  {DIM}({arm.label}, {out['latency_seconds']:.1f}s){RESET}")
        print(out["response"].strip())

        # The citation is the part worth checking mechanically, so check it.
        if check.act_slug is None:
            verdict = f"{RED}no recognisable act cited{RESET}"
        elif check.section_id:
            verdict = f"{GREEN}section {check.cited_section} exists{RESET}"
        else:
            verdict = (
                f"{RED}section {check.cited_section} does NOT exist in {check.act_slug}{RESET}"
            )
        print(f"\n  {DIM}citation check:{RESET} {verdict}")

    print(
        f"\n{DIM}Whether a citation is CORRECT (not merely real) needs the gold "
        f"section,\nso it is only scored on the frozen eval set. See reports/.{RESET}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", help="a question about Indian statutory law")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument(
        "--list-failures",
        action="store_true",
        help="browse real failures from the eval logs (no GPU needed)",
    )
    args = parser.parse_args()

    if args.list_failures:
        show_failures()
        return

    adapter = args.adapter if (REPO_ROOT / args.adapter).exists() else None
    if adapter is None:
        print(f"{YELLOW}No adapter at {args.adapter} -- running base arms only.{RESET}")

    if args.interactive:
        print(f"{DIM}Ctrl-D or 'q' to quit.{RESET}")
        while True:
            try:
                q = input("\nquestion> ").strip()
            except EOFError:
                break
            if q.lower() in {"q", "quit", ""}:
                break
            ask(q, adapter)
        return

    if not args.question:
        parser.error("give a question, or use --interactive / --list-failures")
    ask(args.question, adapter)


if __name__ == "__main__":
    main()
