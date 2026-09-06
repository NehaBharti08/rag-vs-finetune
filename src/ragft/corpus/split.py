"""Assign every section to exactly one split, before any generation happens.

Order matters more than the algorithm here. Splitting at the **section** level
**before** generating any QA means decontamination is guaranteed by
construction: a training pair and an eval question cannot derive from the same
passage, because the passage itself lives in only one split. The four checks in
``ragft.dataset.decontaminate`` then *verify* that guarantee rather than trying
to establish it after the fact by filtering.

The alternative -- generate everything, then filter out overlaps -- is what most
projects do, and it is strictly weaker: near-duplicate detection has a
threshold, and whatever slips under it is silent contamination.

Stratified by ACT so every statute appears in every split. An eval set drawn
mostly from one act would confound the adaptation result with the difference
between, say, criminal procedure and contract law -- and would also destroy the
enactment-recency stratification, since three acts are from 2023 and one from
1872.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from enum import StrEnum
from typing import Any

from ragft.corpus.parse import Section, load_sections
from ragft.settings import REPO_ROOT

SPLIT_PATH = REPO_ROOT / "data" / "corpus" / "splits.json"

# Fixed and committed. Re-running the split must reproduce it exactly, or the
# decontamination guarantee is only as good as someone's memory of a seed.
SPLIT_SEED = 20260807


class Split(StrEnum):
    TRAIN = "train"
    VAL = "val"
    EVAL_UNSEEN = "eval_unseen"


# Train is large because it feeds ~4k generated pairs; val only has to be big
# enough to see overfitting; eval_unseen only has to support ~180 questions.
RATIOS: dict[Split, float] = {
    Split.TRAIN: 0.80,
    Split.VAL: 0.10,
    Split.EVAL_UNSEEN: 0.10,
}


def assign(sections: list[Section], seed: int = SPLIT_SEED) -> dict[str, str]:
    """Map section_id -> split, stratified by book, deterministic given seed."""
    rng = random.Random(seed)
    by_act: dict[str, list[Section]] = defaultdict(list)
    for s in sections:
        by_act[s.act_slug].append(s)

    assignment: dict[str, str] = {}
    for slug in sorted(by_act):
        # Sort before shuffling so the result never depends on the order
        # sections happened to come out of the parser.
        # Natural statutory order (9 before 10, 103 before 103A) so the shuffle
        # never depends on whatever order the fetcher returned.
        group = sorted(by_act[slug], key=lambda s: s.sort_key)
        rng.shuffle(group)

        n = len(group)
        n_train = int(n * RATIOS[Split.TRAIN])
        n_val = int(n * RATIOS[Split.VAL])
        for i, section in enumerate(group):
            if i < n_train:
                split = Split.TRAIN
            elif i < n_train + n_val:
                split = Split.VAL
            else:
                split = Split.EVAL_UNSEEN
            assignment[section.section_id] = split.value
    return assignment


def build(seed: int = SPLIT_SEED) -> dict[str, Any]:
    sections = load_sections()
    assignment = assign(sections, seed)

    by_split: dict[str, list[str]] = defaultdict(list)
    for section_id, split in assignment.items():
        by_split[split].append(section_id)

    per_act: dict[str, Counter[str]] = defaultdict(Counter)
    chars: dict[str, int] = defaultdict(int)
    for s in sections:
        split = assignment[s.section_id]
        per_act[split][s.act_slug] += 1
        chars[split] += s.char_count

    payload = {
        "seed": seed,
        "ratios": {k.value: v for k, v in RATIOS.items()},
        "note": (
            "Section-level split, applied BEFORE any QA generation. This is what "
            "makes decontamination structural rather than best-effort: no training "
            "pair and eval question can share a source passage, because the passage "
            "is in exactly one split."
        ),
        "counts": {k: len(v) for k, v in sorted(by_split.items())},
        "chars": dict(sorted(chars.items())),
        "per_act": {k: dict(v) for k, v in sorted(per_act.items())},
        "assignment": dict(sorted(assignment.items())),
    }
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPLIT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_splits() -> dict[str, str]:
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"{SPLIT_PATH} missing - run ragft.corpus.split first")
    data: dict[str, Any] = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    return dict(data["assignment"])


def sections_for(split: Split | str) -> list[Section]:
    want = str(split)
    assignment = load_splits()
    return [s for s in load_sections() if assignment.get(s.section_id) == want]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    payload = build(args.seed)
    print(f"seed {payload['seed']}")
    for split, count in payload["counts"].items():
        acts = payload["per_act"][split]
        print(
            f"  {split:12s} {count:4d} sections  " f"{payload['chars'][split]:>10,} chars  {acts}"
        )
    total = sum(payload["counts"].values())
    print(f"  {'total':12s} {total:4d}")
    print(f"\nWrote {SPLIT_PATH}")


if __name__ == "__main__":
    main()
