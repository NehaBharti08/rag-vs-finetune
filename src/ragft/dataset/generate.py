"""Generate the synthetic instruction-tuning set, resumably.

Two properties this module exists to guarantee:

**Resumability.** ~1500 model calls over several hours on a shared GPU will be
interrupted. Every completed task is appended to JSONL immediately and its id
recorded, so a restart skips finished work rather than repeating it. There is
no "run it again from scratch" mode by accident.

**Provenance.** Every pair records the section ids and passage hashes it came
from. Decontamination verifies its guarantees against these fields, so a pair
without honest provenance is worse than no pair at all.

The `Source.` citation is built from verified section metadata, never written
by the generator. A model asked to produce its own citation will sometimes
produce a wrong one, and training on wrong citations would teach precisely the
failure the citation metric exists to detect.

Usage::

    uv run python -m ragft.dataset.generate --splits train val
    uv run python -m ragft.dataset.generate --limit 5   # smoke test
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from ragft.corpus.parse import Section, load_sections
from ragft.corpus.split import load_splits
from ragft.dataset.passages import to_passages
from ragft.dataset.schema import TYPE_MIX, QAPair, QAType, chunk_sha256, make_qa_id
from ragft.llmclient import build_client
from ragft.settings import REPO_ROOT, Settings

QA_DIR = REPO_ROOT / "data" / "qa"
PROMPT_DIR = REPO_ROOT / "prompts" / "generation"
GEN_SEED = 20260807

# How many pairs to ask for per call, per type. Batching several per call is
# what keeps the run near 5 hours instead of 20.
PER_CALL: dict[QAType, int] = {
    QAType.FACTUAL: 4,
    QAType.DEFINITION: 2,
    QAType.APPLIED: 2,
    QAType.UNANSWERABLE: 1,
    QAType.MULTIHOP: 4,
}

_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)
_write_lock = threading.Lock()


@dataclass(frozen=True)
class Task:
    task_id: str
    qa_type: QAType
    sections: tuple[Section, ...]
    passages: tuple[str, ...]
    n: int
    split: str


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def build_tasks(sections: list[Section], split: str, rng: random.Random) -> list[Task]:
    """Deterministic task list for one split."""
    tasks: list[Task] = []
    usable = [s for s in sections if to_passages(s)]

    # ONE type per section, assigned in proportion to TYPE_MIX, rather than
    # every type for every section.
    #
    # The textbook corpus had 299 training sections of ~13,000 characters, so
    # asking each for four question types was reasonable. This corpus has 988
    # sections of ~550 characters: the same scheme produced 4,446 tasks (~18
    # hours) and would have drawn four questions out of a single short
    # provision, which yields near-duplicates rather than four questions.
    #
    # Assigning one type per section covers more of the statute book for less
    # compute, and the declared mix is preserved because assignment is
    # proportional. Shuffled first so type does not correlate with statutory
    # order, which would confound type with act.
    single_types: list[QAType] = [t for t in TYPE_MIX if t is not QAType.MULTIHOP]
    weight_total = sum(TYPE_MIX[t] for t in single_types)

    pool = list(usable)
    rng.shuffle(pool)
    assigned: list[tuple[Section, QAType]] = []
    cursor = 0
    for qa_type in single_types:
        share = TYPE_MIX[qa_type] / weight_total
        take = round(share * len(pool))
        for section in pool[cursor : cursor + take]:
            assigned.append((section, qa_type))
        cursor += take
    # Rounding can leave a tail; give it to the largest bucket.
    for section in pool[cursor:]:
        assigned.append((section, single_types[0]))

    for section, qa_type in assigned:
        passages = to_passages(section)
        tasks.append(
            Task(
                task_id=f"{split}:{section.section_id}:{qa_type.value}",
                qa_type=qa_type,
                sections=(section,),
                passages=(passages[0],),
                n=PER_CALL[qa_type],
                split=split,
            )
        )

    # Multi-hop pairs. Prefer two sections from the same book (a real
    # conceptual link), and allow the Biology <-> Anatomy seam, which is where
    # genuinely cross-title questions come from rather than contrived ones.
    # Cap multi-hop to its declared share of PAIRS, not of sections. Pairing
    # every two sections produced 494 tasks x 4 pairs = 41% of the dataset
    # against a declared 20%, which would have silently rewritten the mix that
    # `tests/test_dataset_mix.py` exists to enforce.
    single_pairs = sum(t.n for t in tasks)
    mh_share = TYPE_MIX[QAType.MULTIHOP]
    target_mh_pairs = int(single_pairs * mh_share / (1 - mh_share))
    max_mh_tasks = max(1, target_mh_pairs // PER_CALL[QAType.MULTIHOP])

    pool = list(usable)
    rng.shuffle(pool)
    for a, b in list(zip(pool[::2], pool[1::2], strict=False))[:max_mh_tasks]:
        tasks.append(
            Task(
                task_id=f"{split}:{a.section_id}+{b.section_id}:multihop",
                qa_type=QAType.MULTIHOP,
                sections=(a, b),
                passages=(to_passages(a)[0], to_passages(b)[0]),
                n=PER_CALL[QAType.MULTIHOP],
                split=split,
            )
        )
    return tasks


def render(task: Task) -> str:
    if task.qa_type is QAType.MULTIHOP:
        a, b = task.sections
        return load_prompt("multihop").format(
            n=task.n,
            act_a=a.act_name,
            label_a=a.label,
            title_a=a.title,
            passage_a=task.passages[0],
            act_b=b.act_name,
            label_b=b.label,
            title_b=b.title,
            passage_b=task.passages[1],
        )
    s = task.sections[0]
    return load_prompt(task.qa_type.value).format(
        n=task.n, act=s.act_name, label=s.label, title=s.title, passage=task.passages[0]
    )


def parse_items(raw: str) -> list[dict[str, Any]]:
    """Pull the JSON array out of a model response.

    Local models wrap JSON in prose or fences despite instructions, so the
    array is extracted rather than assumed to be the whole response. A response
    that still will not parse is dropped and counted, not guessed at.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    m = _JSON_ARRAY_RE.search(text)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [i for i in items if isinstance(i, dict)]


def to_pairs(task: Task, items: list[dict[str, Any]], model: str) -> list[QAPair]:
    pairs: list[QAPair] = []
    primary = task.sections[0]
    for item in items:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        why = str(item.get("why", "")).strip()
        if not (question and answer):
            continue
        section_ids = [s.section_id for s in task.sections]
        pairs.append(
            QAPair(
                qa_id=make_qa_id(question, section_ids),
                qa_type=task.qa_type.value,
                question=question,
                answer=answer,
                why=why,
                citation=primary.citation,
                source_section_ids=section_ids,
                source_chunk_sha256=[chunk_sha256(p) for p in task.passages],
                split=task.split,
                act_slugs=sorted({s.act_slug for s in task.sections}),
                generator_model=model,
            )
        )
    return pairs


def run(splits: list[str], workers: int = 3, limit: int | None = None) -> dict[str, Any]:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QA_DIR / "raw.jsonl"
    done_path = QA_DIR / "completed_tasks.txt"

    settings = Settings(_env_file=None)
    client = build_client(settings, "generation")
    model = client.model

    assignment = load_splits()
    all_sections = load_sections()
    rng = random.Random(GEN_SEED)

    tasks: list[Task] = []
    for split in splits:
        subset = [s for s in all_sections if assignment.get(s.section_id) == split]
        tasks.extend(build_tasks(subset, split, rng))

    done: set[str] = set()
    if done_path.exists():
        done = {
            ln.strip() for ln in done_path.read_text(encoding="utf-8").splitlines() if ln.strip()
        }

    pending = [t for t in tasks if t.task_id not in done]
    if limit:
        pending = pending[:limit]

    print(f"model={model} splits={splits} workers={workers}")
    print(f"tasks: {len(tasks)} total, {len(done)} already done, {len(pending)} pending")
    if not pending:
        print("nothing to do")
        return {"generated": 0, "tasks": 0}

    system_prompt = load_prompt("_system")
    stats = {"ok": 0, "empty": 0, "pairs": 0, "failed": 0}
    t_start = time.perf_counter()

    def work(task: Task) -> tuple[Task, list[QAPair]]:
        # Temperature 0.7: deterministic generation over 1500 near-identical
        # prompts collapses into repetitive phrasing, which near-duplicate
        # filtering would then throw away.
        raw = client.complete(render(task), temperature=0.7, system=system_prompt)
        return task, to_pairs(task, parse_items(raw), model)

    with (
        out_path.open("a", encoding="utf-8") as out_fh,
        done_path.open("a", encoding="utf-8") as done_fh,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {pool.submit(work, t): t for t in pending}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            try:
                task, pairs = future.result()
            except Exception as exc:  # noqa: BLE001 - one bad task must not kill the run
                stats["failed"] += 1
                print(f"  ! {task.task_id}: {exc}")
                continue

            with _write_lock:
                for p in pairs:
                    out_fh.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
                done_fh.write(task.task_id + "\n")
                out_fh.flush()
                done_fh.flush()

            stats["pairs"] += len(pairs)
            stats["ok" if pairs else "empty"] += 1

            if i % 25 == 0 or i == len(pending):
                rate = i / (time.perf_counter() - t_start)
                eta = (len(pending) - i) / rate / 60 if rate else 0
                print(
                    f"  [{i}/{len(pending)}] pairs={stats['pairs']} "
                    f"empty={stats['empty']} failed={stats['failed']} "
                    f"{rate * 60:.1f} tasks/min  ETA {eta:.0f}m",
                    flush=True,
                )

    elapsed = time.perf_counter() - t_start
    summary = {
        "model": model,
        "tasks_run": len(pending),
        "elapsed_seconds": round(elapsed, 1),
        "usage": client.usage.summary(),
        **stats,
    }
    (QA_DIR / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{stats['pairs']} pairs in {elapsed / 60:.1f} min -> {out_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.splits, workers=args.workers, limit=args.limit)


if __name__ == "__main__":
    main()
