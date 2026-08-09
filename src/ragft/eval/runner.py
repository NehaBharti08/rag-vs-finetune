"""Run arms over the gold set: resumable, frozen-checked, cached per item.

Three properties, each protecting a different failure mode.

**Frozen-hash assertion.** The runner refuses to produce numbers if the eval
set, judge prompts, or metric source no longer match `configs/eval/frozen.lock`.
A metric invented after seeing results is not a metric, and the only way to make
that rule real rather than aspirational is to make the harness refuse.

**Resumability.** Four arms over 300 items on a shared GPU will be interrupted.
Every response is appended and flushed immediately, keyed by (arm, gold_id), so
a restart skips completed work instead of repeating it.

**One generate() path.** Retrieval is the only branch, and it changes what goes
into the prompt, never how generation happens. `tests/test_arms_parity.py`
asserts the decoding config is identical across arms.

Latency is recorded per item but should NOT be read from a normal run: this box
is shared, so a contended measurement is noise. The dedicated latency pass runs
separately in an exclusive GPU window.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any

from ragft.eval.arms import ArmRunner, ArmSpec, arm_specs, build_prompt, decoding_config
from ragft.eval.frozen import check as frozen_check
from ragft.eval.goldset import GoldItem, load_candidates
from ragft.settings import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "data" / "eval" / "responses"


@dataclass(frozen=True)
class ResponseRecord:
    arm: str
    gold_id: str
    question: str
    reference: str
    stratum: str
    unanswerable: bool
    response: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    retrieved_sections: list[str]
    source_section_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def load_gold(path_name: str = "gold_candidates.jsonl") -> list[GoldItem]:
    """Prefer the frozen gold set; fall back to candidates before freezing."""
    frozen = REPO_ROOT / "data" / "eval" / "gold.jsonl"
    if frozen.exists():
        with frozen.open(encoding="utf-8") as fh:
            return [GoldItem(**json.loads(line)) for line in fh if line.strip()]
    print(f"note: {frozen.name} absent, using {path_name} (harness not yet frozen)")
    return load_candidates()


def completed(arm_name: str) -> set[str]:
    path = RESULTS_DIR / f"{arm_name}.jsonl"
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        return {str(json.loads(line)["gold_id"]) for line in fh if line.strip()}


def run_arm(
    arm: ArmSpec, items: list[GoldItem], runner: ArmRunner, limit: int | None = None
) -> dict[str, Any]:
    from ragft.retrieval.retriever import retriever

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{arm.name}.jsonl"
    done = completed(arm.name)
    todo = [i for i in items if i.gold_id not in done]
    if limit:
        todo = todo[:limit]

    print(f"\n[{arm.name}] {arm.label}: {len(done)} done, {len(todo)} to run")
    if not todo:
        return {"arm": arm.name, "ran": 0, "total": len(done)}

    runner.set_adapter(arm.adapter_path)
    ret = retriever() if arm.use_retrieval else None

    with out_path.open("a", encoding="utf-8") as fh:
        for n, item in enumerate(todo, 1):
            context, sections = "", []
            if ret is not None:
                hits = ret.retrieve(item.question)
                context = ret.format_context(hits)
                sections = [h.section_id for h in hits]

            result = runner.generate(build_prompt(arm, item.question, context))
            record = ResponseRecord(
                arm=arm.name,
                gold_id=item.gold_id,
                question=item.question,
                reference=item.reference,
                stratum=item.stratum,
                unanswerable=item.unanswerable,
                response=result["response"],
                latency_seconds=result["latency_seconds"],
                prompt_tokens=result["prompt_tokens"],
                completion_tokens=result["completion_tokens"],
                retrieved_sections=sections,
                source_section_ids=item.source_section_ids,
            )
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            fh.flush()

            if n % 25 == 0 or n == len(todo):
                print(f"  [{n}/{len(todo)}] {result['latency_seconds']:.2f}s/item", flush=True)

    return {"arm": arm.name, "ran": len(todo), "total": len(done) + len(todo)}


def run(
    arms: list[str] | None = None, adapter: str | None = None, limit: int | None = None
) -> None:
    if frozen_check() != 0:
        raise SystemExit(
            "Eval harness does not match configs/eval/frozen.lock. Refusing to "
            "produce numbers against a harness that changed after freezing."
        )

    items = load_gold()
    specs = [a for a in arm_specs(adapter) if arms is None or a.name in arms]
    if not specs:
        raise SystemExit(f"no arms matched {arms}")

    print(f"gold items: {len(items)}")
    print(f"decoding (identical for every arm): {decoding_config()}")

    runner = ArmRunner()
    summaries = [run_arm(spec, items, runner, limit) for spec in specs]
    print("\n" + json.dumps(summaries, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--adapter", default=None, help="path to a LoRA adapter (enables A3/A4)")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    run(args.arms, args.adapter, args.limit)


if __name__ == "__main__":
    main()
