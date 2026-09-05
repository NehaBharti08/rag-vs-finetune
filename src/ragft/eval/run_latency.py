"""Dedicated latency pass in a verified-exclusive GPU window. Phase 6.

The per-item latencies recorded during the arm runs are not trustworthy for
comparison, and the reason is a flaw in how they were captured: `gpu_contention()`
was sampled when the *report* was generated, not while the arms were running, so
the `exclusive` flag attached to them describes the wrong moment entirely.

This pass fixes that. It samples contention immediately before and immediately
after the measurement and records both, so a reader can see the conditions rather
than trust a flag. If the card was not exclusive at either end, the numbers are
still written but marked untrustworthy for cross-arm comparison.

Fewer items than the arm runs (n=25 x 3 repeats rather than 300), because a
clean window is worth more here than sample size: latency variance from
contention dwarfs the variance between items.

Usage::

    uv run python -m ragft.eval.run_latency --adapter out/.../checkpoint-354
"""

from __future__ import annotations

import argparse
import json
from typing import Any

from ragft.eval.arms import ArmRunner, arm_specs, build_prompt
from ragft.eval.metrics.latency import gpu_contention, summarise
from ragft.eval.runner import load_gold
from ragft.settings import REPO_ROOT

REPORTS = REPO_ROOT / "reports"


def run(adapter: str, n_items: int = 25, repeats: int = 3) -> dict[str, Any]:
    # Sampled at every arm boundary, not just the two ends: a job that arrives
    # AND leaves inside the window is invisible to a before/after pair, and on a
    # shared box that is the common case rather than a corner one.
    samples: list[Any] = []

    def sample(when: str) -> Any:
        state = gpu_contention()
        samples.append({"when": when, **state.to_dict()})
        print(f"contention @{when}: {state.to_dict()}")
        return state

    before = sample("before")

    items = load_gold()[:n_items]
    specs = arm_specs(adapter)
    runner = ArmRunner()
    results: dict[str, Any] = {}

    from ragft.retrieval.retriever import retriever

    for arm in specs:
        runner.set_adapter(arm.adapter_path)
        ret = retriever() if arm.use_retrieval else None
        lat: list[float] = []
        prompt_toks: list[int] = []
        completion_toks: list[int] = []

        for _ in range(repeats):
            for item in items:
                context = ""
                if ret is not None:
                    context = ret.format_context(ret.retrieve(item.question))
                out = runner.generate(build_prompt(arm, item.question, context))
                lat.append(out["latency_seconds"])
                prompt_toks.append(out["prompt_tokens"])
                completion_toks.append(out["completion_tokens"])

        sample(f"after_{arm.name}")
        results[arm.name] = summarise(lat, prompt_toks, completion_toks, contention=before)
        print(
            f"  {arm.name:20s} p50={results[arm.name]['p50_seconds']:.3f}s "
            f"p95={results[arm.name]['p95_seconds']:.3f}s n={len(lat)}"
        )

    after = sample("after")
    peak_other = max(int(s["other_process_mib"]) for s in samples)

    payload = {
        "n_items": n_items,
        "repeats": repeats,
        "contention_samples": samples,
        "contention_before": before.to_dict(),
        "contention_after": after.to_dict(),
        "peak_other_process_mib": peak_other,
        "window_was_exclusive": all(bool(s["exclusive"]) for s in samples),
        "arms": results,
        "note": (
            "Contention is sampled at every arm boundary and counts only memory "
            "held by OTHER processes. An earlier version compared TOTAL GPU memory "
            "against the threshold, so once the 7B model was resident it reported "
            "exclusive=False no matter who else was on the card - it was measuring "
            "itself, and the first run of this pass was scored against that broken "
            "check. The latencies in the main arm runs have a different and "
            "unrelated flaw: they sampled contention at REPORT time rather than "
            "during measurement, so their flag describes the wrong moment. These "
            "supersede them."
        ),
    }
    (REPORTS / "latency.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwindow exclusive: {payload['window_was_exclusive']}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="out/seed42_r16_lr0.0002_e3/checkpoint-354")
    parser.add_argument("--n-items", type=int, default=25)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    run(args.adapter, args.n_items, args.repeats)


if __name__ == "__main__":
    main()
