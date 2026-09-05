"""Cost / quality / latency frontier and the fine-tuning payback point. Phase 6.

Fine-tuning costs GPU-hours once; retrieval costs prefill tokens on every query,
forever. Which is cheaper depends entirely on query volume, so "is fine-tuning
worth it?" has no scalar answer -- it has a crossover point:

    N* = training_gpu_seconds / (rag_gpu_seconds_per_query - ft_gpu_seconds_per_query)

The denominator is the trap. It is only positive if the fine-tuned arm is
actually cheaper per query, and here it is not: the LoRA adapter is attached
unmerged, so every forward pass pays extra matmuls and the fine-tuned arm is
*slower* than the base model despite a far shorter prompt. When the denominator
is negative there is no crossover at any volume, and the honest output is
"never", not an extrapolated line.

That is reported as the finding rather than worked around, because working
around it -- by merging the adapter, or by pricing only training -- would answer
a question nobody asked.

Quality is taken from the judge-free correct-section rate. Cost is in
GPU-seconds, which is robust to contention on a shared box; dollars are shown
against a stated rental rate rather than smuggled in as a fact.
"""

from __future__ import annotations

import json
from typing import Any

from ragft.eval.metrics.latency import REFERENCE_USD_PER_GPU_HOUR, crossover_queries
from ragft.settings import REPO_ROOT

REPORTS = REPO_ROOT / "reports"
TRAIN_SUMMARY = REPO_ROOT / "out" / "seed42_r16_lr0.0002_e3" / "summary.json"

ARM_LABELS = {
    "A1_base_zeroshot": "A1 base, no retrieval",
    "A2_base_rag": "A2 base + RAG",
    "A3_ft_zeroshot": "A3 fine-tuned, no retrieval",
    "A4_ft_rag": "A4 fine-tuned + RAG",
}


def _training_gpu_hours() -> dict[str, Any]:
    """Cost of the adapter actually evaluated -- epoch 1, not the full 3-epoch run.

    The evaluated checkpoint is checkpoint-354, the end of epoch 1. Charging the
    full three-epoch wall clock to it would overstate the training cost of the
    artifact under test by 3x. The epochs that followed made validation loss
    worse, so they are a sunk cost of *finding* the checkpoint, not of producing
    it. Both figures are reported.
    """
    s = json.loads(TRAIN_SUMMARY.read_text(encoding="utf-8"))
    full = s["elapsed_minutes"] / 60
    return {
        "full_run_gpu_hours": round(full, 3),
        "evaluated_checkpoint_gpu_hours": round(full / s["epochs"], 3),
        "epochs_run": s["epochs"],
        "evaluated_epoch": 1,
        "note": (
            "The evaluated adapter is epoch 1. The full 3-epoch run is the search "
            "cost; epoch 1 alone is the production cost of the artifact under test."
        ),
    }


def build() -> dict[str, Any]:
    lat = json.loads((REPORTS / "latency.json").read_text(encoding="utf-8"))
    arms_report = json.loads((REPORTS / "arms_comparison.json").read_text(encoding="utf-8"))
    training = _training_gpu_hours()

    rows = []
    for arm, label in ARM_LABELS.items():
        if arm not in lat["arms"] or arm not in arms_report["arms"]:
            continue
        la = lat["arms"][arm]
        cit = arms_report["arms"][arm]["citation"]
        gpu_s = la["gpu_seconds_per_query"]
        rows.append(
            {
                "arm": arm,
                "label": label,
                "correct_section_rate": cit["section_correct_rate"],
                "correct_act_rate": cit["act_correct_rate"],
                "p50_seconds": la["p50_seconds"],
                "p95_seconds": la["p95_seconds"],
                "gpu_seconds_per_query": gpu_s,
                "mean_prompt_tokens": la["mean_prompt_tokens"],
                "usd_per_1k_queries": round(gpu_s / 3600 * REFERENCE_USD_PER_GPU_HOUR * 1000, 4),
            }
        )

    by_arm = {r["arm"]: r for r in rows}
    crossovers = {}
    # The comparison a deployer actually faces: replace RAG with the fine-tuned
    # model, or add fine-tuning on top of RAG.
    for name, (rag_arm, ft_arm) in {
        "ft_replaces_rag": ("A2_base_rag", "A3_ft_zeroshot"),
        "ft_added_to_rag": ("A2_base_rag", "A4_ft_rag"),
    }.items():
        if rag_arm in by_arm and ft_arm in by_arm:
            crossovers[name] = {
                "baseline": rag_arm,
                "candidate": ft_arm,
                "quality_delta": round(
                    by_arm[ft_arm]["correct_section_rate"]
                    - by_arm[rag_arm]["correct_section_rate"],
                    4,
                ),
                **crossover_queries(
                    training["evaluated_checkpoint_gpu_hours"],
                    by_arm[ft_arm]["gpu_seconds_per_query"],
                    by_arm[rag_arm]["gpu_seconds_per_query"],
                ),
            }

    payload = {
        "training_cost": training,
        "usd_per_gpu_hour_assumed": REFERENCE_USD_PER_GPU_HOUR,
        "window_was_exclusive": lat.get("window_was_exclusive"),
        "peak_other_process_mib": lat.get("peak_other_process_mib"),
        "arms": rows,
        "crossovers": crossovers,
    }
    (REPORTS / "frontier.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (REPORTS / "frontier.md").write_text(render(payload), encoding="utf-8")
    return payload


def render(p: dict[str, Any]) -> str:
    t = p["training_cost"]
    excl = p["window_was_exclusive"]
    lines = [
        "# Cost, latency, and the fine-tuning payback point",
        "",
        "Generated by `ragft.analysis.frontier`. **Do not hand-edit.**",
        "",
        f"Latency measured in a GPU window verified exclusive: **{excl}** "
        f"(peak memory held by other processes: {p['peak_other_process_mib']} MiB).",
        "",
        "## The frontier",
        "",
        "| Arm | Correct section | Correct act | p50 | GPU-s/query | Prompt tokens | $/1k queries |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in p["arms"]:
        lines.append(
            f"| {r['label']} | **{r['correct_section_rate']:.1%}** | {r['correct_act_rate']:.1%} | "
            f"{r['p50_seconds']:.2f}s | {r['gpu_seconds_per_query']:.2f} | "
            f"{r['mean_prompt_tokens']:.0f} | ${r['usd_per_1k_queries']:.3f} |"
        )

    lines += [
        "",
        f"Dollar figures assume **${p['usd_per_gpu_hour_assumed']}/GPU-hour**, stated as an",
        "assumption so a reader can substitute their own. Nothing below depends on it.",
        "",
        "## Training cost",
        "",
        "| What | Cost |",
        "|---|---|",
        f"| Full {t['epochs_run']}-epoch run | {t['full_run_gpu_hours']} GPU-hours |",
        f"| Evaluated checkpoint (epoch {t['evaluated_epoch']}) | **{t['evaluated_checkpoint_gpu_hours']} GPU-hours** |",
        "",
        t["note"],
        "",
        "## Payback",
        "",
    ]

    for name, c in p["crossovers"].items():
        lines += [
            f"### {name.replace('_', ' ')}",
            "",
            f"{c['baseline']} → {c['candidate']}, quality change "
            f"**{c['quality_delta'] * 100:+.1f} points** of correct-section rate.",
            "",
        ]
        if c.get("crossover_queries") is None:
            lines += [
                f"**No crossover at any query volume.** {c['reason']}",
                "",
            ]
        else:
            lines += [
                f"Breaks even after **{c['crossover_queries']:,} queries** "
                f"({c['per_query_delta_gpu_seconds']} GPU-s saved per query).",
                "",
            ]

    lines += [
        "## Why there is no payback here",
        "",
        "The adapter is attached **unmerged**, so every forward pass runs the extra",
        "LoRA matmuls. The fine-tuned arms are therefore slower per query than the",
        "base model *and* than RAG, despite A3 using a ~9x shorter prompt. A one-off",
        "training cost can only amortise against a per-query saving, and there is no",
        "per-query saving to amortise against.",
        "",
        "Merging the adapter into the base weights would remove most of this overhead.",
        "It was not done here because the project's stated artifact is a ~160 MB",
        "adapter rather than a 15 GB merged checkpoint. So the honest scope of this",
        "result is: **fine-tuning does not pay back in an unmerged deployment.** A",
        "merged deployment is a different measurement, and this benchmark did not",
        "make it.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    p = build()
    for r in p["arms"]:
        print(
            f"{r['arm']:22s} correct={r['correct_section_rate']:.1%} "
            f"gpu_s={r['gpu_seconds_per_query']:.2f} p50={r['p50_seconds']:.2f}s"
        )
    for name, c in p["crossovers"].items():
        print(f"{name}: crossover={c.get('crossover_queries')} quality={c['quality_delta']:+.1%}")


if __name__ == "__main__":
    main()
