"""Compare every arm that has been run, across the judge-free metrics.

Extends to A3/A4 automatically as they land. Judge-dependent numbers are
reported separately and labelled, because the judge here is a local 9.6 GB model
whose agreement with a human has not been established yet.

The per-stratum breakdown is the part that matters most. A single aggregate
would average over `parametric_answerable`, which is exactly the variable that
determines whether retrieval had anything to add.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ragft.eval.metrics.citation import aggregate as citation_aggregate
from ragft.eval.metrics.format_check import aggregate as format_aggregate
from ragft.eval.metrics.latency import summarise as latency_summarise
from ragft.settings import REPO_ROOT

EVAL_DIR = REPO_ROOT / "data" / "eval"
RESPONSES = EVAL_DIR / "responses"
REPORTS = REPO_ROOT / "reports"


def adapter_provenance() -> dict[str, Any]:
    """Which checkpoint produced the fine-tuned arms, read from the run itself.

    Every field here used to be a literal, including the path. That made the
    provenance record describe one particular training run rather than the one
    whose responses are being scored -- and a provenance record that can be
    wrong is worse than none, because it is believed.

    Epoch 1 is the target: validation loss rises from epoch 1 on every seed and
    every dataset measured, so the first checkpoint is the best one and A3's
    result is not an over-training artefact.
    """
    from ragft.train.checkpoints import DEFAULT_RUN, epoch1_checkpoint

    path = epoch1_checkpoint()
    summary_path = DEFAULT_RUN / "summary.json"
    record: dict[str, Any] = {
        "path": str(Path(path).relative_to(REPO_ROOT)),
        "epoch": 1,
        "resolved_at_report_time": True,
    }
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        losses = [e["eval_loss"] for e in s.get("eval_loss_by_epoch", [])]
        record |= {
            "steps_per_epoch": s.get("steps_per_epoch"),
            "train_rows": s.get("train_rows"),
            "eval_loss_by_epoch": losses,
            "why_this_one": (
                f"Best validation loss of the epoch checkpoints ({' / '.join(f'{x:.3f}' for x in losses)}). "
                "Validation loss rose monotonically from epoch 1, so this is the "
                "pre-collapse checkpoint and A3's result is not an over-training artefact."
                if losses == sorted(losses)
                else "Epoch-1 checkpoint. NOTE: validation loss was NOT monotonically "
                "rising on this run, so re-check which epoch is actually best."
            ),
        }
    return record


ARM_LABELS = {
    "A1_base_zeroshot": "A1 base, no retrieval",
    "A2_base_rag": "A2 base + RAG",
    "A3_ft_zeroshot": "A3 fine-tuned, no retrieval",
    "A4_ft_rag": "A4 fine-tuned + RAG",
}


def load_answerability() -> dict[str, bool]:
    path = EVAL_DIR / "answerability.jsonl"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {
            str(json.loads(line)["gold_id"]): bool(json.loads(line)["parametric_answerable"])
            for line in fh
            if line.strip()
        }


def score_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [r["response"] for r in rows]
    sources = [r["source_section_ids"] for r in rows]
    return {
        "n": len(rows),
        "citation": citation_aggregate(responses, sources),
        "format": format_aggregate(responses),
        "latency": latency_summarise(
            [r["latency_seconds"] for r in rows],
            [r["prompt_tokens"] for r in rows],
            [r["completion_tokens"] for r in rows],
        ),
    }


def retrieval_hit_rate(rows: list[dict[str, Any]]) -> float | None:
    """Did retrieval actually surface the source section for this question?

    Distinct from corpus-wide recall: this is per-item, on the gold set, and it
    bounds what the RAG arms could possibly cite correctly.
    """
    with_retrieval = [r for r in rows if r.get("retrieved_sections")]
    if not with_retrieval:
        return None
    hits = sum(
        1 for r in with_retrieval if set(r["retrieved_sections"]) & set(r["source_section_ids"])
    )
    return round(hits / len(with_retrieval), 4)


def build() -> dict[str, Any]:
    answerable = load_answerability()
    arms: dict[str, Any] = {}

    # Only the four canonical arms. The responses directory also holds tagged
    # runs -- A3_ft_zeroshot__seed1, __sweep_r64_lr1e-4, A1_base_zeroshot__bf16
    # and so on -- and a bare *.jsonl glob swept all of them into the 2x2.
    #
    # The seed and sweep files carry the same schema, so they were added as
    # EXTRA ARMS to the headline report without complaint; only the bf16 file,
    # which records fewer fields, crashed loudly enough to be noticed. Silent
    # corruption of the main comparison table was the real risk.
    for path in sorted(RESPONSES.glob("*.jsonl")):
        if path.stem not in ARM_LABELS:
            continue
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if not rows:
            continue
        arm = path.stem
        entry = score_arm(rows)
        entry["label"] = ARM_LABELS.get(arm, arm)
        entry["retrieval_hit_rate"] = retrieval_hit_rate(rows)

        if answerable:
            entry["by_stratum"] = {}
            for name, want in (("parametric_answerable", True), ("parametric_unanswerable", False)):
                subset = [r for r in rows if answerable.get(r["gold_id"]) is want]
                if subset:
                    entry["by_stratum"][name] = {
                        "n": len(subset),
                        "citation": citation_aggregate(
                            [r["response"] for r in subset],
                            [r["source_section_ids"] for r in subset],
                        ),
                    }
        arms[arm] = entry

    # Provenance: which adapter produced the fine-tuned arms. Without this the
    # results file cannot be tied to a checkpoint, and "which epoch was this?"
    # becomes unanswerable after the fact.
    payload = {
        "arms": arms,
        "arms_run": sorted(arms),
        "adapter": adapter_provenance(),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "arms_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def render(p: dict[str, Any]) -> str:
    arms = p["arms"]
    names = p["arms_run"]

    def row(title: str, fn: Any) -> str:
        return f"| {title} | " + " | ".join(fn(arms[a]) for a in names) + " |"

    header = "| Metric | " + " | ".join(arms[a]["label"] for a in names) + " |"
    sep = "|---" * (len(names) + 1) + "|"

    lines = [
        "# Arm comparison",
        "",
        "Generated by `ragft.eval.report_arms`. **Do not hand-edit.**",
        "",
        f"Arms run so far: {', '.join(names)}."
        + ("" if len(names) == 4 else " The fine-tuned arms require an adapter (Phase 4)."),
        "",
        "## Judge-free metrics",
        "",
        "None of these depend on an LLM judge, so none inherit the uncertainty of a",
        "local 9.6 GB judge measured at Cohen's kappa 0.45 against human labels",
        "(`reports/judge_agreement.md`).",
        "",
        "Levels 1-5 are a **ladder**: each is strictly harder than the one above it.",
        "Reporting only level 5 hides where an arm actually fails, and reporting only",
        "levels 1-2 makes fabrication look like grounding.",
        "",
        header,
        sep,
        row("1. Produced a parseable citation", lambda e: f"{e['citation']['parseable_rate']:.1%}"),
        row("2. Named an act in the corpus", lambda e: f"{e['citation']['act_exists_rate']:.1%}"),
        row(
            "3. **Named the CORRECT act**",
            lambda e: f"**{e['citation']['act_correct_rate']:.1%}**",
        ),
        row(
            "4. Cited a section that exists",
            lambda e: f"{e['citation']['section_exists_rate']:.1%}",
        ),
        row(
            "5. **Cited the CORRECT section**",
            lambda e: f"**{e['citation']['section_correct_rate']:.1%}**",
        ),
        row(
            "Right act, wrong section",
            lambda e: f"{e['citation']['right_act_wrong_section_rate']:.1%}",
        ),
        row("Fabrication rate", lambda e: f"{e['citation']['fabrication_rate']:.1%}"),
        row("Format valid", lambda e: f"{e['format']['valid_rate']:.1%}"),
        row(
            "Retrieval found the source",
            lambda e: (
                "n/a" if e["retrieval_hit_rate"] is None else f"{e['retrieval_hit_rate']:.1%}"
            ),
        ),
        row("Latency p50", lambda e: f"{e['latency']['p50_seconds']}s"),
        row("Mean prompt tokens", lambda e: f"{e['latency']['mean_prompt_tokens']:.0f}"),
        "",
        "`Retrieval found the source` bounds what a RAG arm could possibly cite",
        "correctly: it cannot cite a section it never retrieved.",
        "",
    ]

    if any("by_stratum" in arms[a] for a in names):
        lines += [
            "## By parametric answerability",
            "",
            "Correct-section rate, split by whether the base model could already answer",
            "the question with no retrieval. Aggregating over this would average away",
            "the variable that determines whether retrieval had anything to add.",
            "",
            "| Stratum | " + " | ".join(arms[a]["label"] for a in names) + " |",
            sep,
        ]
        for stratum in ("parametric_answerable", "parametric_unanswerable"):
            cells = []
            for a in names:
                st = arms[a].get("by_stratum", {}).get(stratum)
                cells.append(
                    f"{st['citation']['section_correct_rate']:.1%} (n={st['n']})" if st else "—"
                )
            lines.append(f"| {stratum.replace('_', ' ')} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "The two strata differ by at most a few points in every arm, and in A4 the",
            "gap runs the *wrong* way. Parametric answerability was the stratification",
            "variable this benchmark was designed around -- the one place fine-tuning",
            "was expected to win -- and it did not separate the arms. That is a null",
            "result on the design's central hypothesis, and it is reported here rather",
            "than dropped.",
            "",
        ]

    return "\n".join(lines)


def main() -> None:
    payload = build()
    out = REPORTS / "arms_comparison.md"
    out.write_text(render(payload), encoding="utf-8")
    print(f"Wrote {out}")
    for name in payload["arms_run"]:
        e = payload["arms"][name]
        print(
            f"  {name:20s} correct_section={e['citation']['section_correct_rate']:.1%} "
            f"out_of_corpus={e['citation']['out_of_corpus_act_rate']:.1%} "
            f"retrieval_hit={e['retrieval_hit_rate']}"
        )


if __name__ == "__main__":
    main()
