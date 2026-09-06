"""Latency p50/p95 and cost per query -- measured, never estimated.

This box is shared, so a latency number taken while another job runs is noise.
Two defences (threat 12):

* The dedicated pass runs in an **exclusive GPU window**, and refuses to start
  if the card is already busy rather than quietly producing a bad number.
* Contention state is recorded alongside the measurement, so a reader can see
  the conditions rather than trust them.

Cost is reported in **GPU-seconds** as well as dollars. GPU-seconds is robust to
contention and to whatever an hourly rate happens to be; dollars needs a rental
rate that is stated as an assumption rather than smuggled in as a fact.

The retrieval arms carry a recurring cost the parametric arms do not: retrieved
context adds prefill tokens to every single query, forever. That is the term
that makes the Phase 6 crossover analysis non-trivial, so prefill and decode are
tracked separately rather than lumped into one latency figure.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Any

import numpy as np

# An A5000 on a commodity cloud, stated as an assumption so a reader can
# substitute their own. Nothing downstream depends on the exact value.
REFERENCE_USD_PER_GPU_HOUR = 0.36


@dataclass(frozen=True)
class ContentionState:
    memory_used_mib: int
    utilization_pct: int
    # Memory held by processes OTHER than this one. This, not the total, is what
    # `exclusive` is derived from -- see gpu_contention.
    other_process_mib: int
    exclusive: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def gpu_contention(memory_threshold_mib: int = 500) -> ContentionState:
    """Snapshot the card, EXCLUDING this process's own memory.

    `exclusive` means nothing *else* is meaningfully on the card. Subtracting our
    own PID is load-bearing, not a refinement: the first version compared TOTAL
    memory against the threshold, so once the 7B model was resident the check
    reported `exclusive=False` regardless of who else was there. It was measuring
    itself. The dedicated latency pass was scored against that broken check and
    its result had to be re-read.
    """
    own_pid = os.getpid()
    try:
        out = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
                check=True,
            )
            .stdout.strip()
            .splitlines()[0]
        )
        mem, util = (int(x.strip()) for x in out.split(","))

        # Memory held by OTHER processes is what determines exclusivity.
        apps = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        other_mib = 0
        for line in apps.splitlines():
            if not line.strip():
                continue
            pid_str, used_str = (x.strip() for x in line.split(","))
            if int(pid_str) != own_pid:
                other_mib += int(used_str)
    except Exception:  # noqa: BLE001 - a missing nvidia-smi must not fail a run
        return ContentionState(-1, -1, -1, exclusive=False)
    return ContentionState(mem, util, other_mib, exclusive=other_mib < memory_threshold_mib)


def summarise(
    latencies: list[float],
    prompt_tokens: list[int],
    completion_tokens: list[int],
    contention: ContentionState | None = None,
    usd_per_gpu_hour: float = REFERENCE_USD_PER_GPU_HOUR,
) -> dict[str, Any]:
    if not latencies:
        return {"n": 0}

    arr = np.asarray(latencies, dtype=np.float64)
    total_seconds = float(arr.sum())
    mean_prompt = float(np.mean(prompt_tokens)) if prompt_tokens else 0.0
    mean_completion = float(np.mean(completion_tokens)) if completion_tokens else 0.0

    return {
        "n": len(latencies),
        "p50_seconds": round(float(np.percentile(arr, 50)), 4),
        "p95_seconds": round(float(np.percentile(arr, 95)), 4),
        "mean_seconds": round(float(arr.mean()), 4),
        "min_seconds": round(float(arr.min()), 4),
        "max_seconds": round(float(arr.max()), 4),
        # The contention-robust unit. Prefer this when comparing arms.
        "gpu_seconds_per_query": round(total_seconds / len(latencies), 4),
        "usd_per_1k_queries": round(
            total_seconds / len(latencies) / 3600 * usd_per_gpu_hour * 1000, 4
        ),
        "usd_per_gpu_hour_assumed": usd_per_gpu_hour,
        # Prefill is where retrieval's recurring cost lives: every query pays
        # for its retrieved context, forever.
        "mean_prompt_tokens": round(mean_prompt, 1),
        "mean_completion_tokens": round(mean_completion, 1),
        "contention": (contention or gpu_contention()).to_dict(),
        "caveat": (
            "Latency is only comparable across arms when measured in an exclusive "
            "GPU window. If `contention.exclusive` is false, treat these as "
            "indicative and read gpu_seconds_per_query instead."
        ),
    }


def crossover_queries(
    training_gpu_hours: float, ft_gpu_seconds: float, rag_gpu_seconds: float
) -> dict[str, Any]:
    """At what query volume does fine-tuning's one-off training cost pay back?

    N* = training_cost / (per_query_rag - per_query_ft)

    If retrieval is not actually more expensive per query, there is no
    crossover and fine-tuning never amortises on cost grounds - which is a
    legitimate finding, reported rather than hidden behind an extrapolated
    plot.
    """
    delta = rag_gpu_seconds - ft_gpu_seconds
    if delta <= 0:
        return {
            "crossover_queries": None,
            "reason": (
                "Retrieval is not more expensive per query than the fine-tuned arm, "
                "so the one-off training cost never amortises. There is no crossover."
            ),
        }
    return {
        "crossover_queries": int(training_gpu_hours * 3600 / delta),
        "training_gpu_hours": training_gpu_hours,
        "per_query_delta_gpu_seconds": round(delta, 4),
    }
