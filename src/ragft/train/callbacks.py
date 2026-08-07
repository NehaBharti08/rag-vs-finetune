"""Trainer callbacks.

Kept in their own module because HuggingFace's ``TrainerCallback`` interface
mandates a fixed signature on every hook -- ``args``, ``state``, ``control``,
``**kwargs`` -- whether or not a given hook uses them. Isolating that here
keeps the ``ARG002`` suppression scoped to the one file that genuinely needs
it, instead of blanket-disabling unused-argument detection across the training
package.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


@dataclass
class StepTimer(TrainerCallback):
    """Times optimizer steps, discarding warmup.

    The first few steps include CUDA context setup, kernel autotuning and
    allocator warmup. Including them would understate steady-state throughput
    by a wide margin, so they are dropped rather than averaged in.
    """

    warmup_steps: int = 3
    _t0: float | None = None
    timings: list[float] = field(default_factory=list)

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._t0 = time.perf_counter()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if self._t0 is None:
            return
        elapsed = time.perf_counter() - self._t0
        if state.global_step > self.warmup_steps:
            self.timings.append(elapsed)

    @property
    def mean_step_seconds(self) -> float:
        """Mean seconds per optimizer step, warmup excluded."""
        return sum(self.timings) / len(self.timings) if self.timings else float("nan")
