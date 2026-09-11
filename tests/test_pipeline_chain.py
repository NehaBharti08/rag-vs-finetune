"""The dataset pipeline must chain: each step reads what the previous one wrote.

These are one-line assertions about CLI defaults, and they exist because getting
them wrong cost a 16-hour reproduction run.

The defaults described a pipeline that could not work. `balance` defaulted to
reading `clean.jsonl`, a file it PRECEDES, so the documented sequence died with
FileNotFoundError after generation had already finished.

The second failure is the one worth testing for. `decontaminate` defaulted to
reading `filtered.jsonl`, skipping `balance` entirely -- and that path does NOT
crash. It silently produces a `clean.jsonl` with the mix as generated
(unanswerable 6.3%) rather than as declared (10%), under-training refusal by
almost half. That is the exact failure `balance` was written to prevent, and it
would have shown up only as a weaker abstention result nobody could explain.
"""

from __future__ import annotations

from ragft.dataset import balance, decontaminate
from ragft.dataset import filter as qa_filter

# generate -> raw -> filter -> balance -> decontaminate -> clean (training reads this)
EXPECTED_CHAIN = [
    (qa_filter.run, "raw.jsonl", "filtered.jsonl"),
    (balance.run, "filtered.jsonl", "balanced.jsonl"),
]


def _default(fn, param: str) -> object:
    import inspect

    return inspect.signature(fn).parameters[param].default


class TestChain:
    def test_filter_reads_raw_writes_filtered(self) -> None:
        assert _default(qa_filter.run, "in_name") == "raw.jsonl"
        assert _default(qa_filter.run, "out_name") == "filtered.jsonl"

    def test_balance_reads_what_filter_wrote(self) -> None:
        """balance once defaulted to clean.jsonl -- a file it precedes."""
        assert _default(balance.run, "in_name") == _default(qa_filter.run, "out_name")

    def test_decontaminate_reads_what_balance_wrote(self) -> None:
        """The silent failure: reading `filtered` skips balancing entirely.

        It does not crash. It produces a clean.jsonl with the mix as generated
        rather than as declared, under-training refusal by almost half.
        """
        assert _default(decontaminate.run, "in_name") == _default(balance.run, "out_name")

    def test_training_reads_decontaminate_output(self) -> None:
        from ragft.train.sft import QA_PATH

        assert QA_PATH.name == "clean.jsonl", (
            "Training must read the decontaminated set. If this changes, the "
            "decontamination guarantee stops applying to the data actually trained on."
        )


class TestNoOrphanSteps:
    def test_every_intermediate_is_consumed(self) -> None:
        """No step may write a file nothing downstream reads.

        An orphan output means the step is decorative: it runs, reports success,
        and changes nothing about what gets trained on.
        """
        produced = {_default(qa_filter.run, "out_name"), _default(balance.run, "out_name")}
        consumed = {_default(balance.run, "in_name"), _default(decontaminate.run, "in_name")}
        orphans = produced - consumed
        assert not orphans, f"nothing downstream reads: {orphans}"
