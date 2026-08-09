"""The four arms must differ ONLY by configuration.

If arms drifted apart in decoding settings or quantization, a difference
between two cells of the 2x2 would be an artefact of plumbing rather than a
property of the adaptation method - and it would be invisible in the results
table. These tests make that drift fail loudly.
"""

from __future__ import annotations

import pytest

from ragft.eval.arms import MAX_NEW_TOKENS, arm_specs, build_prompt, decoding_config
from ragft.settings import BASE_MODEL, QuantConfig

ADAPTER = "out/fake-adapter"


class TestGridShape:
    def test_two_arms_without_an_adapter(self) -> None:
        """Fine-tuned arms cannot exist before an adapter does."""
        assert [a.name for a in arm_specs()] == ["A1_base_zeroshot", "A2_base_rag"]

    def test_four_arms_with_an_adapter(self) -> None:
        assert len(arm_specs(ADAPTER)) == 4

    def test_grid_crosses_both_factors(self) -> None:
        cells = {(a.is_finetuned, a.use_retrieval) for a in arm_specs(ADAPTER)}
        assert cells == {(False, False), (False, True), (True, False), (True, True)}


class TestIdenticalGeneration:
    def test_quantization_is_shared(self) -> None:
        """Threat 3: quantization must not be confounded with adaptation.

        If fine-tuned arms ran 4-bit and base arms bf16, no cell of the grid
        would be interpretable.
        """
        quant = decoding_config()["quant"]
        assert quant["load_in_4bit"] is True
        assert quant["quant_type"] == "nf4"
        assert quant == QuantConfig().model_dump()

    def test_decoding_is_deterministic(self) -> None:
        """Sampling would add variance to a comparison of small differences."""
        assert decoding_config()["do_sample"] is False

    def test_all_arms_use_the_same_checkpoint(self) -> None:
        assert decoding_config()["model"] == BASE_MODEL

    def test_token_budget_is_shared(self) -> None:
        assert decoding_config()["max_new_tokens"] == MAX_NEW_TOKENS


class TestPromptAsymmetryIsDeliberate:
    """Threat 2: the most common cheat in this genre.

    Base arms get few-shot format examples; fine-tuned arms do not. That is an
    asymmetry in the BASE arms' favour - the fine-tuned model was trained on
    the format, so withholding examples from the base model would manufacture a
    format win out of prompt engineering rather than adaptation.
    """

    @pytest.mark.parametrize("arm", [a for a in arm_specs(ADAPTER) if not a.is_finetuned])
    def test_base_arms_get_format_spec_and_examples(self, arm) -> None:  # type: ignore[no-untyped-def]
        prompt = build_prompt(arm, "What is osmosis?", context="[1] ctx")
        assert "**Answer.**" in prompt, "base arm must be told the required format"
        assert prompt.count("**Answer.**") >= 2, "base arm must get worked examples"

    @pytest.mark.parametrize("arm", [a for a in arm_specs(ADAPTER) if a.is_finetuned])
    def test_finetuned_arms_get_no_examples(self, arm) -> None:  # type: ignore[no-untyped-def]
        prompt = build_prompt(arm, "What is osmosis?", context="[1] ctx")
        assert "**Answer.**" not in prompt, "FT arm was trained on the format"

    @pytest.mark.parametrize("arm", arm_specs(ADAPTER))
    def test_every_arm_receives_the_question(self, arm) -> None:  # type: ignore[no-untyped-def]
        assert "What is osmosis?" in build_prompt(arm, "What is osmosis?", context="[1] ctx")


class TestRetrievalIsTheOnlyBranch:
    @pytest.mark.parametrize("arm", [a for a in arm_specs(ADAPTER) if a.use_retrieval])
    def test_retrieval_arms_include_context(self, arm) -> None:  # type: ignore[no-untyped-def]
        assert "SENTINEL_CTX" in build_prompt(arm, "Q?", context="SENTINEL_CTX")

    @pytest.mark.parametrize("arm", [a for a in arm_specs(ADAPTER) if not a.use_retrieval])
    def test_no_retrieval_arms_never_see_context(self, arm) -> None:  # type: ignore[no-untyped-def]
        assert "SENTINEL_CTX" not in build_prompt(arm, "Q?", context="SENTINEL_CTX")
