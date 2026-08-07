"""Tests for the configuration contract.

The behaviours pinned here are the ones whose failure would silently corrupt a
benchmark number rather than crash a run.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ragft.settings import (
    BASE_MODEL,
    LoraConfig,
    Provider,
    RetrievalConfig,
    Settings,
    TrainConfig,
    load_retrieval_config,
    load_train_config,
)


class TestStrictness:
    """A typo must fail the run, not be silently ignored.

    The failure mode: a misspelled key is dropped, the run completes, and the
    resulting number is attributed to a configuration that was never applied.
    """

    def test_unknown_key_in_retrieval_config_raises(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig.model_validate({"top_k_contest": 5})  # typo

    def test_unknown_key_in_train_config_raises(self) -> None:
        with pytest.raises(ValidationError):
            TrainConfig.model_validate({"learning_rat": 2e-4})  # typo

    def test_configs_are_frozen(self) -> None:
        cfg = TrainConfig()
        with pytest.raises(ValidationError):
            cfg.learning_rate = 1e-3  # type: ignore[misc]


class TestLoraScaling:
    """alpha is a ratio to r, and that is load-bearing for the sweep."""

    def test_alpha_derives_from_ratio(self) -> None:
        assert LoraConfig(r=16, alpha_ratio=2.0).alpha == 32
        assert LoraConfig(r=64, alpha_ratio=2.0).alpha == 128

    def test_scaling_is_invariant_across_rank(self) -> None:
        """alpha/r must stay constant when r changes.

        LoRA scales its update by alpha/r. If that ratio moved with rank, the
        r=16 vs r=64 sweep arm would confound rank with a change in effective
        learning rate, and the sweep result would be uninterpretable.
        """
        low, high = LoraConfig(r=16), LoraConfig(r=64)
        assert low.alpha / low.r == pytest.approx(high.alpha / high.r)

    def test_targets_all_seven_projections(self) -> None:
        """Attention AND MLP. Attention-only measurably underperforms."""
        assert set(LoraConfig().target_modules) == {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }


class TestBatchMath:
    def test_effective_batch_size(self) -> None:
        cfg = TrainConfig()
        assert cfg.effective_batch_size == 16
        assert (
            cfg.effective_batch_size
            == cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
        )


class TestRetrievalInvariants:
    def test_candidate_pool_must_exceed_selection(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig.model_validate({"top_k_retrieve": 3, "top_k_context": 5})


class TestFamilyConfound:
    """The generator and judge must never share a family with the student.

    A same-family generator distils its own style into the fine-tuned arm; a
    same-family judge scores its own family favourably. Either invalidates the
    benchmark's conclusions, so this is enforced in code rather than left as a
    note in a README.
    """

    def test_qwen_judge_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGFT_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_JUDGE_MODEL", "qwen2.5:14b")
        with pytest.raises(ValidationError, match="family"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_qwen_generator_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGFT_PROVIDER", "ollama")
        monkeypatch.setenv("OLLAMA_GENERATION_MODEL", "qwen2.5-coder:14b")
        with pytest.raises(ValidationError, match="family"):
            Settings(_env_file=None)  # type: ignore[call-arg]

    def test_non_qwen_defaults_are_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGFT_PROVIDER", "ollama")
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        assert settings.provider is Provider.OLLAMA
        assert "qwen" not in settings.judge_model.lower()

    def test_the_student_model_is_qwen(self) -> None:
        """Guards the premise of the checks above."""
        assert "qwen" in BASE_MODEL.lower()


class TestOpenAIPath:
    def test_openai_without_key_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RAGFT_PROVIDER", "openai")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValidationError, match="OPENAI_API_KEY"):
            Settings(_env_file=None)  # type: ignore[call-arg]


class TestShippedConfigsLoad:
    """The committed YAML must actually validate against the models."""

    def test_retrieval_yaml_loads(self) -> None:
        assert load_retrieval_config().top_k_context == 5

    def test_train_yaml_loads(self) -> None:
        cfg = load_train_config()
        assert cfg.model_name == BASE_MODEL
        assert cfg.lora.r == 16
