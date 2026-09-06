"""Configuration contract for rag-vs-finetune.

Two kinds of configuration, kept deliberately apart:

* :class:`Settings` — deployment concerns (secrets, endpoints, log format).
  Comes from the environment. Never committed.
* The ``*Config`` models — pipeline behaviour (chunk size, top-k, LoRA rank,
  thresholds). Comes from committed YAML under ``configs/``.

The split exists for evaluation. A YAML config is a reproducible, diffable,
citable description of one experimental arm, so a row in a results table traces
to an exact configuration. A secret must never end up in something committed,
and a benchmark must never depend on an undeclared local environment variable.

Every model sets ``extra="forbid"``. The failure mode this prevents is specific
and nasty: a misspelled key is silently ignored, the run completes, and the
resulting number is attributed to a configuration that was never applied.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"

# The student model. Locked: every arm uses this same checkpoint so the
# comparison isolates the adaptation method rather than confounding it with a
# model swap.
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Families that must never generate training data or judge outputs, because the
# student belongs to one of them. A same-family generator distils its own style
# into the fine-tuned arm; a same-family judge scores its own family
# favourably. Both are disqualifying confounds, so this is enforced in code
# rather than left to a note in a README.
FORBIDDEN_JUDGE_FAMILIES = ("qwen",)


class Provider(StrEnum):
    """Where generation and judging calls go."""

    OLLAMA = "ollama"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Deployment settings (environment)
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Secrets and endpoints. Sourced from the environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # the environment legitimately holds unrelated vars
        case_sensitive=False,
    )

    provider: Provider = Field(default=Provider.OLLAMA, alias="RAGFT_PROVIDER")

    # --- Ollama (free path) ---
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_generation_model: str = Field(default="gemma4:e4b", alias="OLLAMA_GENERATION_MODEL")
    ollama_judge_model: str = Field(default="gemma4:e4b", alias="OLLAMA_JUDGE_MODEL")
    ollama_judge2_model: str = Field(default="llama3:8b", alias="OLLAMA_JUDGE2_MODEL")

    # --- OpenAI (optional ~$2 path; models match VidyaRAG's) ---
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_generation_model: str = Field(default="gpt-4o-mini", alias="OPENAI_GENERATION_MODEL")
    openai_judge_model: str = Field(default="gpt-4o-mini", alias="OPENAI_JUDGE_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )

    # --- Embeddings (free path) ---
    fastembed_model: str = Field(default="BAAI/bge-large-en-v1.5", alias="FASTEMBED_MODEL")

    # --- Qdrant ---
    qdrant_mode: Literal["embedded", "server", "cloud"] = Field(
        default="embedded", alias="QDRANT_MODE"
    )
    qdrant_path: str = Field(default="data/index", alias="QDRANT_PATH")
    qdrant_url: str | None = Field(default=None, alias="QDRANT_URL")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="ragft_indialaw_v1", alias="QDRANT_COLLECTION")

    # --- Tracking ---
    wandb_project: str = Field(default="rag-vs-finetune", alias="WANDB_PROJECT")
    wandb_entity: str | None = Field(default=None, alias="WANDB_ENTITY")

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_format: Literal["json", "console"] = Field(default="console", alias="LOG_FORMAT")

    @property
    def generation_model(self) -> str:
        """The model that writes synthetic QA pairs."""
        if self.provider is Provider.OPENAI:
            return self.openai_generation_model
        return self.ollama_generation_model

    @property
    def judge_model(self) -> str:
        """The model that scores answers."""
        if self.provider is Provider.OPENAI:
            return self.openai_judge_model
        return self.ollama_judge_model

    @model_validator(mode="after")
    def _check_provider_usable(self) -> Self:
        if self.provider is Provider.OPENAI and not self.openai_api_key:
            raise ValueError(
                "RAGFT_PROVIDER=openai but OPENAI_API_KEY is unset. "
                "Either set the key or use the free default RAGFT_PROVIDER=ollama."
            )
        return self

    @model_validator(mode="after")
    def _check_no_family_confound(self) -> Self:
        """Refuse to run with a generator or judge from the student's family.

        This is a correctness check, not a style preference: the benchmark's
        conclusions are invalid if it fails.
        """
        for role, name in (("generation", self.generation_model), ("judge", self.judge_model)):
            lowered = name.lower()
            for family in FORBIDDEN_JUDGE_FAMILIES:
                if family in lowered:
                    raise ValueError(
                        f"{role} model {name!r} is from the {family!r} family, which is also "
                        f"the student model's family ({BASE_MODEL}). A same-family generator "
                        f"distils its own style into the fine-tuned arm and a same-family judge "
                        f"scores its own family favourably. Pick a different model."
                    )
        return self


# ---------------------------------------------------------------------------
# Pipeline configuration (committed YAML)
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base for every YAML-backed config. A typo is a startup error."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingConfig(StrictModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    respect_sentence_boundaries: bool = True


class RetrievalConfig(StrictModel):
    """Mirrors VidyaRAG's frozen ``baseline`` profile.

    These values are not chosen here — they are copied from VidyaRAG so the
    retrieval arms of the two projects are the same pipeline rather than a
    lookalike. ``upstream_commit`` records what was mirrored, and a test
    asserts the values still match. If VidyaRAG changes its baseline, that test
    fails loudly instead of the comparison silently drifting.
    """

    upstream_repo: str = "NehaBharti08/VidyaRAG"
    upstream_profile: str = "baseline"
    upstream_commit: str | None = None

    chunking: ChunkingConfig = ChunkingConfig()
    top_k_retrieve: int = 20
    top_k_context: int = 5
    use_hybrid: bool = False
    use_reranker: bool = False
    use_decomposition: bool = False

    @model_validator(mode="after")
    def _pool_wider_than_selection(self) -> Self:
        if self.top_k_retrieve < self.top_k_context:
            raise ValueError(
                f"top_k_retrieve ({self.top_k_retrieve}) must be >= top_k_context "
                f"({self.top_k_context}): a reranker can only add value when given a "
                f"candidate pool wider than the final selection."
            )
        return self


class LoraConfig(StrictModel):
    """LoRA hyperparameters.

    ``alpha`` is expressed as a ratio to ``r`` rather than an absolute number.
    That is deliberate: LoRA's update is scaled by alpha/r, so holding the
    *ratio* fixed keeps the effective update magnitude constant when r changes
    during the sweep. The learning rate then needs no re-tuning per rank, and
    the r=16 vs r=64 comparison measures rank rather than a confounded change
    in effective learning rate.
    """

    r: int = 16
    alpha_ratio: float = 2.0
    dropout: float = 0.05
    # All seven linear projections, not attention only. The QLoRA paper's
    # central ablation: adapting every linear layer is what closes the gap to
    # full fine-tuning. Adapters are ~0.5% of parameters, so the extra cost is
    # noise.
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    @property
    def alpha(self) -> int:
        return int(self.r * self.alpha_ratio)


class TrainConfig(StrictModel):
    model_name: str = BASE_MODEL
    lora: LoraConfig = LoraConfig()

    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    num_train_epochs: int = 3

    # Effective batch = per_device * grad_accum = 16.
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8

    # Provisional. Phase 1 measures the p99 token length of
    # (passage + question + formatted answer) and sets this from the data.
    # Padding to 2048 when the data is 600 tokens wastes ~70% of the budget.
    max_seq_length: int = 2048
    packing: bool = True

    # ~30% slower, but it creates the VRAM headroom that keeps a job alive when
    # another user takes GPU memory on this shared box.
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    seed: int = 42

    # Checkpointing every epoch is what makes the epoch axis free: the sweep
    # evaluates epochs 1/2/3 from a single run instead of launching one run per
    # epoch count, collapsing a 12-run grid to 4.
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


class QuantConfig(StrictModel):
    """4-bit NF4 quantization.

    Applied identically to *every* arm, including the base-model arms. If the
    fine-tuned arms ran 4-bit and the base arms ran bf16, quantization would be
    confounded with adaptation and no cell of the 2x2 would be interpretable.
    """

    load_in_4bit: bool = True
    quant_type: Literal["nf4", "fp4"] = "nf4"
    compute_dtype: Literal["bfloat16", "float16"] = "bfloat16"
    double_quant: bool = True


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Read a YAML config file into a plain dict."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: expected a YAML mapping, got {type(data).__name__}")
    return data


def load_retrieval_config(path: Path | str | None = None) -> RetrievalConfig:
    """Load the retrieval config, defaulting to ``configs/retrieval.yaml``."""
    path = Path(path) if path is not None else CONFIG_DIR / "retrieval.yaml"
    return RetrievalConfig.model_validate(load_yaml(path))


def load_train_config(path: Path | str | None = None) -> TrainConfig:
    """Load a training config, defaulting to ``configs/train/base.yaml``."""
    path = Path(path) if path is not None else CONFIG_DIR / "train" / "base.yaml"
    return TrainConfig.model_validate(load_yaml(path))
