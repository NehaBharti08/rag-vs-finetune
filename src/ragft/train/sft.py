"""QLoRA supervised fine-tuning. Phase 4.

Trains a LoRA adapter on the synthetic statutory QA set. Adapters only -- merged
7B weights are 15 GB and add nothing reproducible, so the artifact is the ~160 MB
adapter and the base checkpoint everyone already has.

Three choices here are load-bearing.

**Prompt-completion format, loss on the completion only.** The model should learn
to *produce* the three-part answer given a question, not to reproduce the
question. Training on the prompt tokens too would spend a thin gradient budget
teaching it to echo inputs.

**The same chat template as inference.** `ragft.eval.arms` calls
`apply_chat_template` with a single user turn; training uses the conversational
prompt-completion form so TRL applies that identical template. If these diverged,
the fine-tuned arms would be evaluated under a format they were never trained on
and the 2x2 would silently measure a template mismatch.

**Checkpoint every epoch.** This is what makes the epoch axis free: epochs 1/2/3
are read out of a single run rather than launching one run per epoch count. It
is also what makes the run resumable on a shared GPU that can lose its slot.

Usage::

    uv run python -m ragft.train.sft --seed 42
    uv run python -m ragft.train.sft --seed 42 --epochs 1 --lr 1e-4 --rank 64
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ragft.settings import REPO_ROOT, QuantConfig, load_train_config
from ragft.train.callbacks import StepTimer

QA_PATH = REPO_ROOT / "data" / "qa" / "clean.jsonl"
OUT_ROOT = REPO_ROOT / "out"


def load_split(split: str) -> Dataset:
    """Conversational prompt-completion pairs for one split.

    Returning `messages`-shaped fields lets TRL apply the model's own chat
    template, which is the same one `ragft.eval.arms` uses at inference.
    """
    rows = [json.loads(line) for line in QA_PATH.open(encoding="utf-8") if line.strip()]
    subset = [r for r in rows if r["split"] == split]
    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": r["question"]}],
                "completion": [{"role": "assistant", "content": r["formatted_answer"]}],
            }
            for r in subset
        ]
    )


def build_model(rank: int, quant: QuantConfig, cfg: Any) -> tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=quant.load_in_4bit,
        bnb_4bit_quant_type=quant.quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant.compute_dtype),
        bnb_4bit_use_double_quant=quant.double_quant,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg.gradient_checkpointing
    )
    model = get_peft_model(
        model,
        PeftLoraConfig(
            r=rank,
            # alpha is a RATIO to r, so effective update magnitude is constant
            # across ranks and the learning rate needs no re-tuning per rank.
            lora_alpha=int(rank * cfg.lora.alpha_ratio),
            lora_dropout=cfg.lora.dropout,
            target_modules=list(cfg.lora.target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    return model, tokenizer


def train(
    seed: int = 42,
    epochs: int | None = None,
    lr: float | None = None,
    rank: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    cfg = load_train_config()
    quant = QuantConfig()
    epochs = epochs if epochs is not None else cfg.num_train_epochs
    lr = lr if lr is not None else cfg.learning_rate
    rank = rank if rank is not None else cfg.lora.r
    name = run_name or f"seed{seed}_r{rank}_lr{lr:g}_e{epochs}"
    out_dir = OUT_ROOT / name

    # W&B offline by default: untracked runs do not exist, but this project has
    # no API key and must not require one. Offline runs are recorded locally
    # and can be synced later with `wandb sync`.
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "rag-vs-finetune-legal")

    train_ds, val_ds = load_split("train"), load_split("val")
    model, tokenizer = build_model(rank, quant, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    steps_per_epoch = max(1, len(train_ds) // cfg.effective_batch_size)
    print(f"[{name}] train {len(train_ds)} | val {len(val_ds)} | trainable {trainable:,}")
    print(f"[{name}] ~{steps_per_epoch} optimizer steps/epoch, {steps_per_epoch * epochs} total")

    args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        max_length=cfg.max_seq_length,
        # Loss on the answer only. The task is to produce the three-part
        # answer, not to reproduce the question.
        completion_only_loss=True,
        # Packing OFF, deliberately, reversing the value in base.yaml.
        #
        # Packing concatenates short examples into full seq_len blocks, which is
        # the right call when compute is the binding constraint. Here the
        # binding constraint is the opposite: 2,830 examples of ~137 tokens is a
        # thin training signal, flagged as the first thing to suspect if Phase 4
        # shows little movement.
        #
        # Packed, an epoch is ~95 optimizer steps. Unpacked it is ~353 - the
        # same data, roughly the same wall clock, but 3.7x more gradient
        # updates. On a small set more updates is what the model needs.
        #
        # It is also cleaner: completion-only masking and packing interact
        # awkwardly, and correctness of the loss mask matters more here than
        # squeezing padding out of 137-token sequences.
        packing=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=cfg.optim,
        bf16=cfg.bf16,
        seed=seed,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=epochs,
        load_best_model_at_end=False,
        report_to=["wandb"],
        run_name=name,
    )

    timer = StepTimer()
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[timer],
    )

    t0 = time.perf_counter()
    # Resume if a checkpoint exists: this box is shared and a run can lose its
    # slot at any point.
    resume = any(out_dir.glob("checkpoint-*")) if out_dir.exists() else False
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.perf_counter() - t0

    adapter_dir = out_dir / "adapter"
    assert trainer.model is not None
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    summary = {
        "run_name": name,
        "seed": seed,
        "epochs": epochs,
        "learning_rate": lr,
        "lora_rank": rank,
        "trainable_params": trainable,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "steps_per_epoch": steps_per_epoch,
        "elapsed_minutes": round(elapsed / 60, 1),
        "mean_step_seconds": round(timer.mean_step_seconds, 3),
        "final_train_loss": round(float(result.training_loss), 4),
        "eval_loss_by_epoch": [
            {"epoch": h.get("epoch"), "eval_loss": round(h["eval_loss"], 4)}
            for h in history
            if "eval_loss" in h
        ],
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
        "adapter_dir": str(adapter_dir.relative_to(REPO_ROOT)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    train(args.seed, args.epochs, args.lr, args.rank, args.run_name)
    _ = Path


if __name__ == "__main__":
    main()
