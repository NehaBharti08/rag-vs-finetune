"""Phase 0 gate: prove the QLoRA stack works, and measure it.

This script exists because every compute estimate in the project plan is
arithmetic until something actually runs. It answers three questions that
cannot be answered from a spreadsheet:

1. Does the pinned dependency matrix actually load a 4-bit Qwen2.5-7B and take
   an optimizer step? (bitsandbytes 0.50 requires ``transformers<5``; trl 1.9
   requires ``>=4.56.2``; the machine's base env ships 5.14.1 on Python 3.13.)
2. What is the real throughput in tokens/sec on this A5000?
3. What is the real peak VRAM, and how much headroom is left on a shared GPU?

Its output is written to ``reports/env_matrix.md`` and every GPU-hour estimate
downstream is recalibrated against it.

Usage::

    uv run python -m ragft.train.smoke --steps 20
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ragft.settings import REPO_ROOT, LoraConfig, QuantConfig, TrainConfig
from ragft.train.callbacks import StepTimer

# Bytes per GiB, used for every memory figure reported below.
GIB = 1024**3


def build_dummy_dataset(n: int, tokenizer: Any, target_tokens: int) -> Dataset:
    """Synthetic text roughly ``target_tokens`` long.

    Deliberately not real corpus text: this measures the stack, not the data.
    Using prose of the right *length* is what matters, since throughput is
    driven by sequence length rather than content.
    """
    sentence = (
        "Osmosis is the diffusion of water across a semipermeable membrane from a region "
        "of lower solute concentration to one of higher solute concentration. "
    )
    # Measure once, then repeat to length, rather than guessing a token ratio.
    per_sentence = len(tokenizer(sentence)["input_ids"])
    repeats = max(1, target_tokens // max(1, per_sentence))
    return Dataset.from_dict({"text": [sentence * repeats for _ in range(n)]})


def gpu_snapshot() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_gib": round(props.total_memory / GIB, 2),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def run_smoke(steps: int, seq_len: int, out_dir: Path) -> dict[str, Any]:
    train_cfg = TrainConfig(max_seq_length=seq_len)
    quant_cfg = QuantConfig()
    lora_cfg: LoraConfig = train_cfg.lora

    torch.cuda.reset_peak_memory_stats()

    tokenizer = AutoTokenizer.from_pretrained(train_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg.load_in_4bit,
        bnb_4bit_quant_type=quant_cfg.quant_type,
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg.compute_dtype),
        bnb_4bit_use_double_quant=quant_cfg.double_quant,
    )

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        train_cfg.model_name,
        quantization_config=bnb_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    load_seconds = time.perf_counter() - load_start
    weights_gib = torch.cuda.memory_allocated() / GIB

    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=train_cfg.gradient_checkpointing
    )

    peft_config = PeftLoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.alpha,
        lora_dropout=lora_cfg.dropout,
        target_modules=list(lora_cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    dataset = build_dummy_dataset(
        n=steps * train_cfg.effective_batch_size + train_cfg.effective_batch_size,
        tokenizer=tokenizer,
        target_tokens=seq_len,
    )

    sft_config = SFTConfig(
        output_dir=str(out_dir / "smoke_run"),
        max_steps=steps,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        # NOTE: trl 1.9 renamed this from `max_seq_length` to `max_length`.
        # Verified by introspecting SFTConfig rather than assumed.
        max_length=seq_len,
        packing=train_cfg.packing,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=train_cfg.optim,
        bf16=train_cfg.bf16,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=train_cfg.seed,
    )

    timer = StepTimer()
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        callbacks=[timer],
    )

    train_start = time.perf_counter()
    result = trainer.train()
    train_seconds = time.perf_counter() - train_start

    peak_gib = torch.cuda.max_memory_allocated() / GIB
    reserved_gib = torch.cuda.max_memory_reserved() / GIB
    gpu = gpu_snapshot()

    # With packing every sequence is a full `seq_len` block, so token count is
    # exact rather than an estimate.
    tokens_per_step = train_cfg.effective_batch_size * seq_len
    tokens_per_sec = tokens_per_step / timer.mean_step_seconds if timer.timings else float("nan")

    return {
        "verdict": "PASS",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "gpu": gpu,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "trl": __import__("trl").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
            "accelerate": __import__("accelerate").__version__,
        },
        "config": {
            "model": train_cfg.model_name,
            "quant": f"{quant_cfg.quant_type} double={quant_cfg.double_quant}",
            "lora_r": lora_cfg.r,
            "lora_alpha": lora_cfg.alpha,
            "target_modules": list(lora_cfg.target_modules),
            "seq_len": seq_len,
            "per_device_batch": train_cfg.per_device_train_batch_size,
            "grad_accum": train_cfg.gradient_accumulation_steps,
            "effective_batch": train_cfg.effective_batch_size,
            "gradient_checkpointing": train_cfg.gradient_checkpointing,
            "optim": train_cfg.optim,
            "packing": train_cfg.packing,
        },
        "params": {
            "trainable": trainable,
            "total": total,
            "trainable_pct": round(100 * trainable / total, 4),
        },
        "measurements": {
            "steps_timed": len(timer.timings),
            "model_load_seconds": round(load_seconds, 1),
            "weights_only_gib": round(weights_gib, 2),
            "peak_allocated_gib": round(peak_gib, 2),
            "peak_reserved_gib": round(reserved_gib, 2),
            "headroom_gib": round(gpu.get("total_gib", 0) - reserved_gib, 2),
            "mean_step_seconds": round(timer.mean_step_seconds, 3),
            "tokens_per_second": round(tokens_per_sec, 1),
            "tokens_per_step": tokens_per_step,
            "total_train_seconds": round(train_seconds, 1),
            "final_loss": round(float(result.training_loss), 4),
        },
    }


def project_costs(
    m: dict[str, Any],
    train_examples: int = 4000,
    avg_tokens_per_example: int = 600,
) -> dict[str, Any]:
    """Turn measured throughput into the GPU-hour budget for later phases.

    The token count must be driven by the *average example length*, not by
    ``seq_len``. With packing enabled, short examples are concatenated to fill
    each ``seq_len`` block, so 4000 examples of ~600 tokens occupy roughly 1200
    sequences of 2048 -- not 4000 of them.

    Multiplying ``train_examples * seq_len`` instead (the obvious-looking
    formula) overstates the epoch by the packing ratio, which at these settings
    is ~3.4x. That is the difference between a ~14 GPU-hour training budget and
    a ~47 GPU-hour one, so it is worth being precise about.

    ``avg_tokens_per_example`` is an ESTIMATE until Phase 1 measures the real
    token-length distribution of (passage + question + formatted answer). The
    same phase sets ``max_seq_length`` from the measured p99.
    """
    tps = m["measurements"]["tokens_per_second"]
    tokens_per_epoch = train_examples * avg_tokens_per_example
    hours_per_epoch = tokens_per_epoch / tps / 3600
    return {
        "assumed_train_examples": train_examples,
        "assumed_avg_tokens_per_example": avg_tokens_per_example,
        "avg_tokens_is_estimate_until_phase1": True,
        "tokens_per_epoch": tokens_per_epoch,
        "hours_per_epoch": round(hours_per_epoch, 2),
        "hours_per_3epoch_run": round(3 * hours_per_epoch, 2),
        "sweep_4_configs_1_epoch": round(4 * hours_per_epoch, 2),
        "final_3_seeds_3_epochs": round(9 * hours_per_epoch, 2),
        "training_total": round(13 * hours_per_epoch, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = run_smoke(steps=args.steps, seq_len=args.seq_len, out_dir=args.out)
    results["projection"] = project_costs(results)

    json_path = args.out / "smoke_results.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nWrote {json_path}")


if __name__ == "__main__":
    main()
