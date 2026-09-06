"""The four arms of the 2x2, sharing exactly one generation path.

The grid crosses two binary factors:

                   no retrieval        with retrieval
    base           A1 zero-shot        A2 standard RAG
    fine-tuned     A3 parametric       A4 fine-tune + RAG

An "arm" is nothing but a configuration: which adapter (if any) is attached,
and whether retrieved context is prepended. There is one `generate()` and every
arm goes through it, so any difference between two cells is the adaptation
method and not an accident of plumbing. `tests/test_arms_parity.py` enforces
this by asserting all four share identical decoding and quantization settings.

Two decisions here are load-bearing enough to state:

**Every arm runs 4-bit NF4, including the base arms.** If the fine-tuned arms
were quantized and the base arms ran bf16, quantization would be confounded
with adaptation and no cell would be interpretable (threat 3).

**Base arms get few-shot format examples; fine-tuned arms get none.** That
looks like an asymmetry and it is - in the base arms' favour. The fine-tuned
model was trained on the format, so examples would be redundant; the base model
was not, so withholding them would manufacture a format win for fine-tuning out
of prompt engineering rather than adaptation (threat 2). Effort is equalised in
the direction that makes the comparison honest rather than flattering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from ragft.settings import BASE_MODEL, REPO_ROOT, QuantConfig

PROMPT_DIR = REPO_ROOT / "prompts" / "inference"

# Decoding is identical across arms and deterministic. Sampling would add
# variance to a comparison whose whole point is small measured differences.
MAX_NEW_TOKENS = 320
DO_SAMPLE = False


@dataclass(frozen=True)
class ArmSpec:
    """One cell of the 2x2."""

    name: str
    label: str
    adapter_path: str | None
    use_retrieval: bool
    prompt_file: str

    @property
    def is_finetuned(self) -> bool:
        return self.adapter_path is not None


def arm_specs(adapter_path: str | None = None) -> list[ArmSpec]:
    """The four arms. Fine-tuned arms are only available once an adapter exists."""
    arms = [
        ArmSpec("A1_base_zeroshot", "base, no retrieval", None, False, "base_zeroshot"),
        ArmSpec("A2_base_rag", "base + RAG", None, True, "base_rag"),
    ]
    if adapter_path:
        arms += [
            ArmSpec(
                "A3_ft_zeroshot", "fine-tuned, no retrieval", adapter_path, False, "ft_zeroshot"
            ),
            ArmSpec("A4_ft_rag", "fine-tuned + RAG", adapter_path, True, "ft_rag"),
        ]
    return arms


def load_prompt(name: str) -> str:
    return (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")


def build_prompt(arm: ArmSpec, question: str, context: str = "") -> str:
    template = load_prompt(arm.prompt_file)
    fields: dict[str, str] = {"question": question}
    if "{format_block}" in template:
        fields["format_block"] = load_prompt("_format").strip()
    if "{context}" in template:
        fields["context"] = context
    return template.format(**fields)


class ArmRunner:
    """Loads the base model once; swaps adapters in place.

    Reloading a 4-bit 7B per arm would cost ~13 s each and, more importantly,
    would make it easy for the arms to drift apart in load configuration. One
    model, one quantization config, adapters attached and detached.
    """

    def __init__(self, quant: QuantConfig | None = None) -> None:
        self.quant = quant or QuantConfig()
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb = BitsAndBytesConfig(
            load_in_4bit=self.quant.load_in_4bit,
            bnb_4bit_quant_type=self.quant.quant_type,
            bnb_4bit_compute_dtype=getattr(torch, self.quant.compute_dtype),
            bnb_4bit_use_double_quant=self.quant.double_quant,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb, dtype=torch.bfloat16, device_map={"": 0}
        )
        self.model.eval()
        self._adapter: str | None = None

    def set_adapter(self, adapter_path: str | None) -> None:
        """Attach or detach a LoRA adapter without reloading the base weights."""
        if adapter_path == self._adapter:
            return
        if adapter_path is None:
            if hasattr(self.model, "unload"):
                self.model = self.model.unload()
        else:
            from peft import PeftModel

            base = self.model.unload() if hasattr(self.model, "unload") else self.model
            self.model = PeftModel.from_pretrained(base, adapter_path)
            self.model.eval()
        self._adapter = adapter_path

    @torch.inference_mode()
    def generate(self, prompt: str) -> dict[str, Any]:
        """The single generation path. Every arm calls exactly this."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        torch.cuda.synchronize()
        latency = time.perf_counter() - t0

        generated = out[0][inputs["input_ids"].shape[1] :]
        return {
            "response": self.tokenizer.decode(generated, skip_special_tokens=True).strip(),
            "latency_seconds": round(latency, 4),
            "prompt_tokens": int(inputs["input_ids"].shape[1]),
            "completion_tokens": int(generated.shape[0]),
        }


def decoding_config() -> dict[str, Any]:
    """Exposed so a test can assert every arm shares it."""
    return {
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": DO_SAMPLE,
        "model": BASE_MODEL,
        "quant": QuantConfig().model_dump(),
    }


_ = Path
