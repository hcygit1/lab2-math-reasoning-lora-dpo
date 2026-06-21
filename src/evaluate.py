from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.lora import load_lora_adapters


PROMPTS = [
    "If 3x + 5 = 20, what is x? Show the steps.",
    "A rectangle has length 12 and width 7. What is its area?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare base and LoRA model outputs.")
    parser.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_dir", default="outputs/sft_lora")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    return parser.parse_args()


def generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("outputs").mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True).to(device).eval()
    lora = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    load_lora_adapters(lora, args.adapter_dir)
    lora = lora.to(device).eval()

    sections = ["# Base vs SFT-LoRA Output Comparison\n"]
    for prompt in PROMPTS:
        sections.append(f"## Prompt\n\n{prompt}\n")
        sections.append("### Base\n\n" + generate(base, tokenizer, prompt, args.max_new_tokens, device) + "\n")
        sections.append("### SFT-LoRA\n\n" + generate(lora, tokenizer, prompt, args.max_new_tokens, device) + "\n")

    Path("outputs/base_vs_lora.md").write_text("\n".join(sections), encoding="utf-8")
    print("Wrote outputs/base_vs_lora.md")


if __name__ == "__main__":
    main()
