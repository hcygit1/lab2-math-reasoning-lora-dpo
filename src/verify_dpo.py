from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import load_math_step_dpo
from src.dpo_loss import dpo_loss, sequence_log_probs
from src.lora import load_lora_adapters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DPO loss on a small math preference batch.")
    parser.add_argument("--adapter_dir", default="outputs/sft_lora")
    parser.add_argument("--reference_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--beta", type=float, default=0.1)
    return parser.parse_args()


def batch_logps(model, tokenizer, texts: list[str], max_length: int, device: str) -> torch.Tensor:
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"].eq(0)] = -100
    with torch.no_grad():
        logits = model(**batch).logits
    return sequence_log_probs(logits, labels)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path("outputs").mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.reference_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.reference_model, trust_remote_code=True)
    load_lora_adapters(policy, args.adapter_dir)
    policy = policy.to(device).eval()
    reference = AutoModelForCausalLM.from_pretrained(args.reference_model, trust_remote_code=True).to(device).eval()

    examples = load_math_step_dpo(sample_count=args.samples)
    chosen_texts = [f"Question:\n{example.prompt}\n\nAnswer:\n{example.chosen}" for example in examples]
    rejected_texts = [f"Question:\n{example.prompt}\n\nAnswer:\n{example.rejected}" for example in examples]

    policy_chosen = batch_logps(policy, tokenizer, chosen_texts, args.max_length, device)
    policy_rejected = batch_logps(policy, tokenizer, rejected_texts, args.max_length, device)
    reference_chosen = batch_logps(reference, tokenizer, chosen_texts, args.max_length, device)
    reference_rejected = batch_logps(reference, tokenizer, rejected_texts, args.max_length, device)

    normal_loss = dpo_loss(
        policy_chosen,
        policy_rejected,
        reference_chosen,
        reference_rejected,
        beta=args.beta,
    )
    swapped_loss = dpo_loss(
        policy_rejected,
        policy_chosen,
        reference_rejected,
        reference_chosen,
        beta=args.beta,
    )
    report = (
        f"DPO normal loss: {normal_loss.item():.6f}\n"
        f"DPO swapped loss: {swapped_loss.item():.6f}\n"
        f"Batch size: {args.samples}\n"
    )
    Path("outputs/dpo_check.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
