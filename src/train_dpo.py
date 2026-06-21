from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from torch.optim import AdamW
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import load_math_step_dpo
from src.dpo_loss import dpo_loss, sequence_log_probs
from src.inject_lora import inject_lora
from src.lora import (
    count_trainable_parameters,
    load_lora_adapters,
    mark_only_lora_as_trainable,
    save_lora_adapters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run small-scale DPO training with custom LoRA adapters.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--init_adapter_dir", default="outputs/sft_lora")
    parser.add_argument("--output_dir", default="outputs/dpo_lora")
    parser.add_argument("--data_file", default=None, help="Optional local Math-Step-DPO parquet file.")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    return parser.parse_args()


def format_pair_text(prompt: str, answer: str) -> str:
    return f"Question:\n{prompt}\n\nAnswer:\n{answer}"


def batch_sequence_logps(model, tokenizer, texts: list[str], max_length: int, device: str) -> torch.Tensor:
    batch = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)
    labels = batch["input_ids"].clone()
    labels[batch["attention_mask"].eq(0)] = -100
    logits = model(**batch).logits
    return sequence_log_probs(logits, labels)


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    adapter_path = Path(args.init_adapter_dir)
    if adapter_path.exists():
        replaced = load_lora_adapters(policy, adapter_path)
        print(f"Loaded initial LoRA adapters from {adapter_path}")
    else:
        target_modules = ("q_proj", "v_proj")
        replaced = inject_lora(
            policy,
            target_modules=target_modules,
            rank=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
        )
        print(f"Initialized fresh LoRA adapters because {adapter_path} does not exist")
    mark_only_lora_as_trainable(policy)
    policy = policy.to(device).train()

    reference = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False

    trainable, total = count_trainable_parameters(policy)
    print(f"Policy LoRA modules: {replaced[:8]}{'...' if len(replaced) > 8 else ''}")
    print(f"Trainable parameters: {trainable}/{total} ({trainable / total:.2%})")

    examples = load_math_step_dpo(sample_count=args.samples, data_file=args.data_file)
    chosen_texts = [format_pair_text(example.prompt, example.chosen) for example in examples]
    rejected_texts = [format_pair_text(example.prompt, example.rejected) for example in examples]

    optimizer = AdamW((p for p in policy.parameters() if p.requires_grad), lr=args.lr)
    loss_rows: list[dict[str, float | int]] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(args.epochs):
        progress = tqdm(range(0, len(examples), args.batch_size), desc=f"dpo epoch {epoch + 1}")
        for start in progress:
            chosen_batch = chosen_texts[start : start + args.batch_size]
            rejected_batch = rejected_texts[start : start + args.batch_size]

            policy_chosen = batch_sequence_logps(policy, tokenizer, chosen_batch, args.max_length, device)
            policy_rejected = batch_sequence_logps(policy, tokenizer, rejected_batch, args.max_length, device)
            with torch.no_grad():
                reference_chosen = batch_sequence_logps(reference, tokenizer, chosen_batch, args.max_length, device)
                reference_rejected = batch_sequence_logps(reference, tokenizer, rejected_batch, args.max_length, device)

            loss = dpo_loss(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=args.beta,
            )
            (loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            raw_loss = float(loss.detach().cpu())
            progress.set_postfix(loss=f"{raw_loss:.4f}")
            loss_rows.append({"step": step, "loss": raw_loss})
            step += 1

    if step % args.gradient_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    save_lora_adapters(
        policy,
        output_dir,
        target_modules=("q_proj", "v_proj"),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    tokenizer.save_pretrained(output_dir)
    with Path("outputs/dpo_loss_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(loss_rows)


if __name__ == "__main__":
    main()
