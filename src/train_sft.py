from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.dataset import load_math_step_dpo
from src.inject_lora import inject_lora
from src.lora import count_trainable_parameters, mark_only_lora_as_trainable, save_lora_adapters


def build_sft_text(prompt: str, answer: str) -> str:
    return f"Question:\n{prompt}\n\nAnswer:\n{answer}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA SFT on math preference data.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--samples", type=int, default=0, help="Number of examples to use; 0 means full dataset.")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=64.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--data_file", default=None, help="Optional local Math-Step-DPO parquet file.")
    parser.add_argument("--output_dir", default="outputs/sft_lora")
    parser.add_argument("--log_dir", default="runs/sft", help="TensorBoard log directory.")
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=resolve_torch_dtype(args.dtype),
    ).to(device)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    target_modules = ("q_proj", "v_proj")
    replaced = inject_lora(model, target_modules=target_modules, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    mark_only_lora_as_trainable(model)
    trainable, total = count_trainable_parameters(model)
    print(f"Injected LoRA into: {replaced[:8]}{'...' if len(replaced) > 8 else ''}")
    print(f"Trainable parameters: {trainable}/{total} ({trainable / total:.2%})")

    examples = load_math_step_dpo(sample_count=args.samples, data_file=args.data_file)
    texts = [build_sft_text(example.prompt, example.chosen) for example in examples]
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    model.train()
    writer = SummaryWriter(log_dir=args.log_dir)

    loss_rows: list[dict[str, float | int]] = []
    step = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        for epoch in range(args.epochs):
            progress = tqdm(range(0, len(texts), args.batch_size), desc=f"epoch {epoch + 1}")
            for start in progress:
                batch_texts = texts[start : start + args.batch_size]
                batch = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                ).to(device)
                labels = batch["input_ids"].clone()
                labels[batch["attention_mask"].eq(0)] = -100
                loss = model(**batch, labels=labels).loss / args.gradient_accumulation_steps
                loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                raw_loss = float(loss.detach().cpu()) * args.gradient_accumulation_steps
                progress.set_postfix(loss=f"{raw_loss:.4f}")
                writer.add_scalar("train/loss", raw_loss, step)
                writer.add_scalar("train/epoch", epoch + 1, step)
                loss_rows.append({"step": step, "loss": raw_loss})
                step += 1
    finally:
        writer.close()

    save_lora_adapters(
        model,
        output_dir,
        target_modules=target_modules,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    tokenizer.save_pretrained(output_dir)
    with Path("outputs/loss_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(loss_rows)


if __name__ == "__main__":
    main()
