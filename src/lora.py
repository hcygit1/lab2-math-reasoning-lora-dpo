from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Wrap a frozen Linear layer with trainable low-rank adapters."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive")
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear only supports torch.nn.Linear")

        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.scale * self.lora_b(self.lora_a(self.dropout(inputs)))


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_a.weight.requires_grad = True
            module.lora_b.weight.requires_grad = True


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable, total


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def save_lora_adapters(
    model: nn.Module,
    output_dir: str | Path,
    target_modules: tuple[str, ...],
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(lora_state_dict(model), path / "adapter_model.pt")
    config = {
        "target_modules": list(target_modules),
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
    }
    (path / "adapter_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def load_lora_adapters(model: nn.Module, adapter_dir: str | Path) -> list[str]:
    from src.inject_lora import inject_lora

    path = Path(adapter_dir)
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    replaced = inject_lora(
        model,
        target_modules=tuple(config["target_modules"]),
        rank=int(config["rank"]),
        alpha=float(config["alpha"]),
        dropout=float(config.get("dropout", 0.0)),
    )
    state = torch.load(path / "adapter_model.pt", map_location="cpu")
    model.load_state_dict(state, strict=False)
    return replaced
