from __future__ import annotations

from torch import nn

from src.lora import LoRALinear


def _get_parent_module(model: nn.Module, module_path: str) -> tuple[nn.Module, str]:
    parts = module_path.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
) -> list[str]:
    """Replace matching Linear modules with LoRA-wrapped modules."""

    replaced: list[str] = []
    named_modules = list(model.named_modules())
    for name, module in named_modules:
        if not name:
            continue
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(target) for target in target_modules):
            continue

        parent, child_name = _get_parent_module(model, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout))
        replaced.append(name)

    if not replaced:
        raise ValueError(f"No target Linear modules matched {target_modules}")
    return replaced
