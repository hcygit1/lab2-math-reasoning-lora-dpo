import torch

from src.lora import LoRALinear


def test_lora_linear_freezes_base_and_trains_adapters_only():
    base = torch.nn.Linear(4, 3, bias=False)
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)

    assert layer.base.weight.requires_grad is False
    assert layer.lora_a.weight.requires_grad is True
    assert layer.lora_b.weight.requires_grad is True


def test_lora_linear_matches_base_output_when_b_is_zero():
    torch.manual_seed(0)
    base = torch.nn.Linear(4, 3, bias=False)
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    x = torch.randn(2, 4)

    torch.testing.assert_close(layer(x), base(x))


def test_lora_linear_inherits_base_dtype():
    base = torch.nn.Linear(4, 3, bias=False, dtype=torch.float64)
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)

    assert layer.lora_a.weight.dtype == torch.float64
    assert layer.lora_b.weight.dtype == torch.float64
