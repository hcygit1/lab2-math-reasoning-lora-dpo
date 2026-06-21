import torch

from src.dpo_loss import dpo_loss


def test_dpo_loss_is_lower_when_policy_prefers_chosen_more_than_reference():
    good = dpo_loss(
        policy_chosen_logps=torch.tensor([3.0]),
        policy_rejected_logps=torch.tensor([0.0]),
        reference_chosen_logps=torch.tensor([1.0]),
        reference_rejected_logps=torch.tensor([0.0]),
        beta=1.0,
    )
    bad = dpo_loss(
        policy_chosen_logps=torch.tensor([0.0]),
        policy_rejected_logps=torch.tensor([3.0]),
        reference_chosen_logps=torch.tensor([1.0]),
        reference_rejected_logps=torch.tensor([0.0]),
        beta=1.0,
    )

    assert good.item() < bad.item()
