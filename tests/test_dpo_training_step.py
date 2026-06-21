import torch

from src.dpo_loss import dpo_loss


def test_dpo_loss_backpropagates_to_policy_logps_only():
    policy_chosen = torch.tensor([2.0, 1.5], requires_grad=True)
    policy_rejected = torch.tensor([0.5, 0.2], requires_grad=True)
    reference_chosen = torch.tensor([1.0, 1.0])
    reference_rejected = torch.tensor([0.3, 0.3])

    loss = dpo_loss(
        policy_chosen_logps=policy_chosen,
        policy_rejected_logps=policy_rejected,
        reference_chosen_logps=reference_chosen,
        reference_rejected_logps=reference_rejected,
        beta=0.1,
    )
    loss.backward()

    assert policy_chosen.grad is not None
    assert policy_rejected.grad is not None
    assert reference_chosen.grad is None
    assert reference_rejected.grad is None
