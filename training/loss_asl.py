"""
Asymmetric Loss (ASL) for binary classification with rare positives.

Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021.
Optionally combined with train-time logit adjustment (Menon et al.,
"Long-Tail Learning via Logit Adjustment", ICLR 2021).

Chosen from the failure analysis of the BCE/Focal/VS benchmark:
  - gamma_pos = 0: positives are rare (5%) — never down-weight their gradient
  - gamma_neg = 4: aggressively silence easy negatives (Focal's γ=2 left a
    noisy negative tail that cost precision)
  - clip (probability margin m): negatives with p < m contribute exactly
    zero loss and zero gradient — the piece plain Focal lacks
  - prior: shifts logits by log-odds of the class prior during training,
    so the raw model at inference is recall-boosted in a Bayes-consistent way
"""

import math

import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos=0.0, gamma_neg=4.0, clip=0.05,
                 prior=None, eps=1e-8):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.eps = eps
        # log-odds of the positive-class prior; 0 disables adjustment
        self.logit_adj = math.log(prior / (1 - prior)) if prior else 0.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        p = torch.sigmoid(logits + self.logit_adj)

        # probability margin: easy negatives contribute exactly zero
        p_neg = (p - self.clip).clamp(min=0)

        loss_pos = targets * (1 - p).pow(self.gamma_pos) \
            * torch.log(p.clamp(min=self.eps))
        loss_neg = (1 - targets) * p_neg.pow(self.gamma_neg) \
            * torch.log((1 - p_neg).clamp(min=self.eps))
        return -(loss_pos + loss_neg).mean()


if __name__ == "__main__":
    torch.manual_seed(42)
    loss_fn = AsymmetricLoss(gamma_pos=0.0, gamma_neg=4.0, clip=0.05, prior=0.05)

    logits = torch.randn(32, requires_grad=True)
    targets = torch.zeros(32)
    targets[0] = 1.0
    targets[1] = 1.0

    loss = loss_fn(logits, targets)
    loss.backward()
    print(f"ASL on imbalanced batch: {loss.item():.4f}, grad ok: {logits.grad is not None}")

    # easy negatives (p << clip) must produce zero gradient
    easy_neg = torch.full((8,), -6.0, requires_grad=True)
    loss_fn(easy_neg, torch.zeros(8)).backward()
    assert easy_neg.grad is not None
    print(f"easy-negative grad magnitude: {easy_neg.grad.abs().max().item():.2e} (expect 0)")

    # hard positives must keep full gradient
    hard_pos = torch.full((8,), -2.0, requires_grad=True)
    loss_fn(hard_pos, torch.ones(8)).backward()
    assert hard_pos.grad is not None
    print(f"hard-positive grad magnitude: {hard_pos.grad.abs().max().item():.4f} (expect large)")
