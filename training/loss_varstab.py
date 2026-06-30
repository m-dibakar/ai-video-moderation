import torch
import torch.nn as nn
import torch.nn.functional as F


class VarianceStabilizedLoss(nn.Module):
    """
    Placeholder for Variance-Stabilized Loss Function.
    IEEE Signal Processing Letters — Dibakar Malakar

    Replace forward() with your actual published formulation later.
    Current implementation: weighted BCE with variance stabilization proxy.
    """
    def __init__(self, eps=1e-6, smooth=0.1, adaptive_weighting=True):
        super().__init__()
        self.eps = eps
        self.smooth = smooth
        self.adaptive_weighting = adaptive_weighting

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smooth) + 0.5 * self.smooth
        probs = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none'
        )
        bernoulli_var = probs * (1 - probs) + self.eps
        stabilizer = 1.0 / torch.sqrt(bernoulli_var)
        stabilizer = stabilizer / (stabilizer.mean() + self.eps)
        stabilized_loss = bce * stabilizer

        if self.adaptive_weighting:
            pos_count = targets.sum() + self.eps
            neg_count = (1 - targets).sum() + self.eps
            total = pos_count + neg_count
            w_pos = total / (2 * pos_count)
            w_neg = total / (2 * neg_count)
            weights = targets * w_pos + (1 - targets) * w_neg
            stabilized_loss = stabilized_loss * weights

        return stabilized_loss.mean()
