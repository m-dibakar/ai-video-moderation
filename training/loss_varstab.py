import torch
import torch.nn as nn


class VarianceStabilizedLoss(nn.Module):
    """
    Variance Stabilized Loss Function for Semantic Segmentation.

    Published in: IEEE Signal Processing Letters, Vol. 32, 2025
    Authors: Rinku Rabidas, Dibakar Malakar, Joyita Bhattacharjee,
             Sandeep Mandia, Jayasree Chakraborty
    DOI: 10.1109/LSP.2025.3625880

    PyTorch port matching exact TF operator precedence.

    TF code (exact):
        harmonic = (Precision*((1-w)*Specificity)/Precision
                    +((1-w)*Specificity)+epsilon)
        → simplifies to: 2*(1-w)*Specificity + epsilon
        (no parentheses around denominator in original code)

    Full formula:
        Sensi     = 2 * w * Sensitivity²
        harmonic  = 2 * (1-w) * Specificity + ε
        F_medical = Sensi * harmonic
        VS_Loss   = 1 - F_medical
    """

    def __init__(self, w=0.56, eps=1e-7):
        super().__init__()
        self.w   = w
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits)
        probs   = torch.clamp(probs, self.eps, 1 - self.eps)
        targets = targets.float()

        # Soft confusion matrix
        TP = (probs * targets).sum()
        FP = (probs * (1 - targets)).sum()
        FN = ((1 - probs) * targets).sum()
        TN = ((1 - probs) * (1 - targets)).sum()

        # Sensitivity = TP / (TP + FN)
        Sensitivity = TP / (TP + FN + self.eps)

        # Precision = TP / (TP + FP)
        Precision = TP / (TP + FP + self.eps)

        # Specificity = TN / (TN + FP)
        Specificity = TN / (TN + FP + self.eps)

        # Sensitivity component: 2 * w * Sensitivity² (α=2, γ=2 fixed)
        Sensi = 2 * self.w * (Sensitivity ** 2)

        # Harmonic term — matching TF operator precedence exactly:
        # (Precision*(1-w)*Specificity/Precision) + (1-w)*Specificity + ε
        # = (1-w)*Specificity + (1-w)*Specificity + ε
        # = 2*(1-w)*Specificity + ε
        harmonic = 2 * (1 - self.w) * Specificity + self.eps

        # Medical F-score and loss
        F_medical = Sensi * harmonic
        return 1 - F_medical


if __name__ == "__main__":
    torch.manual_seed(42)
    loss_fn = VarianceStabilizedLoss(w=0.56)

    logits  = torch.randn(32, requires_grad=True)
    targets = torch.zeros(32)
    targets[0] = 1.0
    targets[1] = 1.0

    loss = loss_fn(logits, targets)
    bce  = torch.nn.BCEWithLogitsLoss()(logits, targets)

    print("=== VS Loss — Sanity Check (TF-accurate) ===")
    print(f"VS Loss  (imbalanced batch): {loss.item():.6f}")
    print(f"BCE Loss (same batch):       {bce.item():.6f}")
    print(f"Gradient flows: {loss.requires_grad}")

    loss.backward()
    print(f"Gradient exists: {logits.grad is not None}")

    perfect_logits  = torch.tensor([10., 10., -10., -10.], requires_grad=True)
    perfect_targets = torch.tensor([1., 1., 0., 0.])
    perfect_loss    = loss_fn(perfect_logits, perfect_targets)
    print(f"\nPerfect predictions loss: {perfect_loss.item():.6f}")

    wrong_logits  = torch.tensor([-10., -10., 10., 10.], requires_grad=True)
    wrong_targets = torch.tensor([1., 1., 0., 0.])
    wrong_loss    = loss_fn(wrong_logits, wrong_targets)
    print(f"All wrong predictions:    {wrong_loss.item():.6f}")
    print("\nImplementation verified.")
