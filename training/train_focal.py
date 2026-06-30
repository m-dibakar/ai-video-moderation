"""
Training Run 2: Focal Loss
Same config as BCE — only the loss function changes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json, time, os
from dataset_loader import load_all_samples, build_imbalanced_split, NSFWDataset

CONFIG = {
    "loss_fn":          "FocalLoss",
    "alpha":            0.75,
    "gamma":            2.0,
    "imbalance_ratio":  0.05,
    "total_samples":    6000,
    "val_fraction":     0.2,
    "epochs":           5,
    "batch_size":       32,
    "learning_rate":    2e-4,
    "seed":             42,
    "dataset_dir":      "nsfw_dataset_v1",
    "checkpoint_path":  "checkpoints/model_focal.pt",
    "metrics_path":     "checkpoints/metrics_focal.json",
}


class FocalLoss(nn.Module):
    """
    Focal Loss — Lin et al., 2017 (RetinaNet paper).
    
    FL(pt) = -α(1-pt)^γ · log(pt)
    
    Key idea: downweight easy examples (confident safe frames)
    so the model focuses on hard ones (violation frames).
    
    α=0.75 → violations get 3x more weight than safe frames
    γ=2.0  → standard focus parameter from the paper
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).logits.squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    accuracy  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm        = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-8)

    return {
        "accuracy":  round(accuracy, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
        "fpr":       round(fpr, 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def train():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"TRAINING: {CONFIG['loss_fn']} (α={CONFIG['alpha']}, γ={CONFIG['gamma']})")
    print(f"Imbalance ratio: {CONFIG['imbalance_ratio']} "
          f"({int(CONFIG['imbalance_ratio']*100)}% NSFW)")
    print(f"Device: {device}")
    print(f"{'='*55}\n")

    safe_paths, nsfw_paths = load_all_samples(CONFIG["dataset_dir"])
    train_samples, val_samples = build_imbalanced_split(
        safe_paths, nsfw_paths,
        imbalance_ratio=CONFIG["imbalance_ratio"],
        total_samples=CONFIG["total_samples"],
        val_fraction=CONFIG["val_fraction"],
        seed=CONFIG["seed"]
    )

    train_ds = NSFWDataset(train_samples, augment=True)
    val_ds   = NSFWDataset(val_samples,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print("Loading Falconsai model as pretrained backbone...")
    model = ViTForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"], weight_decay=0.01
    )

    # ── LOSS FUNCTION: Focal Loss ──────────────────────────────────────────
    criterion = FocalLoss(alpha=CONFIG["alpha"], gamma=CONFIG["gamma"])
    # ──────────────────────────────────────────────────────────────────────

    os.makedirs("checkpoints", exist_ok=True)
    history = {"config": CONFIG, "epochs": []}
    best_f1 = 0.0

    for epoch in range(CONFIG["epochs"]):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        metrics  = evaluate(model, val_loader, device)
        elapsed  = time.time() - t0

        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            **metrics,
            "time_seconds": round(elapsed, 1)
        }
        history["epochs"].append(epoch_log)

        print(f"\nEpoch {epoch+1}/{CONFIG['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"Acc: {metrics['accuracy']:.3f} | "
              f"P: {metrics['precision']:.3f} | "
              f"R: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"FPR: {metrics['fpr']:.3f} | "
              f"Time: {elapsed:.0f}s")
        print(f"  Confusion: TP={metrics['tp']} FP={metrics['fp']} "
              f"TN={metrics['tn']} FN={metrics['fn']}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"  ✓ Best model saved (F1={best_f1:.3f})")
        print()

    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(history, f, indent=2)

    final = history["epochs"][-1]
    print(f"\n{'='*55}")
    print(f"FINAL RESULTS — {CONFIG['loss_fn']}")
    print(f"{'='*55}")
    print(f"  Accuracy:  {final['accuracy']:.4f}")
    print(f"  Precision: {final['precision']:.4f}")
    print(f"  Recall:    {final['recall']:.4f}  ← key metric")
    print(f"  F1 Score:  {final['f1']:.4f}  ← key metric")
    print(f"  FPR:       {final['fpr']:.4f}")
    print(f"  TP: {final['tp']}  FP: {final['fp']}  "
          f"TN: {final['tn']}  FN: {final['fn']}")
    print(f"\nMetrics saved to {CONFIG['metrics_path']}")
    print(f"Model saved to   {CONFIG['checkpoint_path']}")

    return history


if __name__ == "__main__":
    train()
