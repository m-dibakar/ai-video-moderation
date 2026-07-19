"""
Full fine-tuning — Focal Loss, 50 epochs, unfrozen backbone.
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
    "loss_fn":          "FocalLoss_FullFinetune",
    "alpha":            0.75,
    "gamma":            2.0,
    "imbalance_ratio":  0.05,
    "total_samples":    6000,
    "val_fraction":     0.2,
    "epochs":           50,
    "batch_size":       16,
    "learning_rate":    1e-4,
    "backbone_lr":      1e-5,
    "seed":             42,
    "dataset_dir":      "nsfw_dataset_v1",
    "checkpoint_path":  "checkpoints/model_focal_full.pt",
    "metrics_path":     "checkpoints/metrics_focal_full.json",
}


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce          = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt           = torch.exp(-bce)
        alpha_t      = self.alpha * targets + (1 - self.alpha) * (1 - targets)
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
    print(f"TRAINING: {CONFIG['loss_fn']}")
    print(f"Full backbone fine-tuning — {CONFIG['epochs']} epochs")
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

    train_ds     = NSFWDataset(train_samples, augment=True)
    val_ds       = NSFWDataset(val_samples,   augment=False)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print("Loading Falconsai model...")
    model = ViTForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)

    classifier_params = [p for n, p in model.named_parameters() if "classifier" in n]
    backbone_params   = [p for n, p in model.named_parameters() if "classifier" not in n]

    optimizer = torch.optim.AdamW([
        {"params": classifier_params, "lr": CONFIG["learning_rate"]},
        {"params": backbone_params,   "lr": CONFIG["backbone_lr"]},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
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

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images).logits.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        metrics  = evaluate(model, val_loader, device)
        elapsed  = time.time() - t0

        scheduler.step(metrics["f1"])

        history["epochs"].append({
            "epoch": epoch + 1,
            "train_loss": round(avg_loss, 4),
            **metrics,
            "time_seconds": round(elapsed, 1)
        })

        print(f"Epoch {epoch+1:02d}/{CONFIG['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"P: {metrics['precision']:.3f} | "
              f"R: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"FN: {metrics['fn']} | "
              f"Time: {elapsed:.0f}s")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"  ✓ Best saved (F1={best_f1:.3f})")

    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(history, f, indent=2)

    final = history["epochs"][-1]
    print(f"\n{'='*55}")
    print(f"FINAL — {CONFIG['loss_fn']}")
    print(f"{'='*55}")
    print(f"  Precision: {final['precision']:.4f}")
    print(f"  Recall:    {final['recall']:.4f}")
    print(f"  F1:        {final['f1']:.4f}")
    print(f"  FN:        {final['fn']}")
    print(f"  Best F1:   {best_f1:.4f}")

    return history


if __name__ == "__main__":
    train()
