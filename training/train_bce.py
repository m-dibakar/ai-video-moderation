"""
Training Run 1: Binary Cross-Entropy Loss (Baseline)

This is the broken baseline. On 95/5 imbalanced data, BCE will
achieve high accuracy but near-zero recall — it learns to predict
"safe" for everything. We document this failure before fixing it.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import json
import time
from dataset_loader import load_all_samples, build_imbalanced_split, NSFWDataset

# ── Config ─────────────────────────────────────────────────────────────────
CONFIG = {
    "loss_fn":          "BCE",
    "imbalance_ratio":  0.05,       # 95/5 split — realistic OTT
    "total_samples":    6000,
    "val_fraction":     0.2,
    "epochs":           15,
    "batch_size":       32,
    "learning_rate":    2e-4,
    "seed":             42,
    "dataset_dir":      "nsfw_dataset_v1",
    "checkpoint_path":  "checkpoints/model_bce.pt",
    "metrics_path":     "checkpoints/metrics_bce.json",
}


def evaluate(model, loader, device, threshold=0.5):
    """Run evaluation and return metrics dict."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).logits.squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > threshold).astype(int)
            
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
    
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall    = recall_score(all_labels, all_preds, zero_division=0)
    f1        = f1_score(all_labels, all_preds, zero_division=0)
    accuracy  = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    cm        = confusion_matrix(all_labels, all_preds)
    
    # False positive rate = FP / (FP + TN)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2,2) else (0, 0, 0, 0)
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
    print(f"TRAINING: {CONFIG['loss_fn']} Loss")
    print(f"Imbalance ratio: {CONFIG['imbalance_ratio']} "
          f"({int(CONFIG['imbalance_ratio']*100)}% NSFW)")
    print(f"Device: {device}")
    print(f"{'='*55}\n")
    
    # ── Dataset ────────────────────────────────────────────────────────────
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
    
    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True
    )
    
    # ── Model ──────────────────────────────────────────────────────────────
    # Start from Falconsai — already knows NSFW patterns
    # We replace its head with a single binary output
    print("Loading Falconsai model as pretrained backbone...")
    model = ViTForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        num_labels=1,
        ignore_mismatched_sizes=True
    ).to(device)
    
    # Freeze all transformer blocks, only train the classifier head
    # This is called "linear probing" — fast and prevents overfitting
    # on small datasets
    for name, param in model.named_parameters():
        if "classifier" not in name:
            param.requires_grad = False
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({trainable/total*100:.1f}%)")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG["learning_rate"],
        weight_decay=0.01
    )
    
    # ── LOSS FUNCTION: Binary Cross-Entropy ────────────────────────────────
    # Standard BCE — treats every sample equally
    # Expected failure: on 95/5 data, model will predict all-safe
    criterion = nn.BCEWithLogitsLoss()
    # ──────────────────────────────────────────────────────────────────────
    
    # ── Training loop ──────────────────────────────────────────────────────
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    history = {
        "config": CONFIG,
        "epochs": [],
    }
    
    best_f1 = 0.0
    
    for epoch in range(CONFIG["epochs"]):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
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
        
        # Evaluate
        metrics = evaluate(model, val_loader, device)
        elapsed = time.time() - t0
        
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
        
        # Save best model
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"  ✓ Best model saved (F1={best_f1:.3f})")
        
        print()
    
    # Save metrics
    with open(CONFIG["metrics_path"], "w") as f:
        json.dump(history, f, indent=2)
    
    # Final summary
    final = history["epochs"][-1]
    print(f"\n{'='*55}")
    print(f"FINAL RESULTS — {CONFIG['loss_fn']} Loss")
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
