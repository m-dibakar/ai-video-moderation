"""
Full fine-tuning run with Asymmetric Loss (ASL) + logit adjustment.

Loss designed from the failure analysis of the BCE/Focal/VS benchmark:
Focal's weakness was a noisy negative tail (17 negatives > 0.1) costing
precision; BCE's was 8 confidently-missed positives. ASL targets both:
gamma_pos=0 keeps full gradient on rare positives, gamma_neg=4 + a 0.05
probability margin hard-zeroes easy negatives, and the class-prior logit
adjustment (log 0.05/0.95) is the Bayes-consistent recall correction.
Training recipe is otherwise identical to the other three runs.
"""

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json, time, os
from dataset_loader import load_all_samples, build_imbalanced_split, NSFWDataset
from loss_asl import AsymmetricLoss

CONFIG = {
    "loss_fn":          "ASL_FullFinetune",
    "gamma_pos":        0.0,
    "gamma_neg":        4.0,
    "clip":             0.05,
    "prior":            0.05,
    "imbalance_ratio":  0.05,
    "total_samples":    6000,
    "val_fraction":     0.2,
    "epochs":           50,
    "batch_size":       16,
    "learning_rate":    1e-4,
    "backbone_lr":      1e-5,
    "seed":             42,
    "dataset_dir":      "nsfw_dataset_v1",
    "checkpoint_path":  "checkpoints/model_asl_full.pt",
    "metrics_path":     "checkpoints/metrics_asl_full.json",
}


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
    print(f"γ+={CONFIG['gamma_pos']}, γ-={CONFIG['gamma_neg']}, "
          f"m={CONFIG['clip']}, prior={CONFIG['prior']}")
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

    classifier_params = []
    backbone_params   = []
    for name, param in model.named_parameters():
        if "classifier" in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": classifier_params, "lr": CONFIG["learning_rate"]},
        {"params": backbone_params,   "lr": CONFIG["backbone_lr"]},
    ], weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    criterion = AsymmetricLoss(
        gamma_pos=CONFIG["gamma_pos"],
        gamma_neg=CONFIG["gamma_neg"],
        clip=CONFIG["clip"],
        prior=CONFIG["prior"],
    )

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

        # dump every epoch so a killed run keeps its history
        with open(CONFIG["metrics_path"], "w") as f:
            json.dump(history, f, indent=2)

        print(f"Epoch {epoch+1:02d}/{CONFIG['epochs']} | "
              f"Loss: {avg_loss:.4f} | "
              f"P: {metrics['precision']:.3f} | "
              f"R: {metrics['recall']:.3f} | "
              f"F1: {metrics['f1']:.3f} | "
              f"FN: {metrics['fn']} | "
              f"Time: {elapsed:.0f}s", flush=True)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), CONFIG["checkpoint_path"])
            print(f"  ✓ Best saved (F1={best_f1:.3f})", flush=True)

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
