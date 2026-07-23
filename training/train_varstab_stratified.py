"""
VS Loss with stratified batches — the "fair shot" run.

The 50-epoch benchmark showed VS Loss starving: at a 5% positive rate and
batch size 16, ~44% of batches contain zero positives, and VS Loss's
batch-level objective (1 − Sensi × harmonic) pins at ~1.0 with no usable
gradient on those batches. This run keeps everything else identical to
train_varstab_full.py but guarantees 2 positives in every batch of 16,
so every step produces gradient signal. If the diagnosis is right,
VS Loss should improve substantially here.
"""

import random

import torch
from torch.utils.data import DataLoader, Sampler
from transformers import ViTForImageClassification
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import json, time, os
from dataset_loader import load_all_samples, build_imbalanced_split, NSFWDataset
from loss_varstab import VarianceStabilizedLoss

CONFIG = {
    "loss_fn":          "VarianceStabilized_StratifiedBatches",
    "w":                0.56,
    "imbalance_ratio":  0.05,
    "total_samples":    6000,
    "val_fraction":     0.2,
    "epochs":           50,
    "batch_size":       16,
    "pos_per_batch":    2,         # guaranteed positives per batch — the fix under test
    "learning_rate":    1e-4,
    "backbone_lr":      1e-5,
    "seed":             42,
    "dataset_dir":      "nsfw_dataset_v1",
    "checkpoint_path":  "checkpoints/model_varstab_strat.pt",
    "metrics_path":     "checkpoints/metrics_varstab_strat.json",
}


class StratifiedBatchSampler(Sampler):
    """
    Yields batches with a fixed number of positive samples in each.

    Negatives are shuffled and consumed exactly once per epoch; positives
    are cycled (reshuffled each pass), so with 240 train positives and
    ~300 batches × 2 slots, each positive is seen ~2.5× per epoch.
    """

    def __init__(self, labels, batch_size, pos_per_batch, seed):
        self.pos_idx = [i for i, l in enumerate(labels) if l == 1]
        self.neg_idx = [i for i, l in enumerate(labels) if l == 0]
        assert len(self.pos_idx) >= pos_per_batch, (
            f"need at least {pos_per_batch} positives, got {len(self.pos_idx)}"
        )
        self.batch_size = batch_size
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = batch_size - pos_per_batch
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.neg_idx) // self.neg_per_batch

    def __iter__(self):
        negs = self.neg_idx[:]
        self.rng.shuffle(negs)
        pos_pool = []
        for b in range(len(self)):
            batch = []
            while len(batch) < self.pos_per_batch:
                if not pos_pool:
                    pos_pool = self.pos_idx[:]
                    self.rng.shuffle(pos_pool)
                cand = pos_pool.pop()
                if cand in batch:
                    # leftover from the previous pass collided with its own
                    # copy in the refill — defer it to a later batch
                    pos_pool.insert(0, cand)
                else:
                    batch.append(cand)
            batch += negs[b * self.neg_per_batch:(b + 1) * self.neg_per_batch]
            self.rng.shuffle(batch)
            yield batch


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
        "precision": round(float(precision), 4),
        "recall":    round(float(recall), 4),
        "f1":        round(float(f1), 4),
        "fpr":       round(float(fpr), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
    }


def train():
    torch.manual_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"TRAINING: {CONFIG['loss_fn']}")
    print(f"Stratified batches: {CONFIG['pos_per_batch']} positives / batch of {CONFIG['batch_size']}")
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

    sampler = StratifiedBatchSampler(
        labels=[l for _, l in train_samples],
        batch_size=CONFIG["batch_size"],
        pos_per_batch=CONFIG["pos_per_batch"],
        seed=CONFIG["seed"],
    )
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"],
                              shuffle=False, num_workers=4, pin_memory=True)

    print("Loading Falconsai model...")
    model = ViTForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        num_labels=1,
        ignore_mismatched_sizes=True
    )
    model.to(device)  # type: ignore  # Pylance false positive: transformers wraps Module.to

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

    criterion = VarianceStabilizedLoss(w=CONFIG["w"], eps=1e-7)

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
