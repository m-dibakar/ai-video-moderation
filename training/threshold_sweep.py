
"""
Threshold-independent comparison of the three trained checkpoints.

Rebuilds the exact seed-42 validation split used in training, scores it
with each best-F1 checkpoint, and reports PR-AUC / ROC-AUC plus metrics
at swept decision thresholds. Raw probabilities are cached to
checkpoints/sweep_probs.json so the analysis can be re-run without a GPU.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from dataset_loader import load_all_samples, build_imbalanced_split, NSFWDataset

MODELS = {
    "BCE (Baseline)":            "checkpoints/model_bce_full.pt",
    "Focal Loss (α=0.75, γ=2)": "checkpoints/model_focal_full.pt",
    "VS Loss (IEEE SPL)":        "checkpoints/model_varstab_full.pt",
    "VS Loss (stratified)":      "checkpoints/model_varstab_strat.pt",
    "ASL (γ-=4, m=0.05)":        "checkpoints/model_asl_full.pt",
}

PROBS_CACHE = "checkpoints/sweep_probs.json"


def collect_probs(models, out=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    safe_paths, nsfw_paths = load_all_samples("nsfw_dataset_v1")
    _, val_samples = build_imbalanced_split(
        safe_paths, nsfw_paths,
        imbalance_ratio=0.05, total_samples=6000, val_fraction=0.2, seed=42,
    )

    val_ds = NSFWDataset(val_samples, augment=False)
    loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)


    labels = [int(l) for _, l in val_samples]
    if out is None:
        out = {"labels": labels, "probs": {}}

    for name, ckpt in models.items():
        model = ViTForImageClassification.from_pretrained(
            "Falconsai/nsfw_image_detection",
            num_labels=1,
            ignore_mismatched_sizes=True,
        )
        model.to(device)  # type: ignore  # Pylance false positive: transformers wraps Module.to
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        probs = []
        with torch.no_grad():
            for images, _ in loader:
                logits = model(images.to(device)).logits.squeeze(1)
                probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        out["probs"][name] = probs
        print(f"Scored {name}: {len(probs)} samples")

    with open(PROBS_CACHE, "w") as f:
        json.dump(out, f)
    return out


def precision_at_recall(precision, recall, target):
    """Best precision achievable at recall >= target."""
    mask = recall >= target
    return float(precision[mask].max()) if mask.any() else float("nan")


def recall_at_precision(precision, recall, target):
    """Best recall achievable at precision >= target."""
    mask = precision >= target
    return float(recall[mask].max()) if mask.any() else float("nan")


def main():
    if os.path.exists(PROBS_CACHE):
        with open(PROBS_CACHE) as f:
            data = json.load(f)
        print(f"Loaded cached probabilities from {PROBS_CACHE}")
        missing = {n: c for n, c in MODELS.items() if n not in data["probs"]}
        if missing:
            print(f"Scoring {len(missing)} uncached model(s): {list(missing)}")
            data = collect_probs(missing, out=data)
    else:
        data = collect_probs(MODELS)

    y = np.array(data["labels"])
    n_pos = int(y.sum())
    print(f"\nValidation set: {len(y)} samples, {n_pos} positives "
          f"({n_pos / len(y) * 100:.1f}%)\n")

    header = (f"{'Model':<26} {'PR-AUC':>7} {'ROC-AUC':>8} "
              f"{'P@R=.85':>8} {'P@R=.90':>8} {'R@P=.90':>8} {'maxF1':>7}")
    print(header)
    print("─" * len(header))

    results = {}
    for name, probs in data["probs"].items():
        p = np.array(probs)
        pr_auc = average_precision_score(y, p)
        roc = roc_auc_score(y, p)
        prec, rec, thr = precision_recall_curve(y, p)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        results[name] = {
            "pr_auc":  round(float(pr_auc), 4),
            "roc_auc": round(float(roc), 4),
            "p_at_r85": round(precision_at_recall(prec, rec, 0.85), 4),
            "p_at_r90": round(precision_at_recall(prec, rec, 0.90), 4),
            "r_at_p90": round(recall_at_precision(prec, rec, 0.90), 4),
            "max_f1":  round(float(f1.max()), 4),
            "max_f1_threshold": round(float(thr[min(int(f1.argmax()), len(thr) - 1)]), 4),
        }
        r = results[name]
        print(f"{name:<26} {r['pr_auc']:>7.4f} {r['roc_auc']:>8.4f} "
              f"{r['p_at_r85']:>8.4f} {r['p_at_r90']:>8.4f} "
              f"{r['r_at_p90']:>8.4f} {r['max_f1']:>7.4f}")

    with open("checkpoints/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: checkpoints/sweep_results.json")


if __name__ == "__main__":
    main()
