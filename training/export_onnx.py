"""
Export the BCE full-finetune checkpoint (sweep winner) to ONNX for
CPU serverless deployment, quantize to int8, and verify the quantized
model reproduces the threshold-sweep metrics on the seed-42 val split.

Outputs:
  checkpoints/model_bce_full.onnx        fp32 export (intermediate)
  checkpoints/model_bce_full_int8.onnx   deployment artifact
  checkpoints/onnx_verify.json           int8 vs PyTorch metric comparison
"""

import json
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from dataset_loader import load_all_samples, build_imbalanced_split, NSFWDataset

# usage: python export_onnx.py [run_name] [sweep_cache_key]
RUN = sys.argv[1] if len(sys.argv) > 1 else "bce_full"
CACHE_KEY = sys.argv[2] if len(sys.argv) > 2 else "BCE (Baseline)"

CKPT = f"checkpoints/model_{RUN}.pt"
FP32_PATH = f"checkpoints/model_{RUN}.onnx"
INT8_PATH = f"checkpoints/model_{RUN}_int8.onnx"
VERIFY_PATH = f"checkpoints/onnx_verify_{RUN}.json"
PROBS_CACHE = "checkpoints/sweep_probs.json"


def export_fp32():
    model = ViTForImageClassification.from_pretrained(
        "Falconsai/nsfw_image_detection",
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, FP32_PATH,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )

    # parity check on one random batch
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        ref = model(x).logits.numpy()
    sess = ort.InferenceSession(FP32_PATH, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"pixel_values": x.numpy()})[0]
    diff = float(np.abs(ref - out).max())
    print(f"fp32 export parity: max |logit diff| = {diff:.2e}")
    assert diff < 1e-3, "fp32 ONNX export does not match PyTorch"


def verify_int8():
    safe_paths, nsfw_paths = load_all_samples("nsfw_dataset_v1")
    _, val_samples = build_imbalanced_split(
        safe_paths, nsfw_paths,
        imbalance_ratio=0.05, total_samples=6000, val_fraction=0.2, seed=42,
    )
    loader = DataLoader(NSFWDataset(val_samples, augment=False),
                        batch_size=32, shuffle=False, num_workers=4)

    sess = ort.InferenceSession(INT8_PATH, providers=["CPUExecutionProvider"])
    probs, t0 = [], time.time()
    for images, _ in loader:
        logits = sess.run(None, {"pixel_values": images.numpy()})[0].squeeze(1)
        probs.extend((1 / (1 + np.exp(-logits))).tolist())
    elapsed = time.time() - t0
    print(f"int8 CPU inference: {len(probs)} images in {elapsed:.1f}s "
          f"({elapsed / len(probs) * 1000:.0f} ms/image)")

    y = np.array([int(l) for _, l in val_samples])
    p = np.array(probs)

    with open(PROBS_CACHE) as f:
        cache = json.load(f)
    ref = np.array(cache["probs"][CACHE_KEY])

    def metrics(scores):
        prec, rec, thr = precision_recall_curve(y, scores)
        f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
        mask = rec >= 0.90
        return {
            "pr_auc": round(float(average_precision_score(y, scores)), 4),
            "roc_auc": round(float(roc_auc_score(y, scores)), 4),
            "p_at_r90": round(float(prec[mask].max()), 4),
            "max_f1": round(float(f1.max()), 4),
            "max_f1_threshold": round(float(thr[min(int(f1.argmax()), len(thr) - 1)]), 4),
        }

    result = {
        "pytorch_fp32": metrics(ref),
        "onnx_int8": metrics(p),
        "max_prob_diff_vs_pytorch": round(float(np.abs(p - ref).max()), 4),
        "mean_prob_diff_vs_pytorch": round(float(np.abs(p - ref).mean()), 4),
        "ms_per_image_cpu": round(elapsed / len(probs) * 1000, 1),
    }
    with open(VERIFY_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n{'':<14}{'PR-AUC':>8}{'ROC-AUC':>9}{'P@R=.90':>9}{'maxF1':>8}")
    for name, m in [("PyTorch fp32", result["pytorch_fp32"]), ("ONNX int8", result["onnx_int8"])]:
        print(f"{name:<14}{m['pr_auc']:>8.4f}{m['roc_auc']:>9.4f}"
              f"{m['p_at_r90']:>9.4f}{m['max_f1']:>8.4f}")
    print(f"\nprob diff vs PyTorch: max {result['max_prob_diff_vs_pytorch']}, "
          f"mean {result['mean_prob_diff_vs_pytorch']}")
    print("Saved verify json")


if __name__ == "__main__":
    export_fp32()
    quantize_dynamic(FP32_PATH, INT8_PATH, weight_type=QuantType.QInt8)
    import os
    for path in (FP32_PATH, INT8_PATH):
        print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB")
    verify_int8()
