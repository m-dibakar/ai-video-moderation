"""
Benchmark: Compare all three loss functions visually.
Run after all three training scripts complete.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

MODELS = {
    "BCE (Baseline)":              "checkpoints/metrics_bce.json",
    "Focal Loss (α=0.75, γ=2)":   "checkpoints/metrics_focal.json",
    "Var-Stab Loss (IEEE SPL)":    "checkpoints/metrics_varstab.json",
}

COLORS = {
    "BCE (Baseline)":              "#ef4444",
    "Focal Loss (α=0.75, γ=2)":   "#f97316",
    "Var-Stab Loss (IEEE SPL)":    "#22c55e",
}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return data["epochs"]


def main():
    # ── Load all metrics ───────────────────────────────────────────────────
    all_metrics = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Missing: {path} — run training first")
            return
        all_metrics[name] = load_metrics(path)

    epochs = range(1, len(next(iter(all_metrics.values()))) + 1)

    # ── Figure setup ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(
        "Loss Function Comparison: NSFW Content Moderation\n"
        "Dataset: deepghs/nsfw_detect | Imbalance: 95% Safe / 5% NSFW | "
        "Backbone: Falconsai ViT",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = {
        "recall":    fig.add_subplot(gs[0, 0]),
        "precision": fig.add_subplot(gs[0, 1]),
        "f1":        fig.add_subplot(gs[0, 2]),
        "loss":      fig.add_subplot(gs[1, 0]),
        "fpr":       fig.add_subplot(gs[1, 1]),
        "bar":       fig.add_subplot(gs[1, 2]),
    }

    for ax in axes.values():
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # ── Line plots ─────────────────────────────────────────────────────────
    plot_keys = {
        "recall":    ("recall",     "Recall (↑ better)",     True),
        "precision": ("precision",  "Precision",             True),
        "f1":        ("f1",         "F1 Score (↑ better)",   True),
        "loss":      ("train_loss", "Training Loss (↓ better)", False),
        "fpr":       ("fpr",        "False Positive Rate (↓ better)", False),
    }

    for ax_key, (metric, ylabel, higher_better) in plot_keys.items():
        ax = axes[ax_key]
        for name, metrics in all_metrics.items():
            values = [e[metric] for e in metrics]
            ax.plot(epochs, values, label=name,
                    color=COLORS[name], linewidth=2.5, marker="o", markersize=4)
        ax.set_title(ylabel, fontsize=10, pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=9)
        ax.legend(fontsize=7, facecolor="#252525", labelcolor="white",
                  edgecolor="#444")
        ax.grid(True, alpha=0.2, color="#444")
        if higher_better:
            ax.set_ylim(0, 1)

    # ── Final metrics bar chart ────────────────────────────────────────────
    ax = axes["bar"]
    model_names  = list(all_metrics.keys())
    short_names  = ["BCE", "Focal", "VarStab"]
    final_recall    = [all_metrics[m][-1]["recall"]    for m in model_names]
    final_precision = [all_metrics[m][-1]["precision"] for m in model_names]
    final_f1        = [all_metrics[m][-1]["f1"]        for m in model_names]
    final_fn        = [all_metrics[m][-1]["fn"]        for m in model_names]

    x = np.arange(len(model_names))
    w = 0.22
    ax.bar(x - w,   final_precision, w, label="Precision", color="#3b82f6", alpha=0.9)
    ax.bar(x,       final_recall,    w, label="Recall",    color="#8b5cf6", alpha=0.9)
    ax.bar(x + w,   final_f1,        w, label="F1",        color="#22c55e", alpha=0.9)

    ax.set_title("Final Epoch Metrics", fontsize=10, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, facecolor="#252525", labelcolor="white", edgecolor="#444")
    ax.grid(True, alpha=0.2, color="#444", axis="y")

    # Add FN annotation on bars
    for i, fn in enumerate(final_fn):
        ax.annotate(f"FN={fn}", xy=(x[i], 0.02),
                    ha="center", fontsize=8, color="#fbbf24")

    # ── Summary table (text box) ───────────────────────────────────────────
    summary = (
        "SUMMARY (Final Epoch)\n"
        "─────────────────────────────────────────────\n"
        f"{'Model':<22} {'P':>6} {'R':>6} {'F1':>6} {'FN':>5}\n"
        "─────────────────────────────────────────────\n"
    )
    for name, sname in zip(model_names, short_names):
        m = all_metrics[name][-1]
        summary += (
            f"{sname:<22} "
            f"{m['precision']:>6.3f} "
            f"{m['recall']:>6.3f} "
            f"{m['f1']:>6.3f} "
            f"{m['fn']:>5}\n"
        )
    summary += "─────────────────────────────────────────────\n"
    summary += "FN = violations missed (lower is better)\n"
    summary += "* VarStab uses placeholder — replace with IEEE formula"

    fig.text(
        0.01, 0.01, summary,
        fontsize=8.5, fontfamily="monospace", color="#cccccc",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="#1a1a1a",
                  edgecolor="#444", alpha=0.9)
    )

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = "benchmark_results.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
