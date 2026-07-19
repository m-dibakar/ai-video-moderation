import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

MODELS = {
    "BCE (Baseline)":            "checkpoints/metrics_bce_full.json",
    "Focal Loss (α=0.75, γ=2)": "checkpoints/metrics_focal_full.json",
    "VS Loss (IEEE SPL)":        "checkpoints/metrics_varstab_full.json",
}

COLORS = {
    "BCE (Baseline)":            "#ef4444",
    "Focal Loss (α=0.75, γ=2)": "#f97316",
    "VS Loss (IEEE SPL)":        "#22c55e",
}


def load_metrics(path):
    with open(path) as f:
        return json.load(f)["epochs"]


def main():
    all_metrics = {}
    for name, path in MODELS.items():
        if not os.path.exists(path):
            print(f"Missing: {path}")
            return
        all_metrics[name] = load_metrics(path)

    epochs = range(1, len(next(iter(all_metrics.values()))) + 1)

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0f0f0f")
    fig.suptitle(
        "Loss Function Comparison: NSFW Content Moderation\n"
        "Full Fine-tuning (50 epochs) | Dataset: deepghs/nsfw_detect | "
        "Imbalance: 95/5 | Backbone: Falconsai ViT",
        fontsize=13, fontweight="bold", color="white", y=0.98
    )

    gs   = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
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

    plot_keys = {
        "recall":    ("recall",     "Recall (↑ better)"),
        "precision": ("precision",  "Precision"),
        "f1":        ("f1",         "F1 Score (↑ better)"),
        "loss":      ("train_loss", "Training Loss (↓ better)"),
        "fpr":       ("fpr",        "False Positive Rate (↓ better)"),
    }

    for ax_key, (metric, ylabel) in plot_keys.items():
        ax = axes[ax_key]
        for name, metrics in all_metrics.items():
            values = [e[metric] for e in metrics]
            ax.plot(epochs, values, label=name,
                    color=COLORS[name], linewidth=2.5, marker="o",
                    markersize=3, markevery=5)
        ax.set_title(ylabel, fontsize=10, pad=8)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.legend(fontsize=7, facecolor="#252525",
                  labelcolor="white", edgecolor="#444")
        ax.grid(True, alpha=0.2, color="#444")
        if metric in ("recall", "precision", "f1"):
            ax.set_ylim(0, 1)

    # Final bar chart
    ax          = axes["bar"]
    model_names = list(all_metrics.keys())
    short_names = ["BCE", "Focal", "VS Loss\n(IEEE SPL)"]
    best_f1     = [max(e["f1"]        for e in all_metrics[m]) for m in model_names]
    best_recall = [max(e["recall"]    for e in all_metrics[m]) for m in model_names]
    min_fn      = [min(e["fn"]        for e in all_metrics[m]) for m in model_names]
    best_prec   = [max(e["precision"] for e in all_metrics[m]) for m in model_names]

    x = np.arange(len(model_names))
    w = 0.22
    ax.bar(x - w,   best_prec,   w, label="Precision", color="#3b82f6", alpha=0.9)
    ax.bar(x,       best_recall, w, label="Recall",    color="#8b5cf6", alpha=0.9)
    ax.bar(x + w,   best_f1,     w, label="F1",        color="#22c55e", alpha=0.9)
    ax.set_title("Best Metrics (50 epochs)", fontsize=10, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, facecolor="#252525",
              labelcolor="white", edgecolor="#444")
    ax.grid(True, alpha=0.2, color="#444", axis="y")

    for i, fn in enumerate(min_fn):
        ax.annotate(f"FN={fn}", xy=(x[i], 0.02),
                    ha="center", fontsize=8, color="#fbbf24")

    # Summary
    summary = "BEST RESULTS (50-epoch full fine-tuning)\n"
    summary += "─" * 48 + "\n"
    summary += f"{'Model':<22} {'P':>6} {'R':>6} {'F1':>6} {'FN':>5}\n"
    summary += "─" * 48 + "\n"
    for name, sname in zip(model_names, ["BCE", "Focal", "VS Loss (IEEE SPL)"]):
        best_ep = max(all_metrics[name], key=lambda e: e["f1"])
        summary += (f"{sname:<22} "
                    f"{best_ep['precision']:>6.3f} "
                    f"{best_ep['recall']:>6.3f} "
                    f"{best_ep['f1']:>6.3f} "
                    f"{best_ep['fn']:>5}\n")
    summary += "─" * 48 + "\n"
    summary += "VS Loss: IEEE Signal Processing Letters, 2025"

    fig.text(0.01, 0.01, summary, fontsize=8.5,
             fontfamily="monospace", color="#cccccc",
             verticalalignment="bottom",
             bbox=dict(boxstyle="round", facecolor="#1a1a1a",
                       edgecolor="#444", alpha=0.9))

    out = "benchmark_results_full.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
