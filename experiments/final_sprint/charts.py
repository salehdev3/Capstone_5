"""Publication artifacts from saved measurements; no inference or score modification."""

from common import *
import numpy as np


def render(out, rows, classes):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out)
    dest = out / "submission/figures"
    dest.mkdir(parents=True, exist_ok=True)
    paths = {}
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    history = table(out / "classifier/history.csv")
    epochs = [int(r["epoch"]) for r in history]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(
        epochs,
        [float(r["train_subset_accuracy"]) for r in history],
        label="Unaugmented training probe (512 crops)",
    )
    ax.plot(
        epochs,
        [float(r["val_accuracy"]) for r in history],
        label="Selection-validation crops",
    )
    ax.set(
        xlabel="Completed epoch",
        ylabel="Accuracy",
        ylim=(0, 1),
        title="Classifier diagnostic learning curves",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = dest / "classifier_learning.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["learning"] = path
    metric = read(out / "historical_test/species_metrics.json")
    if "confusion_matrix" in metric:
        cm = np.array(metric["confusion_matrix"])
        normal = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        fig, ax = plt.subplots(figsize=(8, 7))
        plot = ax.imshow(normal, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(14), classes, rotation=65, ha="right")
        ax.set_yticks(
            range(14), [f"{c} (n={cm[i].sum()})" for i, c in enumerate(classes)]
        )
        ax.set(
            xlabel="Predicted species",
            ylabel="Reference species",
            title="Historical known-class crop confusion (row normalized)",
        )
        fig.colorbar(plot, ax=ax, label="Within-reference-class fraction")
        fig.tight_layout()
        path = dest / "species_confusion.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths["confusion"] = path
    counts = {
        s: [
            sum(
                b["species"] == name
                for r in rows
                if r["split"] == s
                for b in r["boxes"]
            )
            for name in classes
        ]
        for s in ["train", "val_select", "test"]
    }
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(14)
    for offset, (split, values) in enumerate(counts.items()):
        ax.bar(x + (offset - 1) * 0.25, values, 0.25, label=split)
    ax.set_xticks(x, classes, rotation=50, ha="right")
    ax.set(ylabel="Human-box crops", title="Fixed species support by split")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = dest / "species_support.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    paths["support"] = path
    sweep = out / "threshold_sweep.csv"
    if sweep.exists():
        records = table(sweep)
        fig, ax = plt.subplots(figsize=(7, 4))
        for name in sorted({r["detector"] for r in records}):
            group = [r for r in records if r["detector"] == name]
            ax.scatter(
                [float(r["animal_recall"]) for r in group],
                [float(r["empty_removal"]) for r in group],
                s=9,
                alpha=0.4,
                label=name,
            )
        selected = read(out / "detector_selection.json")["validation"]
        ax.scatter(
            [selected["animal_recall"]],
            [selected["empty_removal"]],
            s=120,
            marker="*",
            color="crimson",
            label="Frozen operating point",
        )
        ax.axvline(0.99, color="grey", linestyle="--", linewidth=1)
        ax.axhline(0.2, color="grey", linestyle="--", linewidth=1)
        ax.set(
            xlabel="Animal-frame recall",
            ylabel="Empty-frame removal",
            title="Joint validation threshold sweep",
            xlim=(0, 1.01),
            ylim=(0, 1.01),
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = dest / "triage_tradeoff.png"
        fig.savefig(path, dpi=170)
        plt.close(fig)
        paths["tradeoff"] = path
    return paths
