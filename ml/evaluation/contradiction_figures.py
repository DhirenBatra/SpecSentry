"""Render the held-out NLI confusion matrices and binary comparison."""

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from ml.evaluation.classification_figures import BLUE, GREEN, GRID, INK, new_figure


def write_figures(destination, held_out, sample_size):
    caption = f"Curated held-out evaluation · {sample_size} requirement pairs"
    for task, title in (
        ("nli_three_class", "NLI three-class confusion matrix"),
        ("nli_binary", "NLI binary contradiction confusion matrix"),
    ):
        matrix = held_out[task]["confusion_matrix"]
        figure, axes = new_figure(title, (7.4, 5.5), caption)
        display = ConfusionMatrixDisplay(
            np.asarray(matrix["counts"]),
            display_labels=[label.replace("_", " ").capitalize() for label in matrix["label_order"]],
        )
        display.plot(ax=axes, cmap="Greens", colorbar=False, values_format="d")
        axes.set_xlabel("Predicted label", color=INK)
        axes.set_ylabel("Curated label", color=INK)
        figure.savefig(destination / f"contradiction_{task}_confusion_matrix.png", dpi=180)

    figure, axes = new_figure("Binary contradiction detection", (8.1, 4.9), caption)
    metrics = ("accuracy", "precision", "recall", "f1")
    positions = np.arange(len(metrics))
    width = 0.34
    for index, (identifier, label, color) in enumerate((
        ("rule_binary", "Rule baseline", GREEN), ("nli_binary", "MiniLM NLI", BLUE),
    )):
        bars = axes.bar(
            positions + width * (index - 0.5),
            [held_out[identifier]["metrics"][metric] for metric in metrics],
            width=width, color=color, label=label,
        )
        axes.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    axes.set_xticks(positions, labels=["Accuracy", "Precision", "Recall", "F1"])
    axes.set_ylim(0, 1.08)
    axes.set_ylabel("Held-out score", color=INK)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2, frameon=False)
    figure.savefig(destination / "contradiction_method_comparison.png", dpi=180)
