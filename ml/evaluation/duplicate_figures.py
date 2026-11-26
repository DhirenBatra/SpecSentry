"""Render held-out confusion counts, metric comparisons, and cosine scores."""

from math import floor

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from ml.evaluation.classification_figures import BLUE, GREEN, GRID, INK, new_figure

NAMES = {"tfidf": "TF-IDF", "sbert": "SBERT (MiniLM)"}


def write_figures(destination, pairs, methods, predictions):
    caption = f"Curated held-out evaluation · {len(pairs)} pairs · Duplicate is the positive class"
    max_count = max(sum(row) for row in methods["tfidf"]["confusion_matrix"]["counts"])
    for method, name in NAMES.items():
        figure, axes = new_figure(f"{name} confusion matrix", (6.7, 5.4), caption)
        display = ConfusionMatrixDisplay(
            np.asarray(methods[method]["confusion_matrix"]["counts"]),
            display_labels=["Non-duplicate", "Duplicate"],
        )
        display.plot(ax=axes, cmap="Greens", colorbar=False, values_format="d",
                     im_kw={"vmin": 0, "vmax": max_count})
        axes.set_xlabel("Predicted label", color=INK)
        axes.set_ylabel("Curated label", color=INK)
        figure.savefig(destination / f"duplicate_{method}_confusion_matrix.png", dpi=180)

    figure, axes = new_figure("Duplicate detection method comparison", (8.1, 4.9), caption)
    metric_names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1"}
    positions = np.arange(len(metric_names))
    width = 0.34
    for index, (method, color) in enumerate(zip(NAMES, (GREEN, BLUE), strict=True)):
        bars = axes.bar(positions + width * (index - 0.5),
                        [methods[method][metric] for metric in metric_names],
                        width=width, color=color, label=NAMES[method])
        axes.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    axes.set_xticks(positions, labels=metric_names.values())
    axes.set_ylim(0, 1.08)
    axes.set_ylabel("Held-out score", color=INK)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2, frameon=False)
    figure.savefig(destination / "duplicate_method_comparison.png", dpi=180)

    figure, initial_axes = new_figure("Cosine similarity by curated label", (9.6, 4.8), caption)
    initial_axes.remove()
    panels = figure.subplots(ncols=2, sharey=True)
    minimum = min(record["similarity"] for rows in predictions.values() for record in rows)
    lower_bin = min(0, floor(20 * minimum))
    bins = np.arange(lower_bin, 21) / 20
    for axes, (method, name) in zip(panels, NAMES.items(), strict=True):
        for label, title, color, style in (
            ("non_duplicate", "Non-duplicate", BLUE, "solid"),
            ("duplicate", "Duplicate", GREEN, "dotted"),
        ):
            scores = [record["similarity"] for pair, record in zip(pairs, predictions[method], strict=True)
                      if pair["true_label"] == label]
            axes.hist(scores, bins=bins, histtype="step", linewidth=2, color=color,
                      linestyle=style, label=title)
        cutoff = methods[method]["threshold"]
        axes.axvline(cutoff, color=INK, lw=1.3, ls="--", label=f"Threshold {cutoff:.3f}")
        axes.set_xlim(bins[0] - 0.02, 1.02)
        axes.set_title(name, loc="left", fontsize=12, color=INK)
        axes.set_xlabel("Cosine similarity", color=INK)
        axes.tick_params(length=0, labelcolor=INK)
        axes.grid(axis="y", color=GRID)
        axes.set_axisbelow(True)
        axes.spines[["top", "right"]].set_visible(False)
        axes.legend(loc="upper left", facecolor="white", edgecolor="none", framealpha=1)
    panels[0].set_ylabel("Pairs per 0.05 interval", color=INK)
    figure.savefig(destination / "duplicate_similarity_distribution.png", dpi=180)
