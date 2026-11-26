"""Render the six held-out classification figures."""

from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

INK = "#1b2c25"
MUTED = "#51675c"
GREEN = "#296044"
BLUE = "#486a80"
GRID = "#dce4df"
TASK_TITLES = {"binary": "Binary classification", "multiclass": "12-class classification"}
COMPARISON_METRICS = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "weighted_f1": "Weighted F1"}


def new_figure(title, size, caption):
    figure = Figure(figsize=size, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    figure.get_layout_engine().set(w_pad=0.18, h_pad=0.14)
    figure.suptitle(title, x=0.026, ha="left", fontsize=15, fontweight="bold", color=INK)
    figure.supxlabel(caption, x=0.026, ha="left", fontsize=9, color=MUTED)
    axes = figure.subplots()
    axes.tick_params(length=0, labelcolor=INK)
    return figure, axes


def confusion_figure(
    destination: Path, task: str, algorithm: str, matrix: list[list[int]],
    label_names: dict[str, str], maximum: int, sample_size: int,
) -> None:
    multiclass = task == "multiclass"
    title = f"{TASK_TITLES[task]}: {algorithm}"
    caption = f"PROMISE_exp held-out test set · {sample_size} requirements · Confusion counts"
    size = (10.2, 7.9) if multiclass else (6.9, 5.5)
    figure, axes = new_figure(title, size, caption)
    image = axes.imshow(matrix, cmap="Greens", vmin=0, vmax=maximum, interpolation="nearest")

    codes = list(label_names)
    tick_positions = range(len(codes))
    true_names = [f"{label_names[code]} ({code})" for code in codes] if multiclass else codes
    axes.set_xticks(tick_positions, labels=codes)
    axes.set_yticks(tick_positions, labels=true_names)
    axes.set_xlabel("Predicted category code" if multiclass else "Predicted label", color=INK)
    axes.set_ylabel("True category" if multiclass else "True label", color=INK)
    for spine in axes.spines.values():
        spine.set_color(GRID)
    scale = figure.colorbar(image, ax=axes, fraction=0.037, pad=0.026)
    scale.set_label("Requirements", color=INK)
    scale.ax.tick_params(labelcolor=INK)

    for true_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            text_color = "white" if count >= 0.6 * maximum else (INK if count else MUTED)
            axes.text(
                predicted_index, true_index, str(count),
                ha="center", va="center", color=text_color,
                fontsize=9 if multiclass else 13,
            )
    figure.savefig(destination, dpi=180)


def comparison_figure(destination: Path, task: str, models: dict, sample_size: int) -> None:
    figure, axes = new_figure(
        f"{TASK_TITLES[task]}: Model comparison",
        (7.5, 4.9),
        f"PROMISE_exp held-out test set · {sample_size} requirements",
    )
    positions = range(len(COMPARISON_METRICS))
    width = 0.34
    for index, (model, color) in enumerate(zip(models.values(), (GREEN, BLUE), strict=True)):
        offset = width * (index - 0.5)
        values = [model["test_metrics"][metric] for metric in COMPARISON_METRICS]
        bars = axes.bar(
            [position + offset for position in positions], values, width=width,
            color=color, label=model["algorithm"],
        )
        axes.bar_label(bars, fmt="%.3f", padding=4, color=INK)

    axes.set_xticks(positions, labels=list(COMPARISON_METRICS.values()))
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes.set_ylim(0, 1.07)
    axes.set_ylabel("Test score", color=INK)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["bottom", "left"]].set_color(GRID)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2, frameon=False)
    figure.savefig(destination, dpi=180)


def write_task_figures(results_dir: Path, task: str, label_names: dict, models: dict, sample_size: int) -> list[str]:
    maximum = max(
        count
        for model in models.values()
        for row in model["confusion_matrix"]["counts"]
        for count in row
    )
    filenames = []
    for identifier, model in models.items():
        filename = f"{task}_{identifier}_confusion_matrix.png"
        confusion_figure(
            results_dir / filename, task, model["algorithm"],
            model["confusion_matrix"]["counts"], label_names, maximum, sample_size,
        )
        filenames.append(filename)
    filename = f"{task}_model_comparison.png"
    comparison_figure(results_dir / filename, task, models, sample_size)
    filenames.append(filename)
    return filenames
