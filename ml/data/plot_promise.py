"""Render the three static PROMISE_exp figures used in the lab report."""

from pathlib import Path
from statistics import median

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

INK = "#1b2c25"
GREEN = "#296044"
BLUE = "#486a80"
GRID = "#dce4df"


def new_chart(title, size, grid_axis, sample_size):
    figure = Figure(figsize=size, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    figure.get_layout_engine().set(w_pad=0.16, h_pad=0.12)
    figure.suptitle(title, x=0.025, ha="left", fontsize=15, fontweight="bold", color=INK)
    figure.supxlabel(
        f"PROMISE_exp · {sample_size:,} cleaned requirements",
        x=0.025,
        ha="left",
        fontsize=9,
        color="#4c6257",
    )
    axes = figure.subplots()
    axes.tick_params(length=0, labelcolor=INK)
    axes.spines[["top", "right"]].set_visible(False)
    axes.spines[["bottom", "left"]].set_color(GRID)
    axes.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    return figure, axes


def write_figures(
    results_dir: Path,
    names: dict[str, str],
    class_counts: dict[str, int],
    binary_counts: dict[str, int],
    lengths: list[int],
) -> None:
    sample_size = len(lengths)
    order = sorted(class_counts, key=lambda code: (-class_counts[code], code))
    counts = [class_counts[code] for code in order]
    labels = [f"{names[code]} ({code})" for code in order]
    fig, ax = new_chart("Requirements by category", (8.0, 5.4), "x", sample_size)
    bars = ax.barh(labels, counts, color=[GREEN if code == "F" else BLUE for code in order])
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.14)
    ax.set_xlabel("Number of requirements", color=INK)
    ax.bar_label(bars, padding=5, color=INK)
    fig.savefig(results_dir / "requirements_per_category.png", dpi=180)

    groups = ("Functional", "Non-Functional")
    counts = [binary_counts[group] for group in groups]
    fig, ax = new_chart("Functional and non-functional requirements", (6.8, 4.6), "y", sample_size)
    bars = ax.bar(groups, counts, width=0.55, color=[GREEN, BLUE])
    ax.set_ylim(0, max(counts) * 1.18)
    ax.set_ylabel("Number of requirements", color=INK)
    ax.bar_label(
        bars,
        labels=[f"{count} ({count / sum(counts):.1%})" for count in counts],
        padding=5,
        color=INK,
    )
    fig.savefig(results_dir / "functional_vs_nonfunctional.png", dpi=180)

    fig, ax = new_chart("Requirement length distribution", (7.4, 4.6), "y", sample_size)
    bin_width = 5
    last_edge = (max(lengths) // bin_width + 1) * bin_width
    ax.hist(lengths, bins=range(0, last_edge + 1, bin_width), color=BLUE, rwidth=0.95)
    center = median(lengths)
    ax.axvline(center, color=GREEN, linestyle="--", linewidth=1.7, label=f"Median {center:g} words")
    ax.set_xlim(0, last_edge)
    ax.set_xlabel("Words per requirement (5-word bins)", color=INK)
    ax.set_ylabel("Number of requirements", color=INK)
    ax.legend(loc="upper right", frameon=False, labelcolor=INK)
    fig.savefig(results_dir / "requirement_length_distribution.png", dpi=180)
