"""Compare held-out DistilBERT metrics and errors with the frozen Phase 3 baselines."""

import hashlib
import json

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from ml.evaluation.classification_figures import (
    BLUE,
    GREEN,
    GRID,
    INK,
    comparison_figure,
    confusion_figure,
)
from ml.experiments.classification_data import PHASE2_COMMIT, SPLIT_INFO, TASK_LABELS

PHASE3_COMMIT = "f76f57c25837f4e0f85622e579a1a3660caff297"
PHASE3_SHA256 = "9c54d1e673e248679f782df7c608d97e65ab379377866880a8c5857eb52f0f5f"
BASELINES = {"binary": "logistic_regression", "multiclass": "linear_svm"}
METRICS = ("accuracy", "macro_f1", "weighted_f1")


def load_baselines(results_dir):
    data = (results_dir / "classification_baseline_results.json").read_bytes()
    # Git for Windows may check out tracked JSON using CRLF.
    if hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest() != PHASE3_SHA256:
        raise ValueError("The frozen Phase 3 results checksum does not match.")
    results = json.loads(data)
    if results["dataset"]["splits"] != SPLIT_INFO or results["dataset"]["phase2_commit"] != PHASE2_COMMIT:
        raise ValueError("Classical and transformer comparisons require the same frozen data.")
    return {task: results["tasks"][task]["models"][model] for task, model in BASELINES.items()}


def compare_to_baseline(task, evaluated, baseline):
    scores = evaluated["test_metrics"]
    baseline_scores = baseline["test_metrics"]
    differences = {metric: scores[metric] - baseline_scores[metric] for metric in METRICS}
    labels = list(TASK_LABELS[task])
    current = evaluated["confusion_matrix"]
    previous = baseline["confusion_matrix"]
    if current["label_order"] != labels or previous["label_order"] != labels:
        raise ValueError("Comparison requires the same category order.")

    changes = []
    for i, true_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):
            if i == j:
                continue
            old = previous["counts"][i][j]
            new = current["counts"][i][j]
            if old != new:
                changes.append({
                    "true_label": true_label, "predicted_label": predicted_label,
                    "phase3_count": old, "distilbert_count": new, "difference": new - old,
                })
    changes.sort(key=lambda pair: -abs(pair["difference"]))
    rare = evaluated["error_summary"]["classes_with_at_most_five_test_examples"]
    return {
        "phase3_algorithm": baseline["algorithm"],
        "phase3_test_metrics": baseline_scores,
        "distilbert_minus_phase3": differences,
        "absolute_difference_magnitude": {metric: abs(delta) for metric, delta in differences.items()},
        "phase3_incorrect_predictions": baseline["error_summary"]["incorrect_predictions"],
        "distilbert_incorrect_predictions": evaluated["error_summary"]["incorrect_predictions"],
        "phase3_top_confusions": baseline["error_summary"]["most_frequent_confusions"],
        "largest_confusion_changes": changes[:5],
        "rare_class_f1": {
            label: {
                "support": score["support"],
                "phase3": baseline["classification_report"][label]["f1-score"],
                "distilbert": score["f1-score"],
            }
            for label, score in rare.items()
        },
    }


def format_errors(rows, actual, predicted, task):
    return [
        {"source_row": row.source_row, "project_id": row.project_id,
         "requirement_text": row.requirement_text, "true_label": truth,
         "predicted_label": prediction, "task": task, "model": "DistilBERT"}
        for row, truth, prediction in zip(rows, actual, predicted, strict=True)
        if truth != prediction
    ]


def write_transformer_figures(results_dir, tasks, baselines, test_size):
    filenames = []
    for task, entry in tasks.items():
        matrix_name = f"{task}_distilbert_confusion_matrix.png"
        comparison_name = f"{task}_distilbert_baseline_comparison.png"
        matrix = entry["confusion_matrix"]["counts"]
        confusion_figure(
            results_dir / matrix_name, task, "DistilBERT", matrix,
            TASK_LABELS[task], max(map(max, matrix)), test_size,
        )
        comparison_figure(
            results_dir / comparison_name, task,
            {"baseline": baselines[task], "distilbert": {
                "algorithm": "DistilBERT", "test_metrics": entry["test_metrics"]
            }},
            test_size,
        )
        filenames.extend((matrix_name, comparison_name))

    figure = Figure(figsize=(9.0, 4.6), layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    figure.suptitle("DistilBERT training and validation", fontsize=15, fontweight="bold", color=INK)
    figure.supxlabel("PROMISE_exp · Dots mark epoch measurements", fontsize=9, color=INK)
    left, right = figure.subplots(1, 2)
    for (task, entry), color in zip(tasks.items(), (GREEN, BLUE), strict=True):
        history = entry["training_history"]
        epochs = [epoch["epoch"] for epoch in history]
        left.plot(epochs, [epoch["training_loss"] for epoch in history], "o-", color=color, label=task.title())
        right.plot(epochs, [epoch["validation_macro_f1"] for epoch in history], "o-", color=color, label=task.title())
    left.set_ylabel("Training loss")
    right.set_ylabel("Validation macro F1")
    right.set_ylim(0, 1)
    for axes in (left, right):
        axes.set_xticks(epochs)
        axes.set_xlabel("Epoch")
        axes.grid(axis="y", color=GRID)
        axes.legend(frameon=False)
    history_name = "distilbert_training_validation.png"
    figure.savefig(results_dir / history_name, dpi=180)
    return [*filenames, history_name]
