"""Evaluate the deterministic ambiguity detector on the fixed curated set."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from backend.app.nlp.ambiguity import WEIGHTS, detect_ambiguity
from ml.data.ambiguity.build_curated_set import DATASET_PATH, RECIPE_PATH, build_examples, write_dataset
from ml.evaluation.classification_figures import BLUE, GREEN, GRID, INK, new_figure
from ml.evaluation.classification_metrics import evaluate_predictions

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research" / "results"
LABELS = {"clear": "Clear", "ambiguous": "Ambiguous"}
METRICS = ("precision", "recall", "f1")
METHODS = {"full": "Full detector", "vague_only": "Vague terms only"}
METRICS_PATH = RESULTS / "ambiguity_detection_results.json"
ANALYSIS_PATH = RESULTS / "ambiguity_error_analysis.json"
ERRORS_PATH = RESULTS / "ambiguity_error_records.csv"
MATRIX_PATH = RESULTS / "ambiguity_confusion_matrix.png"
ABLATION_PATH = RESULTS / "ambiguity_ablation.png"


def evaluate_cases(cases, *, vague_only=False):
    detected = [detect_ambiguity(case["requirement"], vague_only=vague_only) for case in cases]
    truth = [case["label"] for case in cases]
    predicted = ["ambiguous" if result["ambiguous"] else "clear" for result in detected]
    existing_metrics = evaluate_predictions(truth, predicted, LABELS)
    positive = existing_metrics["classification_report"]["ambiguous"]
    matrix = existing_metrics["confusion_matrix"]
    (_, fp), (fn, _) = matrix["counts"]
    error_rows = []
    for case, prediction, result in zip(cases, predicted, detected, strict=True):
        if case["label"] != prediction:
            error_rows.append({
                **case, "prediction": prediction,
                "kind": "false_positive" if prediction == "ambiguous" else "false_negative",
                "score": result["score"], "issues": result["issues"],
            })
    return {
        "accuracy": existing_metrics["test_metrics"]["accuracy"],
        "precision": positive["precision"],
        "recall": positive["recall"],
        "f1": positive["f1-score"],
        "confusion_matrix": matrix,
        "false_positives": fp,
        "false_negatives": fn,
    }, error_rows


def summarize_errors(errors):
    return {
        kind: {
            "counts_by_case_design": dict(Counter(case["challenge"] or case["focus"] for case in errors if case["kind"] == kind)),
            "cases": [
                {key: case[key] for key in ("id", "pair_id", "source_row", "project_id", "focus", "challenge",
                                          "label", "prediction", "score", "issues")}
                for case in errors if case["kind"] == kind
            ],
        }
        for kind in ("false_positive", "false_negative")
    }


def write_errors(errors_by_method):
    fields = ("method", "id", "pair_id", "source_row", "project_id", "focus", "challenge", "label",
              "prediction", "kind", "score", "issues", "requirement")
    with ERRORS_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        for method, errors in errors_by_method.items():
            for error in errors:
                writer.writerow({**error, "method": method, "issues": json.dumps(error["issues"], ensure_ascii=False)})


def write_figures(scores, sample_size):
    caption = f"Curated ambiguity evaluation set · {sample_size} requirements · {sample_size // 2} pairs"
    figure, axes = new_figure("Ambiguity detection confusion matrix", (6.9, 5.5), caption)
    confusion = scores["full"]["confusion_matrix"]
    display = ConfusionMatrixDisplay(
        confusion_matrix=np.array(confusion["counts"]),
        display_labels=list(LABELS.values()),
    )
    display.plot(ax=axes, cmap="Greens", colorbar=False, values_format="d")
    axes.set_xlabel("Predicted label", color=INK)
    axes.set_ylabel("Curated label", color=INK)
    figure.savefig(MATRIX_PATH, dpi=180)

    figure, axes = new_figure("Ambiguity rule ablation", (7.5, 4.9), caption)
    width = 0.34
    positions = range(len(METRICS))
    for index, (method, color) in enumerate(zip(METHODS, (GREEN, BLUE), strict=True)):
        bars = axes.bar(
            [position + width * (index - 0.5) for position in positions],
            [scores[method][metric] for metric in METRICS],
            width=width, color=color, label=METHODS[method],
        )
        axes.bar_label(bars, fmt="%.3f", padding=4, color=INK)
    axes.set_xticks(positions, labels=["Precision", "Recall", "F1"])
    axes.set_yticks([0, 0.25, 0.5, 0.75, 1])
    axes.set_ylim(0, 1.08)
    axes.set_ylabel("Score for the ambiguous class", color=INK)
    axes.set_axisbelow(True)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.spines[["top", "right"]].set_visible(False)
    axes.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncols=2, frameon=False)
    figure.savefig(ABLATION_PATH, dpi=180)


def main():
    recipe, cases = build_examples()
    write_dataset(cases)
    RESULTS.mkdir(parents=True, exist_ok=True)
    scores = {}
    errors = {}
    for method in METHODS:
        scores[method], errors[method] = evaluate_cases(cases, vague_only=method == "vague_only")

    full, vague = scores["full"], scores["vague_only"]
    metrics = {
        "dataset": {
            "name": recipe["dataset"],
            "cases": len(cases),
            "pairs": len(recipe["pairs"]),
            "distinct_projects": len({case["project_id"] for case in cases}),
            "class_counts": dict(Counter(case["label"] for case in cases)),
            "challenge_pair_count": sum("challenge" in pair for pair in recipe["pairs"]),
            "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
            "recipe_sha256": hashlib.sha256(RECIPE_PATH.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "source_commit": recipe["source_commit"],
            "source_sha256": recipe["source_sha256"],
        },
        "positive_label": "ambiguous",
        "signal_weights": WEIGHTS,
        "methods": scores,
        "ablation": {
            "full_minus_vague_only": {metric: full[metric] - vague[metric] for metric in METRICS},
            "additional_true_positives": vague["false_negatives"] - full["false_negatives"],
            "additional_false_positives": full["false_positives"] - vague["false_positives"],
        },
    }
    for path, result in [(METRICS_PATH, metrics), (ANALYSIS_PATH, {name: summarize_errors(rows) for name, rows in errors.items()})]:
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    write_errors(errors)
    write_figures(scores, len(cases))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
