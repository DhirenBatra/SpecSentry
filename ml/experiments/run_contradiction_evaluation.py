"""Compare NLI directions on development, then evaluate the selected detector."""

import csv
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from backend.app.nlp.contradiction_rules import explicit_conflict
from backend.app.nlp.contradictions import (
    BATCH_SIZE,
    MAX_LENGTH,
    MODEL_NAME,
    MODEL_REVISION,
    ContradictionDetector,
    load_nli_model,
    max_contradiction_scores,
    predicted_label,
)
from ml.data.contradictions.build_curated_set import DEVELOPMENT_PATH, EVALUATION_PATH, RECIPE_PATH
from ml.data.download_promise import SOURCE_URL
from ml.data.duplicates.build_curated_set import check_disjoint
from ml.evaluation.classification_metrics import evaluate_predictions
from ml.evaluation.contradiction_figures import write_figures

RESULTS = Path(__file__).resolve().parents[2] / "research" / "results"
THREE_LABELS = {"contradiction": "Contradiction", "neutral": "Neutral", "entailment": "Entailment"}
BINARY_LABELS = {"not_contradiction": "Not contradiction", "contradiction": "Contradiction"}
MIN_BIDIRECTIONAL_F1_GAIN = 0.05


def _read_pairs(path, partition):
    with path.open(encoding="utf-8", newline="") as data:
        pairs = list(csv.DictReader(data))
    actual = Counter(pair["true_label"] for pair in pairs)
    if actual != dict.fromkeys(THREE_LABELS, 20) or any(pair["partition"] != partition for pair in pairs):
        raise ValueError(f"Expected 20 pairs per class in the {partition} partition.")
    return pairs


def _texts(pairs):
    return [(pair["requirement_a"], pair["requirement_b"]) for pair in pairs]


def _binary_label(label):
    return "contradiction" if label == "contradiction" else "not_contradiction"


def binary_report(truth, predictions):
    result = evaluate_predictions(
        [_binary_label(label) for label in truth],
        [_binary_label(label) for label in predictions],
        BINARY_LABELS,
    )
    positive = result["classification_report"]["contradiction"]
    return {
        "metrics": {
            "accuracy": result["test_metrics"]["accuracy"],
            "precision": positive["precision"],
            "recall": positive["recall"],
            "f1": positive["f1-score"],
        },
        "confusion_matrix": result["confusion_matrix"],
    }


def select_directionality(development):
    """Add the second pass only for a material improvement in development F1."""
    best_single = max(("forward", "reverse"), key=development.__getitem__)
    bidirectional_gain = development["max_contradiction"] - development[best_single]
    return "max_contradiction" if bidirectional_gain >= MIN_BIDIRECTIONAL_F1_GAIN else best_single


def evaluate_directions(encoder, pairs):
    texts = _texts(pairs)
    forward = encoder.score_pairs(texts)
    reverse = encoder.score_pairs([(second, first) for first, second in texts])
    combined = [max_contradiction_scores(a, b) for a, b in zip(forward, reverse, strict=True)]
    truth = [pair["true_label"] for pair in pairs]
    scores_by_direction = {"forward": forward, "reverse": reverse, "max_contradiction": combined}
    reports = {}
    labels_by_direction = {}
    for direction, rows in scores_by_direction.items():
        labels = [predicted_label(scores) for scores in rows]
        labels_by_direction[direction] = labels
        multiclass = evaluate_predictions(truth, labels, THREE_LABELS)["test_metrics"]
        reports[direction] = {
            "binary": binary_report(truth, labels),
            "three_class": {key: multiclass[key] for key in ("accuracy", "macro_f1")},
        }
    changed = sum(
        first != second
        for first, second in zip(labels_by_direction["forward"], labels_by_direction["reverse"], strict=True)
    )
    return reports, changed


def evaluate_holdout(detector, pairs):
    nli_results = detector.analyze_pairs(_texts(pairs))
    rules = [explicit_conflict(a, b) for a, b in _texts(pairs)]
    truth = [pair["true_label"] for pair in pairs]
    nli_labels = [result["label"] for result in nli_results]
    rule_labels = ["contradiction" if rule else "not_contradiction" for rule in rules]
    multiclass = evaluate_predictions(truth, nli_labels, THREE_LABELS)
    reports = {
        "nli_three_class": {
            "metrics": {
                key: multiclass["test_metrics"][key]
                for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1")
            },
            "per_class": {label: multiclass["classification_report"][label] for label in THREE_LABELS},
            "confusion_matrix": multiclass["confusion_matrix"],
        },
        "nli_binary": binary_report(truth, nli_labels),
        "rule_binary": binary_report(truth, rule_labels),
    }
    predictions = []
    for pair, nli, rule, rule_label in zip(pairs, nli_results, rules, rule_labels, strict=True):
        predictions.append({
            **{key: pair[key] for key in ("pair_id", "true_label", "family", "design")},
            "nli_label": nli["label"],
            "nli_binary_label": _binary_label(nli["label"]),
            "nli_scores": {label: nli[f"{label}_score"] for label in THREE_LABELS},
            "rule_binary_label": rule_label,
            "rule_match": rule,
        })
    return reports, predictions


def error_rows(pairs, predictions):
    for pair, record in zip(pairs, predictions, strict=True):
        actual = pair["true_label"]
        binary = _binary_label(actual)
        nli_score = record["nli_scores"]["contradiction"]
        for task, truth, prediction, score in (
            ("nli_three_class", actual, record["nli_label"], nli_score),
            ("nli_binary", binary, record["nli_binary_label"], nli_score),
            ("rule_binary", binary, record["rule_binary_label"], int(record["rule_match"] is not None)),
        ):
            if truth != prediction:
                yield {
                    **{key: pair[key] for key in ("pair_id", "family", "design", "requirement_a", "requirement_b")},
                    "method": task,
                    "true_label": truth,
                    "predicted_label": prediction,
                    "contradiction_score": score,
                }


def summarize_errors(errors, task_names):
    summary = {}
    for task in task_names:
        rows = [row for row in errors if row["method"] == task]
        summary[task] = {
            "count": len(rows),
            "by_true_and_predicted": dict(Counter(f"{row['true_label']} -> {row['predicted_label']}" for row in rows)),
            "by_family": dict(Counter(row["family"] for row in rows)),
            "by_design": dict(Counter(row["design"] for row in rows)),
            "pair_ids": [row["pair_id"] for row in rows],
        }
    return summary


def _partition_details(pairs, path):
    return {
        "pairs": len(pairs),
        "label_counts": dict(Counter(pair["true_label"] for pair in pairs)),
        "design_counts": dict(Counter(pair["design"] for pair in pairs)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _projects(pairs):
    return {pair[key] for pair in pairs for key in ("project_id_a", "project_id_b")}


def main():
    import torch

    development = _read_pairs(DEVELOPMENT_PATH, "development")
    encoder = load_nli_model()
    dev_reports, disagreements = evaluate_directions(encoder, development)
    strategy = select_directionality({name: report["binary"]["metrics"]["f1"] for name, report in dev_reports.items()})
    detector = ContradictionDetector(encoder.score_pairs, strategy)

    evaluation = _read_pairs(EVALUATION_PATH, "evaluation")
    check_disjoint(development, evaluation)
    held_out, predictions = evaluate_holdout(detector, evaluation)
    errors = list(error_rows(evaluation, predictions))
    recipe_bytes = RECIPE_PATH.read_bytes().replace(b"\r\n", b"\n")
    recipe = json.loads(recipe_bytes)
    nli_metrics = held_out["nli_binary"]["metrics"]
    rule_metrics = held_out["rule_binary"]["metrics"]
    results = {
        "dataset": {
            "name": recipe["dataset"],
            "source_url": SOURCE_URL,
            "source_corpus_sha256": recipe["source_sha256"],
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "split_seed": recipe["split_seed"],
            "development": _partition_details(development, DEVELOPMENT_PATH),
            "evaluation": _partition_details(evaluation, EVALUATION_PATH),
            "source_projects": len(_projects(development) | _projects(evaluation)),
            "shared_projects": len(_projects(development) & _projects(evaluation)),
        },
        "model": {
            "name": MODEL_NAME,
            "revision": MODEL_REVISION,
            "label2id": encoder.label_ids,
            "id2label": {index: label for label, index in encoder.label_ids.items()},
            "device": encoder.device,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "torch_threads": torch.get_num_threads(),
        },
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in ("torch", "transformers", "scikit-learn", "matplotlib")},
        },
        "directionality": {
            "decision": "Native NLI argmax",
            "selected": strategy,
            "minimum_bidirectional_f1_gain": MIN_BIDIRECTIONAL_F1_GAIN,
            "forward_reverse_label_disagreements": disagreements,
            "development": dev_reports,
        },
        "held_out": held_out,
        "nli_minus_rule_binary": {metric: nli_metrics[metric] - rule_metrics[metric] for metric in nli_metrics},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    for filename, data in (
        ("contradiction_detection_results.json", results),
        ("contradiction_pair_predictions.json", predictions),
        ("contradiction_error_analysis.json", summarize_errors(errors, held_out)),
    ):
        (RESULTS / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    fields = ("pair_id", "method", "family", "design", "requirement_a", "requirement_b",
              "true_label", "predicted_label", "contradiction_score")
    with (RESULTS / "contradiction_error_records.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(errors)
    write_figures(RESULTS, held_out, len(evaluation))
    print(f"Selected directionality: {strategy}")
    print(json.dumps({name: report["metrics"] for name, report in held_out.items()}, indent=2))


if __name__ == "__main__":
    main()
