"""Select both thresholds on development pairs, then evaluate the holdout."""

import csv
import hashlib
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from sklearn.metrics import f1_score

from backend.app.nlp.duplicates import (
    SBERT_BATCH_SIZE,
    SBERT_MODEL,
    SBERT_REVISION,
    TFIDF_OPTIONS,
    DuplicateDetector,
    fit_tfidf,
    load_sbert_model,
    sbert_similarity,
)
from ml.data.download_promise import SOURCE_URL
from ml.data.duplicates.build_curated_set import (
    DEVELOPMENT_PATH,
    EVALUATION_PATH,
    RECIPE_PATH,
    check_disjoint,
)
from ml.evaluation.classification_metrics import evaluate_predictions
from ml.evaluation.duplicate_figures import write_figures

RESULTS = Path(__file__).resolve().parents[2] / "research" / "results"
LABELS = {"non_duplicate": "Non-duplicate", "duplicate": "Duplicate"}
METRICS = ("accuracy", "precision", "recall", "f1")


def read_pairs(path, partition):
    with path.open(encoding="utf-8", newline="") as file:
        pairs = list(csv.DictReader(file))
    if not pairs or any(pair["partition"] != partition for pair in pairs):
        raise ValueError(f"Expected only {partition} pairs in {path.name}.")
    return pairs


def pair_texts(pairs):
    return [(pair["requirement_a"], pair["requirement_b"]) for pair in pairs]


def threshold_labels(scores, threshold):
    return ["duplicate" if score >= threshold else "non_duplicate" for score in scores]


def select_threshold(truth, scores):
    """Use positive-class F1 and prefer the higher threshold on equal F1."""
    best_f1, threshold = max(
        (
            f1_score(truth, threshold_labels(scores, threshold), pos_label="duplicate", zero_division=0),
            threshold,
        )
        for threshold in set(scores)
    )
    return float(threshold), float(best_f1)


def evaluate_method(pairs, detector):
    scores = detector.scorer.score_pairs(pair_texts(pairs))
    predicted = threshold_labels(scores, detector.threshold)
    report = evaluate_predictions([pair["true_label"] for pair in pairs], predicted, LABELS)
    positive = report["classification_report"]["duplicate"]
    return {
        "threshold": detector.threshold,
        "accuracy": report["test_metrics"]["accuracy"],
        "precision": positive["precision"],
        "recall": positive["recall"],
        "f1": positive["f1-score"],
        "confusion_matrix": report["confusion_matrix"],
    }, [
        {"prediction": prediction, "similarity": score}
        for prediction, score in zip(predicted, scores, strict=True)
    ]


def summarize_errors(rows):
    summary = {}
    for kind in ("false_positive", "false_negative"):
        errors = [row for row in rows if row["kind"] == kind]
        summary[kind] = {
            "count": len(errors),
            "counts_by_subtype": dict(Counter(row["subtype"] for row in errors)),
            "pair_ids": [row["pair_id"] for row in errors],
        }
    return summary


def write_outputs(results, pairs, predictions):
    records = []
    errors = {name: [] for name in results["methods"]}
    for index, pair in enumerate(pairs):
        record = {name: pair[name] for name in ("pair_id", "true_label", "subtype")}
        record["methods"] = {}
        for name, method_predictions in predictions.items():
            prediction = method_predictions[index]
            record["methods"][name] = prediction
            if prediction["prediction"] != pair["true_label"]:
                kind = "false_positive" if prediction["prediction"] == "duplicate" else "false_negative"
                errors[name].append({
                    **{key: pair[key] for key in ("pair_id", "subtype", "requirement_a", "requirement_b", "true_label")},
                    **prediction,
                    "method": name,
                    "kind": kind,
                })
        records.append(record)

    for filename, data in (
        ("duplicate_detection_results.json", results),
        ("duplicate_pair_predictions.json", records),
        ("duplicate_error_analysis.json", {name: summarize_errors(rows) for name, rows in errors.items()}),
    ):
        (RESULTS / filename).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")

    fields = (
        "pair_id", "method", "kind", "subtype", "requirement_a", "requirement_b",
        "true_label", "prediction", "similarity",
    )
    with (RESULTS / "duplicate_error_records.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fields, lineterminator="\n")
        writer.writeheader()
        for rows in errors.values():
            writer.writerows(rows)


def partition_details(pairs, path):
    return {
        "pairs": len(pairs),
        "class_counts": dict(Counter(pair["true_label"] for pair in pairs)),
        "subtype_counts": dict(Counter(pair["subtype"] for pair in pairs)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def projects(pairs):
    return {pair[key] for pair in pairs for key in ("project_id_a", "project_id_b")}


def main():
    development = read_pairs(DEVELOPMENT_PATH, "development")
    development_pairs = pair_texts(development)
    development_texts = list(dict.fromkeys(text for pair in development_pairs for text in pair))
    dev_truth = [pair["true_label"] for pair in development]
    tfidf = fit_tfidf(development_texts)
    model = load_sbert_model()
    scorers = (tfidf, sbert_similarity(model))
    selected = {}
    for scorer in scorers:
        threshold, dev_f1 = select_threshold(dev_truth, scorer.score_pairs(development_pairs))
        selected[scorer.method] = (DuplicateDetector(scorer, threshold), dev_f1)

    evaluation = read_pairs(EVALUATION_PATH, "evaluation")
    check_disjoint(development, evaluation)
    methods = {}
    predictions = {}
    for method, (detector, dev_f1) in selected.items():
        methods[method], predictions[method] = evaluate_method(evaluation, detector)
        methods[method]["development_f1"] = dev_f1
    differences = {metric: methods["sbert"][metric] - methods["tfidf"][metric] for metric in METRICS}

    recipe_bytes = RECIPE_PATH.read_bytes().replace(b"\r\n", b"\n")
    recipe = json.loads(recipe_bytes)
    results = {
        "dataset": {
            "name": recipe["dataset"],
            "source_url": SOURCE_URL,
            "source_corpus_sha256": recipe["source_sha256"],
            "recipe_sha256": hashlib.sha256(recipe_bytes).hexdigest(),
            "split_seed": recipe["split_seed"],
            "development": partition_details(development, DEVELOPMENT_PATH),
            "evaluation": partition_details(evaluation, EVALUATION_PATH),
            "projects": len(projects(development) | projects(evaluation)),
            "shared_projects": len(projects(development) & projects(evaluation)),
        },
        "threshold_selection": {
            "metric": "development F1 for duplicate",
            "candidates": "distinct development cosine scores",
            "ties": "higher threshold",
            "positive_when": "similarity >= threshold",
        },
        "configuration": {
            "tfidf": {"parameters": TFIDF_OPTIONS, "fit_text_count": len(development_texts)},
            "sbert": {
                "model": SBERT_MODEL,
                "revision": SBERT_REVISION,
                "device": str(model.device),
                "max_tokens": model.max_seq_length,
                "batch_size": SBERT_BATCH_SIZE,
            },
            "python": platform.python_version(),
            "versions": {
                package: version(package)
                for package in ("numpy", "scipy", "scikit-learn", "sentence-transformers",
                                "transformers", "torch", "matplotlib")
            },
        },
        "methods": methods,
        "comparison": {
            "sbert_minus_tfidf": differences,
            "absolute_difference": {metric: abs(delta) for metric, delta in differences.items()},
        },
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_outputs(results, evaluation, predictions)
    write_figures(RESULTS, evaluation, methods, predictions)
    print(json.dumps({key: results[key] for key in ("methods", "comparison")}, indent=2))


if __name__ == "__main__":
    main()
