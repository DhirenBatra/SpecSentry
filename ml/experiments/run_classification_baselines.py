"""Fit Phase 3 baselines, select on validation, and evaluate once on test."""

import csv
import hashlib
import json
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from itertools import product
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from ml.evaluation.classification_figures import write_task_figures
from ml.evaluation.classification_metrics import evaluate_predictions
from ml.experiments.classification_data import (
    PHASE2_COMMIT,
    SPLIT_INFO,
    TASK_LABELS,
    TASK_TARGETS,
    ensure_disjoint,
    read_split,
    target_labels,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "ml" / "artifacts"
RESULTS_DIR = REPO_ROOT / "research" / "results"
RESULTS_PATH = RESULTS_DIR / "classification_baseline_results.json"
ERRORS_PATH = RESULTS_DIR / "classification_test_errors.csv"

SEED = 42
VECTORIZER_FIXED = {
    "lowercase": False,
    "strip_accents": None,
    "token_pattern": r"\w+|[^\w\s]",
    "stop_words": None,
    "norm": "l2",
    "use_idf": True,
    "smooth_idf": True,
}
# These four options isolate n-grams, min_df, and sublinear TF with a small search.
TFIDF_OPTIONS = (
    ((1, 1), 1, False),
    ((1, 2), 1, False),
    ((1, 2), 2, False),
    ((1, 2), 2, True),
)
CLASS_WEIGHTS = (None, "balanced")
ERROR_FIELDS = (
    "source_row", "project_id", "requirement_text", "true_label", "predicted_label", "task", "model",
)


@dataclass(frozen=True)
class Algorithm:
    name: str
    estimator: type
    fixed_parameters: dict
    c_values: tuple[float, float]


ALGORITHMS = {
    "logistic_regression": Algorithm(
        "Logistic Regression",
        LogisticRegression,
        {"solver": "lbfgs", "max_iter": 1000, "random_state": SEED},
        (1.0, 4.0),
    ),
    "linear_svm": Algorithm(
        "Linear SVM",
        LinearSVC,
        {"dual": "auto", "max_iter": 10000, "random_state": SEED},
        (0.5, 2.0),
    ),
}


@dataclass(frozen=True)
class Selection:
    pipeline: Pipeline
    parameters: dict
    validation_macro_f1: float
    search: list[dict]


def candidates(algorithm_id: str):
    for tfidf, c_value, class_weight in product(
        TFIDF_OPTIONS, ALGORITHMS[algorithm_id].c_values, CLASS_WEIGHTS
    ):
        ngrams, min_df, sublinear_tf = tfidf
        yield {
            "tfidf__ngram_range": ngrams,
            "tfidf__min_df": min_df,
            "tfidf__sublinear_tf": sublinear_tf,
            "classifier__C": c_value,
            "classifier__class_weight": class_weight,
        }


def build_pipeline(algorithm_id: str, parameters: dict) -> Pipeline:
    algorithm = ALGORITHMS[algorithm_id]
    return Pipeline([
        ("tfidf", TfidfVectorizer(**VECTORIZER_FIXED)),
        ("classifier", algorithm.estimator(**algorithm.fixed_parameters)),
    ]).set_params(**parameters)


def select_pipeline(task: str, algorithm_id: str, train, validation) -> Selection:
    train_text = [row.clean_text for row in train]
    validation_text = [row.clean_text for row in validation]
    train_targets = target_labels(train, task)
    validation_targets = target_labels(validation, task)
    labels = list(TASK_LABELS[task])
    search = []
    best = None

    for parameters in candidates(algorithm_id):
        pipeline = build_pipeline(algorithm_id, parameters)
        pipeline.fit(train_text, train_targets)
        predictions = pipeline.predict(validation_text)
        score = float(f1_score(
            validation_targets, predictions, labels=labels, average="macro", zero_division=0
        ))
        search.append({"parameters": parameters, "validation_macro_f1": score})
        if best is None or score > best.validation_macro_f1:
            best = Selection(pipeline, parameters, score, search)

    if best is None:
        raise ValueError(f"No candidates configured for {algorithm_id}.")
    return best


def versions() -> dict[str, str]:
    packages = ("scikit-learn", "scipy", "numpy", "joblib", "threadpoolctl", "matplotlib")
    return {"python": platform.python_version(), **{name: version(name) for name in packages}}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_errors(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ERROR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_selected(
    task: str, algorithm_id: str, selection: Selection, test, validation,
    training_count: int, environment: dict,
) -> tuple[dict, list[dict]]:
    algorithm = ALGORITHMS[algorithm_id]
    labels = TASK_LABELS[task]
    truth = target_labels(test, task)
    predictions = selection.pipeline.predict([row.clean_text for row in test])
    evaluation = evaluate_predictions(truth, predictions, labels)

    artifact_path = ARTIFACTS_DIR / f"{task}_{algorithm_id}.joblib"
    metadata_path = artifact_path.with_suffix(".metadata.json")
    joblib.dump(selection.pipeline, artifact_path, compress=3)
    validation_sample = [row.clean_text for row in validation[:8]]
    loaded = joblib.load(artifact_path)
    if loaded.predict(validation_sample).tolist() != selection.pipeline.predict(validation_sample).tolist():
        raise ValueError(f"Reloaded model predictions differ for {artifact_path.name}.")

    metadata = {
        "task": task,
        "algorithm": algorithm.name,
        "selected_hyperparameters": selection.parameters,
        "fixed_parameters": {"tfidf": VECTORIZER_FIXED, "classifier": algorithm.fixed_parameters},
        "training_count": training_count,
        "labels": list(labels),
        "label_names": labels,
        "validation_macro_f1": selection.validation_macro_f1,
        "test_metrics": evaluation["test_metrics"],
        "artifact": artifact_path.relative_to(REPO_ROOT).as_posix(),
        "artifact_bytes": artifact_path.stat().st_size,
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "dataset_train_sha256": SPLIT_INFO["train"]["sha256"],
        "versions": environment,
    }
    write_json(metadata_path, metadata)
    error_rows = [
        {
            "source_row": row.source_row,
            "project_id": row.project_id,
            "requirement_text": row.requirement_text,
            "true_label": actual,
            "predicted_label": str(predicted),
            "task": task,
            "model": algorithm.name,
        }
        for row, actual, predicted in zip(test, truth, predictions, strict=True)
        if actual != predicted
    ]
    return {
        **metadata,
        **evaluation,
        "metadata": metadata_path.relative_to(REPO_ROOT).as_posix(),
        "validation_search": selection.search,
    }, error_rows


def run() -> None:
    train = read_split("train")
    validation = read_split("validation")
    ensure_disjoint({"train": train, "validation": validation})

    selected = {}
    for task in TASK_LABELS:
        selected[task] = {}
        for algorithm_id, algorithm in ALGORITHMS.items():
            selection = select_pipeline(task, algorithm_id, train, validation)
            selected[task][algorithm_id] = selection
            print(f"{task} {algorithm.name}: validation macro F1 {selection.validation_macro_f1:.4f}")

    # Test text is read only after all four fitted pipelines have been selected.
    test = read_split("test")
    ensure_disjoint({"train": train, "validation": validation, "test": test})
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    environment = versions()
    results = {
        "seed": SEED,
        "compute_threads": 1,
        "versions": environment,
        "dataset": {
            "name": "PROMISE_exp",
            "phase2_commit": PHASE2_COMMIT,
            "splits": SPLIT_INFO,
            "test_projects": len({row.project_id for row in test}),
            "portability_test_examples": sum(row.original_label == "PO" for row in test),
        },
        "selection": {
            "metric": "macro_f1",
            "fit_split": "train",
            "selection_split": "validation",
            "evaluation_split": "test",
            "tie_break": "First candidate (or algorithm for task ranking) in the declared order.",
            "validation_candidates_per_algorithm_task": len(TFIDF_OPTIONS) * len(CLASS_WEIGHTS) * 2,
        },
        "error_records": ERRORS_PATH.relative_to(REPO_ROOT).as_posix(),
        "tasks": {},
    }
    errors = []

    for task, models in selected.items():
        task_result = {
            "target_column": TASK_TARGETS[task],
            "best_by_validation": max(models, key=lambda identifier: models[identifier].validation_macro_f1),
            "models": {},
        }
        results["tasks"][task] = task_result
        for algorithm_id, selection in models.items():
            record, model_errors = evaluate_selected(
                task, algorithm_id, selection, test, validation, len(train), environment
            )
            task_result["models"][algorithm_id] = record
            errors.extend(model_errors)
            print(f"{task} {record['algorithm']}: held-out macro F1 {record['test_metrics']['macro_f1']:.4f}")
        task_result["figures"] = write_task_figures(
            RESULTS_DIR, task, TASK_LABELS[task], task_result["models"], len(test)
        )

    write_json(RESULTS_PATH, results)
    write_errors(ERRORS_PATH, errors)
    print(f"Saved four pipelines, {len(errors)} error records, and six figures.")
    print(f"Results written to {RESULTS_PATH}.")


def main() -> None:
    try:
        with threadpool_limits(limits=1):
            run()
    except (OSError, ValueError, csv.Error) as error:
        sys.exit(f"Classification baselines stopped. {error}")


if __name__ == "__main__":
    main()
