import csv
import hashlib
import json
import unittest

import joblib
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from ml.evaluation.classification_metrics import evaluate_predictions
from ml.experiments.classification_data import TASK_LABELS, ensure_disjoint, read_split, target_labels
from ml.experiments.run_classification_baselines import (
    ALGORITHMS,
    ARTIFACTS_DIR,
    ERRORS_PATH,
    RESULTS_PATH,
    build_pipeline,
    candidates,
)

REQUIRED_METRICS = {"accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"}


class ClassificationBaselinesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_split("train")
        cls.validation = read_split("validation")
        cls.train_text = [row.clean_text for row in cls.train]
        cls.validation_text = [row.clean_text for row in cls.validation]
        cls.results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    def test_expected_labels_in_frozen_train_and_validation(self):
        ensure_disjoint({"train": self.train, "validation": self.validation})
        self.assertEqual(len(TASK_LABELS["multiclass"]), 12)
        for task, names in TASK_LABELS.items():
            for split in (self.train, self.validation):
                self.assertEqual(set(target_labels(split, task)), set(names))

    def test_each_pipeline_fits_and_predicts_the_expected_shape(self):
        with threadpool_limits(limits=1):
            for task, names in TASK_LABELS.items():
                for identifier, algorithm in ALGORITHMS.items():
                    with self.subTest(task=task, algorithm=identifier):
                        pipeline = build_pipeline(identifier, next(candidates(identifier)))
                        pipeline.fit(self.train_text, target_labels(self.train, task))
                        predictions = pipeline.predict(self.validation_text)
                        self.assertIsInstance(pipeline, Pipeline)
                        self.assertIsInstance(pipeline.named_steps["classifier"], algorithm.estimator)
                        self.assertEqual(predictions.shape, (len(self.validation),))
                        self.assertEqual(set(pipeline.classes_), set(names))
                        self.assertTrue(set(predictions) <= set(names))

    def test_tokenization_keeps_punctuation_and_case(self):
        pipeline = build_pipeline("linear_svm", next(candidates("linear_svm")))
        analyze = pipeline.named_steps["tfidf"].build_analyzer()
        self.assertEqual(analyze("Keep v2.1 <= 95%"), ["Keep", "v2", ".", "1", "<", "=", "95", "%"])

    def test_evaluation_has_required_metrics_and_per_class_support(self):
        pipeline = joblib.load(ARTIFACTS_DIR / "multiclass_logistic_regression.joblib")
        evaluation = evaluate_predictions(
            target_labels(self.validation, "multiclass"),
            pipeline.predict(self.validation_text),
            TASK_LABELS["multiclass"],
        )
        self.assertEqual(set(evaluation["test_metrics"]), REQUIRED_METRICS)
        self.assertTrue(all(0 <= score <= 1 for score in evaluation["test_metrics"].values()))
        report = evaluation["classification_report"]
        self.assertTrue(set(TASK_LABELS["multiclass"]) <= report.keys())
        self.assertEqual(sum(report[label]["support"] for label in TASK_LABELS["multiclass"]), 143)
        self.assertEqual(sum(map(sum, evaluation["confusion_matrix"]["counts"])), 143)

    def test_all_four_saved_pipelines_load_and_predict(self):
        for task, result in self.results["tasks"].items():
            self.assertEqual(set(result["models"]), set(ALGORITHMS))
            for identifier, model in result["models"].items():
                with self.subTest(task=task, algorithm=identifier):
                    path = ARTIFACTS_DIR / f"{task}_{identifier}.joblib"
                    metadata = json.loads(path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
                    self.assertEqual(metadata["artifact_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                    self.assertEqual(metadata["training_count"], 677)
                    self.assertEqual(metadata["test_metrics"], model["test_metrics"])
                    self.assertEqual(metadata["selected_hyperparameters"], model["selected_hyperparameters"])

                    pipeline = joblib.load(path)
                    predictions = pipeline.predict(self.validation_text)
                    self.assertEqual(predictions.shape, (143,))
                    self.assertEqual(set(pipeline.classes_), set(TASK_LABELS[task]))
                    score = f1_score(
                        target_labels(self.validation, task), predictions,
                        labels=list(TASK_LABELS[task]), average="macro", zero_division=0,
                    )
                    self.assertAlmostEqual(score, metadata["validation_macro_f1"])

    def test_selection_records_and_error_file_match_the_evaluation(self):
        expected_errors = 0
        for result in self.results["tasks"].values():
            for model in result["models"].values():
                trials = model["validation_search"]
                self.assertEqual(len(trials), 16)
                best = max(trials, key=lambda trial: trial["validation_macro_f1"])
                self.assertEqual(model["selected_hyperparameters"], best["parameters"])
                self.assertEqual(model["validation_macro_f1"], best["validation_macro_f1"])
                self.assertEqual(sum(map(sum, model["confusion_matrix"]["counts"])), 148)
                expected_errors += model["error_summary"]["incorrect_predictions"]

        with ERRORS_PATH.open(encoding="utf-8", newline="") as file:
            errors = list(csv.DictReader(file))
        self.assertEqual(len(errors), expected_errors)
        self.assertTrue(all(row["requirement_text"] for row in errors))
        self.assertTrue(all(row["true_label"] != row["predicted_label"] for row in errors))


if __name__ == "__main__":
    unittest.main()
