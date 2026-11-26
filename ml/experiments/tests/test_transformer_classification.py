import unittest

import torch
from transformers import DataCollatorWithPadding

from ml.evaluation.classification_metrics import evaluate_predictions
from ml.evaluation.transformer_results import compare_to_baseline, format_errors, load_baselines
from ml.experiments.classification_data import TASK_LABELS, read_split, target_labels
from ml.experiments.run_transformer_classification import RESULTS_DIR, class_weights, create_batch, load_tokenizer
from ml.experiments.transformer_data import MAX_LENGTH, batch_indices, decode_labels, label_ids, prepare_split


class TransformerClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = read_split("train")
        cls.tokenizer = load_tokenizer(local_files_only=True)

    def test_label_ids_round_trip_for_each_task(self):
        for task, names in TASK_LABELS.items():
            with self.subTest(task=task):
                ids = label_ids(task)
                self.assertEqual(list(ids), list(names))
                self.assertEqual(list(ids.values()), list(range(len(names))))
                self.assertEqual(decode_labels(list(ids.values()), task), list(names))

    def test_dataset_preparation_preserves_order_and_targets(self):
        rows = self.train[:9]
        for task in TASK_LABELS:
            encoded = prepare_split(rows, self.tokenizer, task)
            self.assertEqual(len(encoded), len(rows))
            self.assertEqual(decode_labels([row["labels"] for row in encoded], task), target_labels(rows, task))
            self.assertEqual(
                encoded[0]["input_ids"],
                self.tokenizer(rows[0].clean_text, truncation=True, max_length=MAX_LENGTH)["input_ids"],
            )

    def test_tokenizer_and_dynamic_padding(self):
        examples = prepare_split(self.train[:12], self.tokenizer, "binary")
        batches = batch_indices(examples, 5, shuffle_seed=42)
        self.assertCountEqual([i for batch in batches for i in batch], range(len(examples)))
        self.assertEqual(batches, batch_indices(examples, 5, shuffle_seed=42))
        collator = DataCollatorWithPadding(self.tokenizer)
        indices = batches[0]
        inputs, labels = create_batch(examples, indices, collator, torch.device("cpu"))
        self.assertEqual(tuple(labels.shape), (len(indices),))
        self.assertLessEqual(inputs["input_ids"].shape[1], MAX_LENGTH)
        for index, mask, ids in zip(indices, inputs["attention_mask"], inputs["input_ids"], strict=True):
            length = int(mask.sum())
            self.assertEqual(ids[:length].tolist(), examples[index]["input_ids"])
            self.assertEqual(ids[0].item(), self.tokenizer.cls_token_id)
            self.assertEqual(ids[length - 1].item(), self.tokenizer.sep_token_id)

    def test_training_only_class_weights(self):
        samples = prepare_split(self.train, self.tokenizer, "multiclass")
        device = torch.device("cpu")
        weights = class_weights(samples, "multiclass", device)
        ids = label_ids("multiclass")
        self.assertGreater(weights[ids["PO"]].item(), weights[ids["F"]].item())
        self.assertAlmostEqual(
            weights[[sample["labels"] for sample in samples]].mean().item(), 1.0, places=5
        )
        self.assertIsNone(class_weights(samples, "binary", device))

    def test_metrics_and_prediction_formatting(self):
        rows = [next(row for row in self.train if row.binary_label == label) for label in TASK_LABELS["binary"]]
        actual = target_labels(rows, "binary")
        decoded = decode_labels([0, 0], "binary")
        evaluated = evaluate_predictions(actual, decoded, TASK_LABELS["binary"])
        metrics = evaluated["test_metrics"]
        self.assertEqual(set(metrics), {"accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"})
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(evaluated["confusion_matrix"]["counts"], [[1, 0], [1, 0]])
        self.assertEqual(evaluated["classification_report"]["Non-Functional"]["f1-score"], 0.0)
        errors = format_errors(rows, actual, decoded, "binary")
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["requirement_text"], rows[1].requirement_text)
        self.assertEqual(errors[0]["true_label"], actual[1])
        self.assertEqual(errors[0]["predicted_label"], decoded[1])
        self.assertEqual(errors[0]["task"], "binary")

    def test_comparison_uses_frozen_phase3_baselines(self):
        baselines = load_baselines(RESULTS_DIR)
        self.assertEqual(baselines["binary"]["algorithm"], "Logistic Regression")
        self.assertEqual(baselines["multiclass"]["algorithm"], "Linear SVM")
        for task, baseline in baselines.items():
            comparison = compare_to_baseline(task, baseline, baseline)
            self.assertEqual(comparison["distilbert_minus_phase3"], {"accuracy": 0, "macro_f1": 0, "weighted_f1": 0})
            self.assertEqual(comparison["largest_confusion_changes"], [])


if __name__ == "__main__":
    unittest.main()
