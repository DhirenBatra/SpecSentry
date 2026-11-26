"""Calculate the held-out metrics and summarize observed mistakes."""

from collections import Counter

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_predictions(truth, predictions, label_names: dict[str, str]) -> dict:
    labels = list(label_names)
    report = classification_report(
        truth,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
    }
    confusing_pairs = Counter(
        (actual, str(predicted))
        for actual, predicted in zip(truth, predictions, strict=True)
        if actual != predicted
    )
    ranked_pairs = sorted(confusing_pairs.items(), key=lambda item: (-item[1], item[0]))
    worst_f1 = min(report[label]["f1-score"] for label in labels)
    return {
        "test_metrics": metrics,
        "classification_report": report,
        "confusion_matrix": {
            "label_order": labels,
            "counts": confusion_matrix(truth, predictions, labels=labels).tolist(),
        },
        "error_summary": {
            "incorrect_predictions": sum(confusing_pairs.values()),
            "most_frequent_confusions": [
                {"true_label": actual, "predicted_label": predicted, "count": count}
                for (actual, predicted), count in ranked_pairs[:5]
            ],
            "worst_f1_classes": {
                label: report[label]
                for label in labels
                if report[label]["f1-score"] == worst_f1
            },
            "classes_with_at_most_five_test_examples": {
                label: report[label] for label in labels if report[label]["support"] <= 5
            },
        },
    }
