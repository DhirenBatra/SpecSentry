# Classical requirement classification

## Reproduce the experiment

Use Python 3.12 and the existing Phase 2 split CSVs in `ml/data/processed/`. The runner verifies their SHA-256 fingerprints and opens them read-only. If an input is missing or differs from the frozen snapshot, restore the original split before running.

From the repository root in PowerShell, install the pinned dependencies into the ML environment once:

```powershell
py -3.12 -m venv ml/.venv
.\ml\.venv\Scripts\python.exe -m pip install -r ml/experiments/requirements.txt
```

Run the complete experiment with one command:

```powershell
.\ml\.venv\Scripts\python.exe -B -m ml.experiments.run_classification_baselines
```

On macOS or Linux, create the environment with `python3.12 -m venv ml/.venv` and use `ml/.venv/bin/python`. The results record the library versions, split fingerprints, seed, and all validation trials.

## Fitting and selection

Each scikit-learn pipeline uses the Phase 2 `clean_text` field. The vectorizer preserves case and tokenizes word runs and individual punctuation marks. TF-IDF and the classifier fit on the 677 training rows. Candidate and algorithm selection use macro F1 on the 143 validation rows. The runner selects all four fitted models before reading the 148 held-out test rows. The saved pipelines retain their train-only fits.

The grid uses four TF-IDF settings. In order, these are unigrams with `min_df=1`, unigrams plus bigrams with `min_df=1`, unigrams plus bigrams with `min_df=2`, and that last setting with sublinear TF enabled. Each combines with two `C` values and unweighted or balanced classes, giving 16 candidates per algorithm and task. Logistic Regression tries `C=1.0` and `4.0`. Linear SVM tries `C=0.5` and `2.0`. The run uses seed 42 and one compute thread. Ties keep the first candidate in the declared order.

| Task | Classifier | Terms | `min_df` | Sublinear TF | `C` | Class weights |
| --- | --- | --- | ---: | --- | ---: | --- |
| Binary | Logistic Regression | Unigrams + bigrams | 2 | Yes | 1.0 | balanced |
| Binary | Linear SVM | Unigrams + bigrams | 1 | No | 0.5 | balanced |
| Multiclass | Logistic Regression | Unigrams | 1 | No | 4.0 | balanced |
| Multiclass | Linear SVM | Unigrams + bigrams | 1 | No | 0.5 | balanced |

## Held-out error analysis

Validation selected Logistic Regression for binary and Linear SVM for multiclass. The [structured results](../../research/results/classification_baseline_results.json) contain all five test metrics, the complete classification reports with precision, recall, F1 and support for each label, the confusion counts, and ranked errors.

| Task | Classifier | Incorrect predictions | Most frequent mistake (true → predicted) | Lowest-F1 classes |
| --- | --- | ---: | --- | --- |
| Binary | Logistic Regression | 19 | Non-Functional → Functional (16) | Functional (0.8652) |
| Binary | Linear SVM | 21 | Non-Functional → Functional (12) | Functional (0.8397) |
| Multiclass | Logistic Regression | 60 | Security → Functional (5) | Legal, Portability, Scalability (all 0.0000) |
| Multiclass | Linear SVM | 41 | Security → Functional (4) | Fault Tolerance, Legal, Portability (all 0.0000) |

Both multiclass pipelines predicted Operability for the only Portability test row. Legal had two test rows. The test set covers six projects, and the multiclass macro F1 scores fell from 0.6165 to 0.3449 for Logistic Regression and from 0.6514 to 0.4550 for Linear SVM between validation and test. The held-out scores describe this fixed set of projects, with especially limited evidence for the rare labels.

## Outputs

- `ml/artifacts/` holds the four fitted `.joblib` pipelines and a `.metadata.json` sidecar for each. Use the pinned environment to load these locally generated files with `joblib.load()` and call `.predict()` on cleaned requirements.
- `research/results/classification_test_errors.csv` holds the 141 incorrect predictions, one record per model and mistake, including the original text, true and predicted labels, task, and model. The text-bearing error CSV and model artifacts are local. Git tracks the aggregate results and figures.

| Task | Logistic Regression confusion matrix | Linear SVM confusion matrix | Model comparison |
| --- | --- | --- | --- |
| Binary | [Figure](../../research/results/binary_logistic_regression_confusion_matrix.png) | [Figure](../../research/results/binary_linear_svm_confusion_matrix.png) | [Figure](../../research/results/binary_model_comparison.png) |
| Multiclass | [Figure](../../research/results/multiclass_logistic_regression_confusion_matrix.png) | [Figure](../../research/results/multiclass_linear_svm_confusion_matrix.png) | [Figure](../../research/results/multiclass_model_comparison.png) |

After generating the outputs, run the Phase 3 tests from the repository root:

```powershell
.\ml\.venv\Scripts\python.exe -B -m unittest discover -s ml/experiments/tests -v
```
