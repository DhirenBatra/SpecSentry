# DistilBERT requirement classification

The experiment fine-tunes the public `distilbert-base-uncased` checkpoint independently for binary and 12-class PROMISE_exp classification. [Pretrained checkpoint at revision `12040ac`](https://huggingface.co/distilbert/distilbert-base-uncased/tree/12040accade4e8a0f71eabdb258fecc2e7e948be)

## Reproduce

Use Python 3.12 and the existing Phase 2 CSVs. The runner verifies their fingerprints and the committed Phase 3 results. From the repository root in PowerShell, install the appropriate PyTorch 2.15.0 build, then the rest of the pinned packages. For CPU:

```powershell
py -3.12 -m venv ml/.venv
.\ml\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.15.0
.\ml\.venv\Scripts\python.exe -m pip install -r ml/experiments/requirements-transformer.txt
```

For CUDA use the corresponding CUDA build. The runner detects CUDA automatically. Run both experiments with one command:

```powershell
.\ml\.venv\Scripts\python.exe -B -u -m ml.experiments.run_transformer_classification
```

On macOS or Linux, use `ml/.venv/bin/python`. The first run downloads public pretrained assets into the ignored transformer cache.

## Training protocol

- Fit on 677 training rows and rank checkpoints by macro F1 on 143 validation rows. Ties select the earliest epoch. The 148 test rows are opened after both checkpoints are selected and saved.
- Tokenize the existing `clean_text` with DistilBERT. The Phase 2 median was 15 words and the maximum was 86. The 128-token cap allows for WordPiece expansion and special tokens. Length-aware batches pad dynamically.
- Freeze the embeddings and first four transformer blocks. Fine-tune the final two blocks and classification head to limit CPU compute and optimizer memory. This trains approximately 14.8 million parameters per task.
- Use three epochs, batch size 16, seed 42, AdamW, peak learning rate `2e-5`, weight decay 0.01, 10% warmup followed by linear decay, and gradient clipping at 1.0.
- Use unweighted cross-entropy for binary. For multiclass, weight cross-entropy by the square root of inverse training class frequency, normalized to mean weight 1 across training examples.

## Measured results

The run used an Intel Core i5-1135G7 CPU with four PyTorch threads, PyTorch 2.15.0+cpu, and Transformers 5.24.1. Both tasks selected epoch 3. The longest inputs were 105, 41, and 59 tokens in train, validation, and test. The cap truncated zero examples.

| Task | Validation macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | 0.7588 | 0.6486 | 0.6895 | 0.6049 | 0.5795 | 0.6026 |
| Multiclass | 0.0759 | 0.4392 | 0.1196 | 0.0875 | 0.0585 | 0.2752 |

The signed changes subtract the Phase 3 score from the DistilBERT score. Both runs underperformed their classical baselines on all three comparison metrics.

| Task | Phase 3 baseline | Accuracy change | Macro F1 change | Weighted F1 change |
| --- | --- | ---: | ---: | ---: |
| Binary | Logistic Regression | -0.2230 | -0.2918 | -0.2696 |
| Multiclass | Linear SVM | -0.2838 | -0.3965 | -0.4285 |

## Error analysis

- Binary DistilBERT made 52 errors, compared with 19 for Logistic Regression. Functional → Non-Functional errors rose from 3 to 46. Non-Functional → Functional errors fell from 16 to 6. Functional recall was 0.2813.
- Multiclass DistilBERT made 83 errors, compared with 41 for Linear SVM. It predicted Functional for 147 of 148 test rows. Security → Functional rose from 4 to 19 and Usability → Functional from 3 to 19.
- Multiclass F1 was zero for Availability, Fault Tolerance, Legal, Look-and-feel, Maintainability, Operability, Performance, Portability, Scalability, and Usability. DistilBERT labeled the only Portability row Functional. The baseline labeled it Operability. The test set covers six projects, and one row provides limited evidence for Portability.

## Outputs and tests

The local `ml/artifacts/transformer_classification/binary/` and `multiclass/` folders contain the selected safetensors weights, tokenizers, configs, and metadata. Final evaluation reloaded both saved models and tokenizers. Git tracks the aggregate metadata, metrics, and figures. Weights, caches, and the original text in `research/results/transformer_classification_test_errors.csv` stay local.

The [results JSON](../../research/results/transformer_classification_results.json) contains every category's precision, recall, F1 and support, plus confusion counts, signed and absolute differences, version information, and measured training histories.

| Task | Confusion matrix | Baseline comparison |
| --- | --- | --- |
| Binary | [Figure](../../research/results/binary_distilbert_confusion_matrix.png) | [Figure](../../research/results/binary_distilbert_baseline_comparison.png) |
| Multiclass | [Figure](../../research/results/multiclass_distilbert_confusion_matrix.png) | [Figure](../../research/results/multiclass_distilbert_baseline_comparison.png) |

[Training loss and validation macro F1 by epoch](../../research/results/distilbert_training_validation.png)

After the pretrained tokenizer is cached, run the lightweight tests:

```powershell
.\ml\.venv\Scripts\python.exe -B -m unittest discover -s ml/experiments/tests -p "test_transformer_classification.py" -v
```
