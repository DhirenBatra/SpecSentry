# NLI-based requirement contradiction detection

Natural Language Inference assigns a relationship to a premise and hypothesis. Contradiction means the premise conflicts with the hypothesis. Entailment means it supports the hypothesis. Neutral means the relationship is underdetermined by the premise. A cross-encoder reads both requirements together, so reversing the pair can change the scores.

## Model and baseline

The pinned checkpoint is `cross-encoder/nli-MiniLM2-L6-H768` at revision `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d`. It has six encoder layers and was trained on SNLI and MultiNLI. The loader checks that both mappings in the configuration agree before loading the tokenizer and model. [Checkpoint and training data](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/tree/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d) · [Pinned config](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/blob/b95119ce93d3e065de6214e38cd4a97b0f2f2c6d/config.json)

| Output index | Label |
| ---: | --- |
| 0 | contradiction |
| 1 | entailment |
| 2 | neutral |

Inference uses the original checkpoint in eval mode with gradients disabled. Requirements are tokenized jointly, up to 256 total tokens, in batches of 16. The loader selects CUDA if available and CPU otherwise. Each result returns the model's three softmax scores, the argmax label, and the pair. Softmax scores are uncalibrated.

The baseline flags two cases. It matches opposite deontic modals such as `must` and `must not` when the subject and remaining clause match. It also checks whether two explicit numeric intervals for the same required action, units, and wording have an empty intersection. It skips numeric clauses containing conditional cues. It treats `may` as permission, requires matching units, and leaves antonyms, role implication, and condition reasoning to the NLI comparison.

## Directionality and split

The anchored 50/50 split has 60 development and 60 held-out pairs. Both partitions have 20 examples of each label. All variants and any donor of a source row stay together. [Curated evaluation set](../data/contradictions/README.md)

The experiment scores A → B and B → A separately on development. The combined strategy keeps the full score vector from the direction with the larger contradiction score, choosing A → B on equal scores. All strategies use native argmax. The experiment picks the higher-F1 single direction for the binary contradiction task, preferring A → B on ties. It uses both directions only if their combination raises development F1 by at least 0.05 above the stronger single direction. The runner freezes this choice before opening the held-out data.

| Development strategy | Binary precision | Binary recall | Binary F1 | Three-class macro F1 |
| --- | ---: | ---: | ---: | ---: |
| **A → B** | 0.7826 | 0.9000 | 0.8372 | 0.8520 |
| B → A | 0.7037 | 0.9500 | 0.8085 | 0.8125 |
| Bidirectional maximum | 0.6897 | 1.0000 | 0.8163 | 0.8137 |

Reversing the pair changed the argmax label on 12 of 60 development pairs. The combined strategy found two more contradictions than A → B and added four false positives. A → B had the highest development F1 and was used for the final CPU evaluation.

## Held-out results

Three-class NLI accuracy was **0.8833**, with macro precision **0.9136**, macro recall **0.8833**, and macro F1 **0.8796**.

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Contradiction | 0.7407 | 1.0000 | 0.8511 | 20 |
| Neutral | 1.0000 | 0.6500 | 0.7879 | 20 |
| Entailment | 1.0000 | 1.0000 | 1.0000 | 20 |

In the binary comparison, contradiction is the positive class.

| Method | Accuracy | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniLM NLI | 0.8833 | 0.7407 | 1.0000 | 0.8511 | 7 | 0 |
| Rule baseline | 0.7500 | 1.0000 | 0.2500 | 0.4000 | 0 | 15 |

All seven NLI errors were neutral pairs flagged as contradictions. Six involved related requirements. For example, the model treated encryption and receipts for financial transactions as conflicting with a contradiction score of 0.990 (`NLI-663`). It also flagged registered customers' permission to purchase movies paired with unregistered visitors' permission to browse the catalog (`NLI-425`), and two compatible behaviors after a missed shot (`NLI-533`). An unrelated repair-facility and account-type pair scored 0.909 (`NLI-247`). The holdout had zero neutral–entailment confusions. [Per-pair predictions](../../research/results/contradiction_pair_predictions.json)

The rules found three opposite obligations, including direct negation and permission/prohibition wording, and two incompatible numeric bounds with matching clauses. They missed all four held-out conflicts in each of the role, action, and condition families, two of four numeric conflicts, and one of four obligation conflicts. The NLI model recovered those conflicts on this set. [Error counts and case IDs](../../research/results/contradiction_error_analysis.json)

The 60 held-out pairs are curated, partly share anchors, and have no independent label adjudication. Six of 20 source projects appear in both partitions. Entailment cases are paraphrases, so their perfect held-out score covers equivalence only. Pairwise NLI has no access to definitions, role hierarchies, other requirements, or actual operating conditions from a full specification.

The results JSON contains configuration, split checksums, metrics for all three development strategies, held-out metrics, and confusion counts. The local `contradiction_error_records.csv` contains both texts for each wrong decision, with the true label, predicted label, and contradiction score. The rule baseline's score is its binary match indicator. An NLI error can have a record in both evaluation tasks. [Results JSON](../../research/results/contradiction_detection_results.json) · [Three-class matrix](../../research/results/contradiction_nli_three_class_confusion_matrix.png) · [Binary matrix](../../research/results/contradiction_nli_binary_confusion_matrix.png) · [Binary comparison](../../research/results/contradiction_method_comparison.png)

## Reproduce

Use the Phase 2 processed corpus and the ML virtual environment with the PyTorch build for your device. [Data setup](../data/README.md) · [PyTorch setup](TRANSFORMER.md)

From the repository root with the ML environment active, run the experiment.

```sh
python -m pip install -r ml/experiments/requirements-transformer.txt
python -B -m ml.data.contradictions.build_curated_set
python -B -m ml.experiments.run_contradiction_evaluation
```

The first run downloads the pinned checkpoint under ignored `ml/artifacts/contradiction_detection/huggingface/`. Set `HF_HUB_OFFLINE=1` after the download to use the local cache. Set `OMP_NUM_THREADS=4` to match the reported CPU run.

For single-pair or document analysis, construct `ContradictionDetector(load_nli_model().score_pairs, directionality)` with the selected strategy from the results JSON. `analyze_contradiction(a, b)` returns the NLI result. `find_contradictions(requirements)` returns flagged pairs with zero-based document positions. It batches inference while scanning unordered pairs. A single direction uses the document order as the original pair order. The number of scored pairs grows quadratically with the document length.

With `pytest==9.2.0` installed, run the focused tests.

```sh
python -B -m pytest -p no:cacheprovider -q ml/experiments/tests/test_contradiction_detection.py
```

Set `SPECSENTRY_TEST_NLI=1` to include the three-example smoke test using the cached checkpoint with local files only.
