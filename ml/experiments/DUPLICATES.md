# Semantic duplicate requirement detection

The experiment compares unigram TF-IDF and frozen all-MiniLM-L6-v2 embeddings using cosine similarity. The two methods use the same 60 development and 60 held-out pairs. [Dataset construction and duplicate definition](../data/duplicates/README.md)

## Methods

- TF-IDF fits the vocabulary and inverse document frequencies on distinct development texts only. Word unigrams include one-character tokens such as single-digit constraints. It uses lowercase matching, retains stopwords, and L2-normalizes vectors. A text with no vocabulary matches produces a zero vector with similarity zero.
- SBERT encodes distinct texts in batches of 32 using the released MiniLM weights at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The model outputs normalized vectors and uses a 256-token context. The loader selects CUDA when available and CPU otherwise. [Model card and pinned revision](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41)
- Both methods use dot products of normalized rows for cosine similarity. Each method selects the cutoff with the highest positive-class F1 among its observed development scores, choosing the higher cutoff on ties. The decision is `similarity >= threshold`. The runner freezes both cutoffs before reading the held-out pair file.

## Held-out results

TF-IDF fit on 90 distinct development texts. MiniLM ran on CPU with PyTorch 2.15.0 and Sentence Transformers 6.3.0. Precision, recall, and F1 use duplicate as the positive class.

| Method | Threshold | Dev F1 | Accuracy | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + cosine | 0.1406 | 0.6824 | 0.5333 | 0.5179 | 0.9667 | 0.6744 | 27 | 1 |
| SBERT + cosine | 0.5158 | 0.7059 | 0.5833 | 0.5455 | 1.0000 | 0.7059 | 25 | 0 |
| Absolute metric difference | | | 0.0500 | 0.0276 | 0.0333 | 0.0315 | | |

SBERT made three more correct decisions. It recovered the MLS search paraphrase that TF-IDF missed and correctly rejected two unrelated pairs that TF-IDF flagged. The missed paraphrase used different words for the same search operation and had TF-IDF cosine 0.105. One of the unrelated pairs shared the phrase “The system shall allow” between MLS search and doughnut decoration. [Per-pair decisions and scores](../../research/results/duplicate_pair_predictions.json)

Both methods flagged all 25 hard negatives, five in each subtype. Changing the authorized role to a student advisor (`DUP-126`), changing record to display (`DUP-323`), adding negation (`DUP-505`), and changing a five-second limit to eight seconds (`DUP-154`) all produced false positives. The five- and eight-second limits had cosine 0.979 for TF-IDF and 0.984 for SBERT. Both methods also flagged requiring a printed installation guide versus installing without one (`DUP-053`). Every hard negative received a higher TF-IDF similarity than any held-out paraphrase. SBERT pushed paraphrase scores into the same range as many hard negatives. [Error counts and IDs](../../research/results/duplicate_error_analysis.json)

Fourteen of the 21 source projects contribute text to both partitions. The balanced design, five held-out negatives per subtype, shared project templates, and lack of independent label review limit what the measured gain says about new SRS documents. High cosine can indicate shared topic or wording even when satisfying one requirement would leave the other unmet.

The runner saves metrics, both cutoffs at full precision, and confusion counts. The local `research/results/duplicate_error_records.csv` contains both texts, the label, the prediction, the cosine score, and the method for each false positive and false negative. [Results JSON](../../research/results/duplicate_detection_results.json) · [TF-IDF matrix](../../research/results/duplicate_tfidf_confusion_matrix.png) · [SBERT matrix](../../research/results/duplicate_sbert_confusion_matrix.png) · [Metric comparison](../../research/results/duplicate_method_comparison.png) · [Similarity distribution](../../research/results/duplicate_similarity_distribution.png)

## Reproduce

Use the ML virtual environment with Python 3.12, the Phase 2 processed corpus, and the PyTorch build for your device. [Data setup](../data/README.md) · [PyTorch setup](TRANSFORMER.md)

Run these commands from the repository root with the ML environment active.

```sh
python -m pip install -r ml/experiments/requirements-duplicates.txt
python -B -m ml.data.duplicates.build_curated_set
python -B -m ml.experiments.run_duplicate_evaluation
```

The first run downloads the pinned public encoder to the ignored `ml/artifacts/duplicate_detection/huggingface/` cache. After a successful run, set `HF_HUB_OFFLINE=1` to rerun with the cached files. The structured results record data checksums, settings, device, and library versions.

For downstream use, construct `DuplicateDetector(sbert_similarity(load_sbert_model()), threshold)` from `backend.app.nlp.duplicates` with the full-precision threshold in the results JSON. For the baseline, pass the distinct development texts to `fit_tfidf` and use its selected cutoff. `analyze_duplicate(a, b)` scores one pair. `find_duplicates(requirements)` encodes each distinct string once and returns matching pairs with zero-based positions and their texts. It checks all unordered pairs and has quadratic comparison cost.

Install `pytest==9.2.0` in the ML environment to run the focused suite.

```sh
python -B -m pytest -p no:cacheprovider -q ml/experiments/tests/test_duplicate_detection.py
```

Set `SPECSENTRY_TEST_SBERT=1` to also run the real paraphrase smoke test after populating the cache. That test loads the cached model offline.
