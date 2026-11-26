# SpecSentry analysis API

```text
Ordered requirements
  ├─ Phase 3 classification and Phase 5 ambiguity analysis
  └─ Pair analysis
       ├─ SBERT cosine similarity
       └─ Forward NLI
               ↓
        Quality score and evidence
               ↓
         POST /api/analyze response
```

## Start the API

Use Python 3.12 and the existing ML virtual environment. Install the backend with the pinned ML dependencies and keep the original `binary_logistic_regression.joblib` and `multiclass_linear_svm.joblib` files in `ml/artifacts/`. The loader verifies them against the recorded Phase 3 checksums before deserialization. Restore the saved Phase 3 files if a fresh checkout lacks them. [Phase 3 artifact guide](../ml/experiments/README.md) · [PyTorch setup](../ml/experiments/TRANSFORMER.md)

From the repository root in PowerShell:

```powershell
.\ml\.venv\Scripts\python.exe -m pip install -r backend/requirements-analysis.txt
Set-Location backend
$env:OMP_NUM_THREADS = "4"
..\ml\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On macOS or Linux, activate the ML environment, install `backend/requirements-analysis.txt` from the repository root, and run `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000` from `backend/`.

The first valid analysis request loads the saved classifiers and the pinned SBERT and NLI checkpoints. A process-wide cache and load lock let concurrent requests share the same models. Inference is serialized within that process to limit CPU and memory contention. The pretrained loaders cache public downloads under ignored `ml/artifacts/` directories. Set `HF_HUB_OFFLINE=1` when using populated caches offline. [Swagger UI](http://127.0.0.1:8000/docs)

## Production methods

- **Classification.** Phase 3 TF-IDF + Logistic Regression for Functional vs Non-Functional and TF-IDF + Linear SVM for the 12 PROMISE_exp categories. Phase 3 had higher held-out macro F1 for both tasks. Binary scores were 0.8713 for Logistic Regression and 0.5795 for DistilBERT. Multiclass scores were 0.4550 for Linear SVM and 0.0585 for DistilBERT. Classification uses the Phase 2 whitespace normalization and calls each saved pipeline once per document. [Classical results](../research/results/classification_baseline_results.json) · [DistilBERT results](../research/results/transformer_classification_results.json)
- **Ambiguity.** The Phase 5 rules run on the original text and return the frozen severity, score, evidence spans, and explanations. Offsets refer to the returned text. [Ambiguity guide](../ml/experiments/AMBIGUITY.md)
- **Potential duplicates.** `sentence-transformers/all-MiniLM-L6-v2` encodes each distinct text once per document in batches. The detector reuses those embeddings for cosine similarity and flags pairs at `similarity >= 0.5158489942550659`. [Frozen Phase 6 setting](../research/results/duplicate_detection_results.json)
- **Potential contradictions.** `cross-encoder/nli-MiniLM2-L6-H768` evaluates every unordered pair in batches of 16, with the earlier input as A and later input as B. It reports a match when contradiction is the native three-class argmax. It evaluates shared and distinct topics alike, independently of duplicate decisions. [Frozen Phase 7 setting](../research/results/contradiction_detection_results.json)

## Request and response

`POST /api/analyze` accepts JSON in this form.

```json
{
  "requirements": [
    {"id": "R1", "text": "The system shall store audit logs for at least 90 days."},
    {"id": "R2", "text": "The system shall delete audit logs within 30 days."}
  ]
}
```

Send 1–20 requirements. IDs must be nonempty and unique after trimming whitespace, with at most 64 characters. Each text must contain a non-whitespace character and may contain up to 500 characters.

| Response field | Contents |
| --- | --- |
| `summary` | Requirement and finding counts, quality score, and unique requirements affected by each pairwise check |
| `requirements` | Original text, ID, multiclass code and name, binary category, and ambiguity score, severity, spans, and explanations |
| `duplicate_pairs` | Both requirement IDs, cosine similarity, threshold, and method |
| `contradiction_pairs` | Both requirement IDs, contradiction, neutral, and entailment scores, and the winning label, model, and direction |
| `score_breakdown` | The three document penalties used in the quality score |

Pair IDs refer to entries in `requirements`. Invalid requests return HTTP 422 before loading models. A missing artifact or model-loading failure returns HTTP 503 with a setup hint. Inference failures return HTTP 500 with diagnostic details in the server log.

## Quality score

Let `n` be the total requirements, `a` the number flagged as ambiguous, `d` the number of distinct requirement IDs in potential duplicate pairs, and `c` the number of distinct IDs in potential contradiction pairs. Calculate the point deductions and final score as follows.

```text
ambiguity_penalty     = round(20 × a / n, 2)
duplicate_penalty     = round(30 × d / n, 2)
contradiction_penalty = round(50 × c / n, 2)
quality_score        = round(clamp(100 - sum(the three penalties), 0, 100), 2)
```

Counting affected requirements once per issue type keeps a cluster of overlapping pairs from repeatedly charging for the same requirement. Contradictions carry the largest weight at equal coverage. Classification labels contribute zero points to the deductions. The result is a deterministic heuristic based on the flags.

## Demo

From the repository root, send the demo to the running server. On macOS or Linux, use `curl` in place of `curl.exe`.

```powershell
curl.exe -sS -H "Content-Type: application/json" --data-binary "@examples/analyze_demo.json" http://127.0.0.1:8000/api/analyze
```

The first request to a fresh local Uvicorn process returned HTTP 200 in **20.243 seconds** on CPU with four threads. This single smoke test includes loading both pretrained models from cache and analyzing all 55 pairs. The 11 demo requirements yielded a **41.82** quality score, one ambiguous requirement, two potential duplicates, and 17 potential contradictions. The intended receipt paraphrase and log-retention conflict were both flagged. [Demo input](../examples/analyze_demo.json) · [Smoke result](../research/results/analysis_api_smoke_results.json)

Several contradiction flags connected independent requirements. For example, NLI flagged email receipts and encrypted database storage. SBERT also paired the conflicting retention limits because they share similar wording. These flags contributed to the reported quality score.

## Limits

The 20-requirement limit allows at most 190 NLI evaluations per request. NLI reads up to 256 tokens per pair, so long or heavily tokenized text can be truncated. It compares the submitted text pair without the surrounding specification and its decisions can depend on request order. A pair can appear in both finding lists. Model flags and their scores need interpretation in the specification's context.
