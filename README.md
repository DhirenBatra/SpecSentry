# SpecSentry

SpecSentry is a final-year Computational Linguistics and NLP lab project for reviewing natural-language software requirements in a Software Requirements Specification (SRS). It classifies pasted requirements and flags ambiguous wording, potential duplicates, and potential contradictions for review.

## Current status

- Phase 1 — Complete. The initial React landing page verified the local FastAPI connection and linked to its API documentation.
- Phase 2 — Complete. The PROMISE_exp pipeline prepares and verifies the data for binary and multiclass requirement classification.
- Phase 3 — Complete. Four classical classification pipelines were selected on validation and evaluated on the held-out test projects.
- Phase 4 — Complete. Binary and 12-class DistilBERT checkpoints were evaluated against the Phase 3 baselines.
- Phase 5 — Complete. An interpretable rule-based ambiguity detector was evaluated on a separate curated set.
- Phase 6 — Complete. TF-IDF and SBERT duplicate detectors were evaluated using development-only cutoffs.
- Phase 7 — Complete. A frozen NLI contradiction detector was compared with conservative conflict rules.
- Phase 8 — Complete. FastAPI exposes the unified analysis engine with per-requirement evidence and a document quality score.
- Phase 9 — Complete. A separate endpoint provides optional, user-triggered assistance for selected ambiguity and contradiction findings.
- Phase 10 — Complete. The React workspace connects paste input and the demo to the analysis dashboard and optional assistance.

## Dataset

```text
PROMISE_exp → download → validation → cleaning → processed dataset
            → train / validation / test → Phase 3 model training
```

The pinned PROMISE_exp CSV contains 969 requirements from 47 project IDs. Conservative cleaning yields 968 requirements, and the split keeps each project in one partition. The [dataset documentation](ml/data/README.md) covers provenance, labels, split evaluation, and reproduction commands.

## Classical classification baselines

```text
                         ┌─ Logistic Regression
Requirement text → TF-IDF
                         └─ Linear SVM
```

Each classifier was evaluated for Functional vs Non-Functional and for the original 12 PROMISE_exp categories. The pipelines fit on 677 training rows, used 143 validation rows for selection by macro F1, and reported final results on 148 test rows. Bold marks the validation winner for each task.

| Task | Classifier | Validation macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Test weighted F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | **Logistic Regression** | 0.8094 | 0.8716 | 0.8750 | 0.8813 | 0.8713 | 0.8722 |
| Binary | Linear SVM | 0.7864 | 0.8581 | 0.8549 | 0.8583 | 0.8562 | 0.8584 |
| Multiclass | Logistic Regression | 0.6165 | 0.5946 | 0.3665 | 0.3469 | 0.3449 | 0.5786 |
| Multiclass | **Linear SVM** | 0.6514 | 0.7230 | 0.4803 | 0.4484 | 0.4550 | 0.7037 |

The test set has one Portability example, which both multiclass pipelines missed. See the [experiment guide](ml/experiments/README.md) for the rerun command, configurations, error analysis, and figures.

## DistilBERT classification

```text
PROMISE_exp → DistilBERT → Binary ───┐
                        → 12-class ─┴→ Held-out evaluation → Phase 3 comparison
```

Both models selected epoch 3 by validation macro F1. The test metrics use the same 148 rows as Phase 3.

| Task | Validation macro F1 | Test accuracy | Test macro precision | Test macro recall | Test macro F1 | Test weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Binary | 0.7588 | 0.6486 | 0.6895 | 0.6049 | 0.5795 | 0.6026 |
| Multiclass | 0.0759 | 0.4392 | 0.1196 | 0.0875 | 0.0585 | 0.2752 |

DistilBERT underperformed both classical baselines. Binary test macro F1 was 0.5795 versus 0.8713 for Logistic Regression. Multiclass test macro F1 was 0.0585 versus 0.4550 for Linear SVM. The multiclass model predicted Functional for 147 of 148 test rows, including the only Portability example. See the [DistilBERT experiment guide](ml/experiments/TRANSFORMER.md) for the rerun command, configuration, errors, and figures.

## Interpretable ambiguity detection

The detector returns matched spans, explanations, and a 0–10 heuristic severity score. It checks vague terms, weak quantifiers, weak obligations, undefined comparisons, unresolved opening references, and open-ended phrases. The evaluation uses a separately labeled, 100-case curated set with 50 ambiguous and 50 clear cases.

| Method | Accuracy | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full detector | 0.9100 | 0.9556 | 0.8600 | 0.9053 | 2 | 7 |
| Vague terms only | 0.6100 | 0.8667 | 0.2600 | 0.4000 | 2 | 37 |

The additional rules caught 30 more ambiguous cases while the false-positive count stayed at two. These scores describe the curated pairs. The [ambiguity experiment guide](ml/experiments/AMBIGUITY.md) documents the scoring, error cases, figures, and reproduction command.

## Semantic duplicate detection

The curated set has 120 PROMISE_exp-derived pairs with 60 paraphrases and 60 non-duplicates. Both methods use cosine similarity. TF-IDF fits only on development texts, while SBERT uses frozen `all-MiniLM-L6-v2` embeddings. Each cutoff maximizes duplicate-class F1 on 60 development pairs. The remaining 60 pairs give the following results.

| Method | Threshold | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + cosine | 0.1406 | 0.5333 | 0.5179 | 0.9667 | 0.6744 |
| SBERT + cosine | 0.5158 | 0.5833 | 0.5455 | 1.0000 | 0.7059 |

SBERT recovered one more paraphrase and rejected two more unrelated pairs. Both methods flagged all 25 hard negatives despite changes in roles, actions, negation, numeric limits, or intent. [Dataset design](ml/data/duplicates/README.md) · [Experiment, error analysis, and figures](ml/experiments/DUPLICATES.md)

## NLI contradiction detection

A frozen `cross-encoder/nli-MiniLM2-L6-H768` model and a rule baseline were evaluated on 120 curated PROMISE_exp-derived pairs. Forward NLI achieved the highest development binary F1. On the 60 held-out pairs, its three-class accuracy was 0.8833 and macro F1 was 0.8796.

| Binary method | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| MiniLM NLI | 0.8833 | 0.7407 | 1.0000 | 0.8511 |
| Rule baseline | 0.7500 | 1.0000 | 0.2500 | 0.4000 |

NLI found all 20 held-out contradictions and falsely flagged seven neutral pairs. The rules flagged five contradictions with zero false positives. [Dataset design](ml/data/contradictions/README.md) · [Experiment, error analysis, and figures](ml/experiments/CONTRADICTIONS.md)

## Unified analysis API

`POST /api/analyze` accepts 1–20 text requirements with unique IDs. It uses the saved Phase 3 classifiers, Phase 5 ambiguity rules, frozen SBERT cutoff, and forward NLI argmax. The response includes ambiguity spans, pair evidence, a 0–100 quality score, and its penalty breakdown.

The real CPU demo returned HTTP 200 for 11 requirements in 20.243 seconds, including first-request model loading. It reported one ambiguous requirement, two potential duplicate pairs, and 17 potential contradiction pairs. [API setup, request examples, and scoring formula](backend/ANALYSIS_API.md) · [Recorded smoke result](research/results/analysis_api_smoke_results.json)

## Optional AI assistance

`POST /api/ai/assist` can rewrite an ambiguous requirement or explain a potential contradiction using evidence already returned by the NLP pipeline. It uses the OpenAI Responses API with `gpt-4.1-mini` by default. Assistance runs only when a caller explicitly requests it. [Configuration and request examples](backend/AI_ASSISTANCE.md)

## Architecture

```text
React workspace ── GET /api/health ───────────→ FastAPI connection status
                ├─ POST /api/analyze ────────→ Local NLP → Findings and quality dashboard
                └─ explicit POST /api/ai/assist → OpenAI → Explanation and suggestion
```

The NLP pipeline detects ambiguity, potential duplicates, and potential contradictions, classifies requirements, and computes the quality score independently of the optional assistance provider.

## Final workspace

- Paste one requirement per line or load the Phase 8 demo. The editor assigns IDs, tracks the count, and checks the local backend connection.
- Review the quality score and penalty breakdown, finding counts, and predicted category distribution.
- Filter the requirement explorer by ambiguity status and inspect the matched phrases, severity, and rule explanations.
- Compare potential duplicate and contradiction pairs side by side. Expand the pair list for longer results and inspect the returned cosine and NLI scores.
- Request a rewrite or conflict explanation for a selected finding, then start a fresh analysis when ready.

## Stack and layout

- Frontend — React, Vite, Tailwind CSS 4 via the Vite plugin, and ESLint
- Backend — Python, FastAPI, Pydantic, and Uvicorn
- Tests — pytest, FastAPI TestClient, and HTTPX2
- Dataset pipeline — Python standard library and Matplotlib for report figures
- Classical NLP baselines — scikit-learn and joblib
- Transformer experiments — PyTorch and Hugging Face Transformers
- Semantic duplicate detection — Sentence Transformers and scikit-learn

```text
SpecSentry/
├── frontend/
│   └── src/                 # Requirements editor, analysis dashboard, and API service
├── backend/
│   ├── app/
│   │   ├── api/             # Health, analysis, and optional assistance routes
│   │   ├── core/            # Local configuration
│   │   ├── nlp/             # Standalone ambiguity, duplicate, and contradiction detectors
│   │   ├── schemas/         # Pydantic response models
│   │   ├── services/        # Analysis orchestration and model loading
│   │   └── main.py
│   ├── tests/
│   └── requirements.txt
├── ml/                      # Data, preprocessing, experiments, evaluation, artifacts
├── research/                # Literature and results
├── .env.example
├── .gitignore
└── README.md
```

## Run locally

Use Python 3.12 and Node.js 24 LTS. Start each service in a separate terminal from the repository root.

### Backend

Use the existing ML virtual environment or create one with Python 3.12. The backend needs the original Phase 3 `binary_logistic_regression.joblib` and `multiclass_linear_svm.joblib` files in `ml/artifacts/`. Use the PyTorch build for your device. On the first valid analysis, the pretrained loaders use their local caches or download the pinned checkpoints. [Analysis setup and artifacts](backend/ANALYSIS_API.md) · [PyTorch setup](ml/experiments/TRANSFORMER.md)

Windows PowerShell

```powershell
.\ml\.venv\Scripts\python.exe -m pip install -r backend/requirements-analysis.txt
Set-Location backend
..\ml\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

macOS or Linux

```bash
./ml/.venv/bin/python -m pip install -r backend/requirements-analysis.txt
cd backend
../ml/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- API base — `http://localhost:8000`
- Health — [http://localhost:8000/api/health](http://localhost:8000/api/health)
- Interactive docs — [http://localhost:8000/docs](http://localhost:8000/docs)

### Frontend

```sh
cd frontend
npm ci
npm run dev
```

In Windows PowerShell, use `npm.cmd` in place of `npm` if script execution is disabled. Open the workspace and look for **Backend connected** in the header. [Local frontend](http://localhost:5173)

To override the public API URL, set `VITE_API_BASE_URL` in a root `.env.local` file and restart Vite.

### Optional OpenAI assistance

Install `backend/requirements-ai.txt` in the backend's Python environment and set `OPENAI_API_KEY` in the backend process environment before starting Uvicorn. `SPECSENTRY_LLM_MODEL` overrides the default `gpt-4.1-mini`, and `SPECSENTRY_LLM_PROVIDER=disabled` turns assistance off. The health and analysis flows run without these settings. [Configuration and examples](backend/AI_ASSISTANCE.md)

### Demo workflow

1. Click **Load Demo Requirements** to load the 11-requirement sample. [Demo input](examples/analyze_demo.json)
2. Click **Analyze Requirements**. The loading state stays visible during analysis. The first CPU request can take tens of seconds while the models load.
3. Review the quality score and chart. Choose **Ambiguous** in the explorer to inspect evidence, then review potential duplicates and contradictions. Use **Show more** to inspect additional pairs.
4. With optional AI configured, click **Suggest Rewrite** or **Explain Conflict** on one finding. The suggestions appear beside the original text for review.
5. Click **Analyze New Requirements** to return to a blank editor.

### Known limitations

- The editor accepts 1–20 nonblank requirements, with up to 500 characters per requirement.
- Classification labels and pair scores are predictions. NLI can flag unrelated pairs, and SBERT can flag requirements that share wording while differing in meaning. Review the surrounding context before using the findings or the heuristic quality score.
- Optional AI assistance sends the selected text and evidence to the configured provider on request. Treat suggestions as drafts for review.

### Checks

- From `backend/`, run `python -m pytest` with the backend's Python environment active.
- From `frontend/`, run `npm run lint` and `npm run build`.
