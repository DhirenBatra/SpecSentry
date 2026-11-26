# SpecSentry

SpecSentry is a final-year Computational Linguistics and NLP lab project for reviewing natural-language software requirements in a Software Requirements Specification (SRS). Ambiguous wording, repeated requirements, and conflicts can be hard to catch by hand, especially as an SRS grows. The planned system will flag these issues and point reviewers to the relevant requirements.

## Current status

- Phase 1 — Complete. The React landing page checks the local FastAPI service and links to its API documentation.
- Phase 2 — Complete. The PROMISE_exp pipeline prepares and verifies the data for binary and multiclass requirement classification. Model training begins in Phase 3.

## Dataset

```text
PROMISE_exp → download → validation → cleaning → processed dataset
            → train / validation / test → Phase 3 model training
```

The pinned PROMISE_exp CSV contains 969 requirements from 47 project IDs. Conservative cleaning yields 968 requirements, and the split keeps each project in one partition. The [dataset documentation](ml/data/README.md) covers provenance, labels, split evaluation, and reproduction commands.

## Planned architecture

```text
Phase 1    React landing page → GET /api/health → FastAPI
Planned    SRS input → Requirement extraction → NLP analysis → Quality report → Optional LLM assistance
```

The NLP pipeline will detect ambiguous requirements, identify semantic duplicates and potential contradictions, classify requirement types, and generate an interpretable quality report. It will run independently of the LLM. Optional LLM assistance will explain flagged issues and suggest clearer wording.

## Stack and layout

- Frontend — React, Vite, Tailwind CSS 4 via the Vite plugin, and ESLint
- Backend — Python, FastAPI, Pydantic, and Uvicorn
- Tests — pytest, FastAPI TestClient, and HTTPX2
- Dataset pipeline — Python standard library and Matplotlib for report figures

```text
SpecSentry/
├── frontend/
│   └── src/                 # Landing page, components, and API service
├── backend/
│   ├── app/
│   │   ├── api/             # HTTP routes
│   │   ├── core/            # Local configuration
│   │   ├── nlp/             # Reserved for NLP analysis
│   │   ├── schemas/         # Pydantic response models
│   │   ├── services/        # Reserved for services such as SRS ingestion
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

Windows PowerShell

```powershell
cd backend
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

macOS or Linux

```bash
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
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

In Windows PowerShell, use `npm.cmd` in place of `npm` if script execution is disabled. Open [http://localhost:5173](http://localhost:5173) and look for **Connected** under Backend Status.

The default setup works without environment variables. To override the API URL, copy `.env.example` to `.env.local` at the repository root, set `VITE_API_BASE_URL`, and restart Vite. Vite includes this value in the browser bundle. Keep credentials out of `VITE_` variables.

### Checks

- From `backend/`, run `python -m pytest` with the virtual environment active or use its Python executable from the setup commands.
- From `frontend/`, run `npm run lint` and `npm run build`.

## Next phases

1. Train and evaluate binary and multiclass requirement classifiers on the Phase 2 splits.
2. Add SRS input and requirement extraction. Prepare separate datasets and evaluation plans for ambiguity, semantic duplicate, and contradiction detection.
3. Add a requirement explorer and a quality report that links findings back to the SRS.
4. Add optional LLM explanations and rewrite suggestions for findings from the NLP pipeline.
