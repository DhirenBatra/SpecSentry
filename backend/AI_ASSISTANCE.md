# Optional AI assistance

```text
Existing NLP detection → Select a finding → Explicit POST /api/ai/assist
                                              ↓
                                   Explanation and suggestion
```

The LLM does not determine whether a requirement is ambiguous, duplicate, contradictory, or its classification. Those decisions come from SpecSentry's existing NLP pipeline.

## Configure the backend

Install the optional SDK in the Python environment that runs FastAPI. Run this from the repository root with that environment active.

```sh
python -m pip install -r backend/requirements-ai.txt
```

Set the variables in the server process environment. The example environment file shows the settings with a blank key. [Environment template](../.env.example)

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Required to request OpenAI assistance. Keep the key in the backend environment. |
| `SPECSENTRY_LLM_PROVIDER` | Defaults to `openai`. Set `disabled` to turn off assistance. |
| `SPECSENTRY_LLM_MODEL` | Defaults to `gpt-4.1-mini`. A different OpenAI model must support Responses API structured output. |

The SDK initializes on the first valid assistance request with configuration. The local health and analysis routes work independently of these settings. [OpenAI quickstart](https://developers.openai.com/api/docs/quickstart) · [Structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)

## POST /api/ai/assist

Send a JSON object with one of the two actions below. Use the original text and evidence from the analysis response. Each requirement in the request can contain up to 500 characters. These examples illustrate the request format and the corresponding response fields.

### Rewrite ambiguity

Pass the selected requirement's text and up to 12 entries from its `ambiguity.issues`. The offsets must match the original text. The provider receives the text and the selected matched phrases, categories, and start offsets.

```json
{
  "action": "rewrite_ambiguity",
  "requirement": "The interface should load quickly.",
  "issues": [
    {
      "text": "quickly",
      "category": "vague_term",
      "explanation": "The requirement does not specify a measurable response-time threshold.",
      "start": 26,
      "end": 33,
      "weight": 2
    }
  ]
}
```

```json
{
  "explanation": "Quickly leaves the acceptable load time undefined.",
  "suggested_rewrite": "The interface should load within [define maximum load time]."
}
```

### Explain a potential contradiction

Pass both original texts in pair order and the matching entry from `contradiction_pairs` as `evidence`. The provider receives the texts, the existing label, and its three NLI scores. The suggestion focuses on the conditions and decisions that an engineer should clarify.

```json
{
  "action": "explain_contradiction",
  "requirement_a": "The service shall retain audit logs for at least 90 days.",
  "requirement_b": "The service shall delete audit logs within 30 days.",
  "evidence": {
    "requirement_a_id": "R1",
    "requirement_b_id": "R2",
    "is_contradiction": true,
    "label": "contradiction",
    "contradiction_score": 0.93,
    "neutral_score": 0.06,
    "entailment_score": 0.01,
    "model": "cross-encoder/nli-MiniLM2-L6-H768",
    "directionality": "forward"
  }
}
```

```json
{
  "explanation": "The retention and deletion windows may conflict if they cover the same audit logs.",
  "suggested_resolution": "Clarify the log types and retention scope. Specify [define the rule for overlapping cases]."
}
```

Invalid input returns HTTP 422. Disabled or misconfigured assistance returns 503. A provider failure or an invalid suggestion returns 502, and a provider timeout returns 504. Swagger exposes both request shapes and response schemas. [Local Swagger UI](http://127.0.0.1:8000/docs)

## Cost and limits

Only an explicit assist request sends the selected excerpts to OpenAI. The SDK uses a 15-second timeout, zero automatic retries, and a 320-output-token limit. Requests use strict JSON schema and set `store=false` on the Responses API call. Each backend process keeps up to 32 successful responses in memory and reuses them for identical provider inputs.

The API checks the shape and consistency of supplied evidence. The caller supplies the finding from the analysis response. Prompts ask for placeholders where information is missing, and validation rejects new numeric literals and recognized quantities. The guard covers common number and unit forms. An engineer still needs to review the wording for intent and unsupported business assumptions before adopting a suggestion.
