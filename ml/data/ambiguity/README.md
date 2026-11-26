# Curated ambiguity evaluation set

This diagnostic set contains 100 requirements in 50 minimal pairs derived from 50 distinct PROMISE_exp source rows across 19 projects. Each pair has one clear and one ambiguous variant. The original PROMISE_exp class labels describe requirement types. The ambiguity labels here come from the specified edits.

## Construction and provenance

1. Start from the existing Phase 2 `clean_text` field. The builder checks the SHA-256 of the entire cleaned corpus before selecting the specified source rows.
2. Apply the optional exact clear-side edits from [curation.json](curation.json). These may clarify units, add a bound, preserve a quoted literal, or make an actor or reference explicit.
3. Apply the listed ambiguity edit to that clear variant. The change introduces an unspecified bound, obligation, actor, quality criterion, or scope. Labels follow this pair design. The builder uses deterministic string operations with fixed inputs.

Each row records a case ID, pair ID, original source row and project ID, intended ambiguity focus, optional challenge tag, label, and requirement text. Sixteen pairs target harder contexts such as cue words in clear requirements, unlisted wording, missing values, and multiple plausible later antecedents. An explicit cited specification ID counts as a defined reference for the designated clear case. [Phase 2 source details](../README.md)

The builder reads the existing corpus and leaves its files and splits intact. The recipe is tracked in Git. The generated text CSV stays local because it includes adapted source language.

From the repository root:

```powershell
.\ml\.venv\Scripts\python.exe -B -m ml.data.ambiguity.build_curated_set
```

This is a curated evaluation set for the lab. Paired cases share content, the category coverage was chosen deliberately, and the design labels have no independent human adjudication. The balanced set cannot estimate ambiguity prevalence or detector accuracy on new specifications.
