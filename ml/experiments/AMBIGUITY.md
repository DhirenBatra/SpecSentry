# Interpretable ambiguity detection

SpecSentry flags language that leaves a test-relevant part of a standalone requirement open to multiple reasonable interpretations. That can be the expected behavior, an acceptance threshold, a triggering condition, a required actor, or the scope of a list. A flag is a cue for review.

## Detector and score

`detect_ambiguity(text)` returns `ambiguous`, `score`, `severity`, and `issues`. Each issue includes the original matching text, category, an explanation, weight, and zero-based start and end character offsets. Blank strings raise `ValueError`. Other input types raise `TypeError`.

| Category | Examples | Weight |
| --- | --- | ---: |
| Vague or subjective terms | quickly, intuitive, user-friendly, sufficient | 2 |
| Weak quantifiers | some, several, usually, approximately | 1 |
| Weak obligations | should, may, if possible, as needed | 1 |
| Undefined comparatives or superlatives | faster, lower, minimal, optimal | 2 |
| Unclear opening references | it, this, that, they, these | 2 |
| Open-ended phrases | as soon as possible, as required, etc. | 3 |

Case-insensitive phrase rules match whole tokens. Quoted literals are skipped, and the longest overlapping phrase wins when matches start together. Local checks allow an adjacent timing limit, a numeric comparison or explicit tolerance, recognized permission and date uses of “may,” a named “as required” citation, and introductory expletive “it.” The reference rule only checks opening pronouns with no preceding antecedent.

The score is the sum of weights for distinct category and normalized trigger pairs, capped at 10. Case, whitespace, and hyphen variation do not add weight for repeats. Any surviving issue sets `ambiguous=true`. Scores 0–2 are low, 3–5 medium, and 6–10 high. A score of zero has no issues. “The system should respond quickly” scores 3 from the weak obligation and vague timing. The score measures rule signals, not model confidence or probability.

## Evaluation and ablation

The curated set has 50 ambiguous and 50 clear requirements across 50 pairs. Precision, recall, and F1 use ambiguous as the positive class. Both modes use the same tokenization, guards, and overlapping-span resolution. The vague-only mode retains only vague-term matches before scoring and prediction. [Curation protocol and provenance](../data/ambiguity/README.md)

| Method | Accuracy | Precision | Recall | F1 | False positives | False negatives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full detector | 0.9100 | 0.9556 | 0.8600 | 0.9053 | 2 | 7 |
| Vague terms only | 0.6100 | 0.8667 | 0.2600 | 0.4000 | 2 | 37 |

The full confusion matrix is `[[48, 2], [7, 43]]` in clear-then-ambiguous order, with actual labels in rows. Adding the other five categories recovered 30 ambiguous requirements with zero additional false positives. Recall increased by 0.6000 and F1 by 0.5053.

Both false positives were definitions outside the narrow context checks. “Sufficient” had a six-user capacity criterion and “appropriate” named the exact tab. The seven misses were four unlisted phrases, two cases with a missing unit or interval, and one later pronoun with two plausible antecedents. The tracker preserves the case IDs and rule explanations. Full-text error rows stay local with the generated evaluation CSV.

The rules cover a limited lexicon and local contexts. Missing semantics, later or cross-document references, and specialized definitions can escape them. The curated pairs were designed around particular ambiguity mechanisms. The measured performance applies to those cases.

## Reproduce

From the root, use the existing ML environment with the pinned Phase 3 requirements and the Phase 2 cleaned corpus:

```powershell
.\ml\.venv\Scripts\python.exe -B -m ml.experiments.run_ambiguity_evaluation
```

The command materializes the curated set and writes:

- [Metrics and both confusion matrices](../../research/results/ambiguity_detection_results.json)
- [False-positive and false-negative analysis](../../research/results/ambiguity_error_analysis.json)
- [Full detector confusion matrix figure](../../research/results/ambiguity_confusion_matrix.png)
- [Ablation comparison figure](../../research/results/ambiguity_ablation.png)
- Local error records at `research/results/ambiguity_error_records.csv`

From the backend directory, run `python -B -m pytest` using the backend environment.
