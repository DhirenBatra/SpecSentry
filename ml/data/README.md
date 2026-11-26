# PROMISE_exp requirement classification

SpecSentry uses PROMISE_exp for **requirement classification**. The binary task distinguishes Functional from Non-Functional requirements. The multiclass task uses the original 12 categories. PROMISE_exp provides requirement-type labels only. Ambiguity detection, semantic duplicate detection, and contradiction detection will each need a separate data and annotation strategy.

## Source and snapshot

PROMISE_exp was introduced by Lima et al. in *Software Engineering Repositories: Expanding the PROMISE Database* (SBES 2019). [Publication](https://doi.org/10.1145/3350768.3350776)

The downloader uses `1-exploratory-analysis/data/PROMISE_exp.csv` from `AleksandarMitrevski/se-requirements-classification` at commit `1f5dc4501a1956f21011083594d925bac49f198c`. It verifies SHA-256 `1b2cc33743588748ced7abd74a527e92f6f6d0dbe5342692cd2aa101e864f34f` before writing. A verified existing copy is reused unchanged. A different file at the raw path causes an error. [Commit-pinned CSV](https://raw.githubusercontent.com/AleksandarMitrevski/se-requirements-classification/1f5dc4501a1956f21011083594d925bac49f198c/1-exploratory-analysis/data/PROMISE_exp.csv)

The actual CSV is UTF-8 and comma-delimited with headers `ProjectID`, `RequirementText`, and `_class_`. It has **969 requirements and 47 distinct project IDs**. IDs 41 and 44 are absent from the observed 1–49 range. Inspection found zero missing fields and zero exact text duplicates. Raw text averages 17.91 whitespace-delimited words, with a median of 15 and a range of 4–86.

## Labels and counts

Task A maps `F` to **Functional** and the other 11 codes to **Non-Functional**. The raw binary counts are 444 and 525. After cleaning, they are 444 and 524. Task B uses `original_label` as the category code and `label_name` for display. The readable category names come from the [upstream dataset overview](https://github.com/AleksandarMitrevski/se-requirements-classification/blob/1f5dc4501a1956f21011083594d925bac49f198c/README.md#data-set-overview).

| Code | Category | Raw | Cleaned |
| --- | --- | ---: | ---: |
| A | Availability | 31 | 31 |
| F | Functional Requirement | 444 | 444 |
| FT | Fault Tolerance | 18 | 18 |
| L | Legal | 15 | 15 |
| LF | Look-and-feel | 49 | 49 |
| MN | Maintainability | 24 | 24 |
| O | Operability | 77 | 77 |
| PE | Performance | 67 | 66 |
| PO | Portability | 12 | 12 |
| SC | Scalability | 22 | 22 |
| SE | Security | 125 | 125 |
| US | Usability | 85 | 85 |
| | Total | 969 | 968 |

The binary task is close to balanced. The multiclass task is imbalanced. Functional Requirement accounts for 45.9% of the cleaned data, while Portability has 12 rows (1.2%) and Legal has 15 rows across five projects.

## Files and cleaning

- `raw/PROMISE_exp.csv` holds the unchanged source bytes. The pipeline checks the checksum, columns, record widths, and category set before it writes outputs.
- `processed/requirements.csv` holds the 968 cleaned rows. `train.csv`, `validation.csv`, and `test.csv` contain the same columns. The CSVs are materialized locally from the pinned source. Git tracks the pipeline and aggregate research outputs.
- `source_row` is the one-based CSV record number after the header. `project_id` and `requirement_text` preserve the original fields. `clean_text` trims the edges and collapses each run of whitespace to a single space. This changed 189 source text fields. Case, punctuation, and wording are preserved.
- The cleaner skips rows with a blank required field. This snapshot had none and needed no encoding repairs. Source rows 498 and 503 have identical normalized text, project ID 10, and code `PE`. The processed output keeps row 498 so the same example counts once. Both rows remain in the raw CSV.

## Split evaluation

Both approaches target 70% training, 15% validation, and 15% test with seed 42. Both retain all 12 categories in each partition.

| Approach | Train | Validation | Test | Train/validation shared projects | Train/test shared projects |
| --- | ---: | ---: | ---: | ---: | ---: |
| Row-stratified by category | 677 | 148 | 143 | 41 | 37 |
| Project-aware (selected) | 677 | 143 | 148 | 0 | 0 |

The row-stratified comparison shuffles each category and allocates records by largest-remainder rounding of the target percentages. It shares 35 projects between validation and test.

The project-aware method shuffles the project IDs 5,000 times. For each order, it picks whole-project validation and test sets closest to their row targets and puts the remaining projects in training. Candidates are ranked by training category coverage, then held-out category coverage, then by size and category distribution error. The selected split has 33 training projects, 8 validation projects, and 6 test projects. Its row shares are 69.94%, 14.77%, and 15.29%, with no project IDs or identical clean text shared between any partitions.

Project-aware splitting limits project leakage while retaining category coverage. Rare-class evaluation will still have small sample sizes, including one Portability requirement in test and one Maintainability requirement in validation. The test set represents six projects, so it provides limited evidence about performance on new projects.

## Reproduce and verify

Run from the repository root with Python 3.12. In PowerShell:

```powershell
py -3.12 -m venv ml/.venv
.\ml\.venv\Scripts\python.exe -m pip install -r ml/data/requirements.txt
.\ml\.venv\Scripts\python.exe -m ml.data.download_promise
.\ml\.venv\Scripts\python.exe -m ml.data.preprocess_promise
.\ml\.venv\Scripts\python.exe -m unittest discover -s ml/data/tests -v
```

On macOS or Linux, run `python3.12 -m venv ml/.venv` and use `ml/.venv/bin/python` for the remaining commands. Matplotlib renders the figures.

The [machine-readable statistics](../../research/results/dataset_statistics.json) record raw and processed counts and word lengths, cleaning decisions, and both split evaluations. The figures use the cleaned dataset:

- [Requirements per category](../../research/results/requirements_per_category.png)
- [Functional vs Non-Functional distribution](../../research/results/functional_vs_nonfunctional.png)
- [Requirement length distribution](../../research/results/requirement_length_distribution.png)
