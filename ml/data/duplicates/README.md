# Curated requirement-pair evaluation set

Two requirements count as duplicates when they specify the same obligation under the same actor, action, object, polarity, scope, and constraints. A build satisfying either wording should also satisfy the other under the same interpretation of the requirement.

## Provenance and construction

The recipe uses the `clean_text` field from the Phase 2 PROMISE_exp corpus. The builder verifies that file's SHA-256 against the value in [curation.json](curation.json). Each group starts with a distinct source row. Exact edits can clarify punctuation or isolate a single obligation from adjacent sentences. The positive variant changes phrasing or units while preserving the obligation. For example, a 60-second refresh interval becomes an update once per minute. The negative variant changes a meaningful detail or uses an unrelated source row. Plain Python applies the fixed edits. Labels follow the pair construction. [PROMISE_exp provenance](../README.md)

| Pair type | All pairs | Development | Held out |
| --- | ---: | ---: | ---: |
| Paraphrase | 60 | 30 | 30 |
| Unrelated | 10 | 5 | 5 |
| Changed actor | 10 | 5 | 5 |
| Changed action | 10 | 5 | 5 |
| Negation | 10 | 5 | 5 |
| Changed numeric constraint | 10 | 5 | 5 |
| Changed intent, trigger, or scope | 10 | 5 | 5 |
| **Total** | **120** | **60** | **60** |

Each anchor gives one paraphrase and one negative pair. There are 60 distinct anchors and 10 additional donor rows for unrelated pairs. The builder uses seed 42 to assign five anchors from each negative subtype to development and five to final evaluation. Keeping the two pairs and any donor together preserves the class and subtype balance and prevents the same anchor from appearing on both sides. The builder checks that neither source row IDs nor full texts after case and whitespace normalization cross partitions.

## Local files

With the Phase 2 corpus in place, run this from the repository root.

```sh
python -B -m ml.data.duplicates.build_curated_set
```

The builder writes `curated_duplicate_pairs.csv`, `curated_duplicate_pairs_development.csv`, and `curated_duplicate_pairs_evaluation.csv` here. Each row includes the texts, pair and group IDs, partition, design label, subtype, and PROMISE_exp row and project IDs for both texts. The assembled text CSVs are local. The tracked recipe reproduces them from the pinned corpus.

This curated diagnostic set is small and deliberately balanced. Projects and specification templates can recur across partitions. The labels have no independent human adjudication. Pair difficulty and the proportion of duplicate pairs in a real specification can differ from this design.
