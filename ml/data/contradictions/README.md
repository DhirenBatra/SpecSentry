# Curated NLI evaluation set

Each pair labels the relationship between two requirements about the same assumed specification. A contradiction means the two obligations, permissions, or constraints cannot both hold for the same actor, scope, condition, and state. Entailment pairs are designed as paraphrases under the same interpretation. Neutral pairs can coexist and neither requirement implies the other.

## Construction and provenance

The builder verifies the SHA-256 of the existing Phase 2 PROMISE_exp `clean_text` corpus. It uses 40 distinct source rows as anchors. The checked-in recipe specifies controlled rewrites to isolate obligations, clarify scopes, or express incompatible alternatives. For each anchor it contains one incompatible variant, one paraphrase, and one neutral variant. Nine neutral variants use unrelated donor rows from PROMISE_exp. The other 31 share the anchor's topic or context while requiring different behavior. The builder materializes the fixed recipe using standard library operations. Labels follow the controlled edits. [PROMISE_exp provenance](../README.md) · [Curation recipe](curation.json)

| Class | Total | Development | Held out |
| --- | ---: | ---: | ---: |
| Contradiction | 40 | 20 | 20 |
| Entailment | 40 | 20 | 20 |
| Neutral | 40 | 20 | 20 |
| **Total** | **120** | **60** | **60** |

Contradictions span five families with eight anchors each. Obligation pairs include explicit negation, permission versus prohibition, and required versus forbidden behavior. Role pairs grant access to actors explicitly excluded by the anchor. Numeric pairs impose incompatible limits. Action pairs require opposite outcomes. Condition pairs force incompatible outcomes under overlapping triggers. Related neutral edits often reuse actors, actions, or conditions to make lexical overlap insufficient.

The fixed split seed is 42. Each family contributes four anchor groups to development and four to evaluation. An anchor, its three pairs, and any donor all stay in one partition. The builder checks source row IDs and normalized full texts for overlap. Development contains four unrelated neutral pairs and evaluation contains five.

## Reproduce

From the repository root with the Phase 2 corpus in place, run the builder.

```sh
python -B -m ml.data.contradictions.build_curated_set
```

The builder writes `curated_nli_pairs.csv`, `curated_nli_pairs_development.csv`, and `curated_nli_pairs_evaluation.csv` beside the recipe. The local CSVs contain both texts, labels, case and group IDs, families, and the supporting PROMISE_exp row and project IDs.

This is a curated evaluation set. Source projects can recur across partitions. The groups reuse an anchor for several cases and labels lack independent adjudication. Entailment is represented by equivalence, with no systematic coverage of one-way implications. The balanced class mix and controlled edits limit what the results establish about contradictions in complete specifications.
