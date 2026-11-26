"""Materialize the fixed pair edits and a group-stratified 50/50 split."""

import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from ml.data.ambiguity.build_curated_set import replace_exact
from ml.data.preprocess_promise import PROCESSED_DIR

DATA_DIR = Path(__file__).resolve().parent
RECIPE_PATH = DATA_DIR / "curation.json"
SOURCE_PATH = PROCESSED_DIR / "requirements.csv"
DATASET_PATH = DATA_DIR / "curated_duplicate_pairs.csv"
DEVELOPMENT_PATH = DATA_DIR / "curated_duplicate_pairs_development.csv"
EVALUATION_PATH = DATA_DIR / "curated_duplicate_pairs_evaluation.csv"
FIELDS = (
    "pair_id", "group_id", "partition", "true_label", "subtype",
    "source_row_a", "source_row_b", "project_id_a", "project_id_b",
    "requirement_a", "requirement_b",
)


def _normalize(text):
    return " ".join(text.casefold().split())


def _sentence(text):
    text = text.strip()
    return text if text.endswith((".", "?", "!")) else text + "."


def check_disjoint(development, evaluation):
    """Reject shared source rows or normalized full text across partitions."""
    for fields, normalizer in (
        (("source_row_a", "source_row_b"), int),
        (("requirement_a", "requirement_b"), _normalize),
    ):
        development_values = {normalizer(pair[field]) for pair in development for field in fields}
        evaluation_values = {normalizer(pair[field]) for pair in evaluation for field in fields}
        if development_values & evaluation_values:
            raise ValueError(f"Split leakage on {fields}.")


def build_pairs():
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    source_bytes = SOURCE_PATH.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != recipe["source_sha256"]:
        raise ValueError("The Phase 2 cleaned corpus differs from the pinned file.")
    source_csv = io.StringIO(source_bytes.decode("utf-8"), newline="")
    originals = {int(row["source_row"]): row for row in csv.DictReader(source_csv)}

    rng = random.Random(recipe["split_seed"])
    pairs = []
    used_rows = set()
    for subtype, groups in recipe["groups"].items():
        if len(groups) != 10:
            raise ValueError(f"Expected ten anchor requirements for the {subtype} stratum.")
        development_rows = set(rng.sample([group["row"] for group in groups], k=5))
        for group in groups:
            anchor_row = group["row"]
            donor_row = group.get("donor_row")
            group_rows = {anchor_row} if donor_row is None else {anchor_row, donor_row}
            if group_rows & used_rows or anchor_row == donor_row:
                raise ValueError(f"Reused source row in the group for row {anchor_row}.")
            used_rows.update(group_rows)

            anchor = _sentence(originals[anchor_row]["clean_text"])
            anchor = _sentence(replace_exact(anchor, group.get("base", []), anchor_row))
            paraphrase = _sentence(replace_exact(anchor, group["paraphrase"], anchor_row))
            if donor_row is None:
                negative = _sentence(replace_exact(anchor, group["negative"], anchor_row))
            else:
                negative = _sentence(originals[donor_row]["clean_text"])
            if len({_normalize(anchor), _normalize(paraphrase), _normalize(negative)}) != 3:
                raise ValueError(f"Expected three distinct texts for anchor {anchor_row}.")

            group_id = f"DUP-{anchor_row:03d}"
            partition = "development" if anchor_row in development_rows else "evaluation"
            for label, tag, right_row, right in (
                ("duplicate", "paraphrase", anchor_row, paraphrase),
                ("non_duplicate", subtype, donor_row or anchor_row, negative),
            ):
                pairs.append({
                    "pair_id": f"{group_id}-{label}",
                    "group_id": group_id,
                    "partition": partition,
                    "true_label": label,
                    "subtype": tag,
                    "source_row_a": anchor_row,
                    "source_row_b": right_row,
                    "project_id_a": originals[anchor_row]["project_id"],
                    "project_id_b": originals[right_row]["project_id"],
                    "requirement_a": anchor,
                    "requirement_b": right,
                })

    development = [pair for pair in pairs if pair["partition"] == "development"]
    evaluation = [pair for pair in pairs if pair["partition"] == "evaluation"]
    check_disjoint(development, evaluation)
    return pairs, development, evaluation


def main():
    pairs, development, evaluation = build_pairs()
    for path, rows in (
        (DATASET_PATH, pairs), (DEVELOPMENT_PATH, development), (EVALUATION_PATH, evaluation),
    ):
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path.name}: {len(rows)} pairs · {dict(Counter(pair['true_label'] for pair in rows))}")


if __name__ == "__main__":
    main()
