"""Build three labeled variants per PROMISE_exp anchor and split by anchor."""

import csv
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path

from ml.data.duplicates.build_curated_set import SOURCE_PATH, check_disjoint

DATA_DIR = Path(__file__).resolve().parent
RECIPE_PATH = DATA_DIR / "curation.json"
DATASET_PATH = DATA_DIR / "curated_nli_pairs.csv"
DEVELOPMENT_PATH = DATA_DIR / "curated_nli_pairs_development.csv"
EVALUATION_PATH = DATA_DIR / "curated_nli_pairs_evaluation.csv"
FIELDS = (
    "pair_id", "group_id", "partition", "true_label", "family", "design",
    "source_row_a", "source_row_b", "project_id_a", "project_id_b",
    "requirement_a", "requirement_b",
)


def _sentence(text):
    text = text.strip()
    if not text:
        raise ValueError("Curated requirement text is empty.")
    return text if text.endswith((".", "!", "?")) else text + "."


def build_pairs():
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    source = SOURCE_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != recipe["source_sha256"]:
        raise ValueError("The Phase 2 cleaned corpus differs from the pinned file.")
    originals = {
        int(row["source_row"]): row
        for row in csv.DictReader(io.StringIO(source.decode("utf-8"), newline=""))
    }
    rng = random.Random(recipe["split_seed"])
    pairs = []
    used_rows = set()
    for family, entries in recipe["groups"].items():
        if len(entries) != 8:
            raise ValueError(f"Expected eight anchors in the {family} stratum.")
        development_rows = set(rng.sample([entry["row"] for entry in entries], 4))
        for entry in entries:
            row = entry["row"]
            donor = entry["neutral"] if isinstance(entry["neutral"], int) else None
            source_rows = [row] if donor is None else [row, donor]
            if row == donor or used_rows.intersection(source_rows):
                raise ValueError(f"Reused source row in group for row {row}.")
            used_rows.update(source_rows)
            first = _sentence(entry.get("anchor", originals[row]["clean_text"]))
            variants = {
                "contradiction": _sentence(entry["contradiction"]),
                "entailment": _sentence(entry["entailment"]),
                "neutral": _sentence(originals[donor]["clean_text"] if donor is not None else entry["neutral"]),
            }
            if len({" ".join(text.casefold().split()) for text in (first, *variants.values())}) != 4:
                raise ValueError(f"Expected four distinct texts for anchor {row}.")

            group_id = f"NLI-{row:03d}"
            partition = "development" if row in development_rows else "evaluation"
            designs = {
                "contradiction": "incompatible",
                "entailment": "paraphrase",
                "neutral": "unrelated" if donor is not None else "related",
            }
            for label, second in variants.items():
                right_row = donor if label == "neutral" and donor is not None else row
                pairs.append({
                    "pair_id": f"{group_id}-{label}",
                    "group_id": group_id,
                    "partition": partition,
                    "true_label": label,
                    "family": family,
                    "design": designs[label],
                    "source_row_a": row,
                    "source_row_b": right_row,
                    "project_id_a": originals[row]["project_id"],
                    "project_id_b": originals[right_row]["project_id"],
                    "requirement_a": first,
                    "requirement_b": second,
                })
    development = [pair for pair in pairs if pair["partition"] == "development"]
    evaluation = [pair for pair in pairs if pair["partition"] == "evaluation"]
    check_disjoint(development, evaluation)
    return pairs, development, evaluation


def main():
    pairs, development, evaluation = build_pairs()
    for path, rows in ((DATASET_PATH, pairs), (DEVELOPMENT_PATH, development), (EVALUATION_PATH, evaluation)):
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{path.name}: {len(rows)} pairs {dict(Counter(row['true_label'] for row in rows))}")


if __name__ == "__main__":
    main()
