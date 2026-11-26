"""Build a curated ambiguity set from fixed edits, without running a detector."""

import csv
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

from ml.data.preprocess_promise import PROCESSED_DIR

DATA_DIR = Path(__file__).resolve().parent
RECIPE_PATH = DATA_DIR / "curation.json"
DATASET_PATH = DATA_DIR / "curated_ambiguity_evaluation.csv"
SOURCE_PATH = PROCESSED_DIR / "requirements.csv"
FIELDS = ("id", "pair_id", "source_row", "project_id", "focus", "challenge", "label", "requirement")


def replace_exact(text, edits, source_row):
    for before, after in edits:
        if not before or text.count(before) != 1:
            raise ValueError(f"Expected one occurrence of {before!r} in source row {source_row}.")
        text = text.replace(before, after, 1)
    return text


def build_examples():
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    source_bytes = SOURCE_PATH.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != recipe["source_sha256"]:
        raise ValueError("The Phase 2 source corpus differs from the pinned file.")
    reader = csv.DictReader(io.StringIO(source_bytes.decode("utf-8"), newline=""))
    originals = {int(row["source_row"]): row for row in reader}

    examples = []
    for index, pair in enumerate(recipe["pairs"], start=1):
        source = originals[pair["row"]]
        text = source["clean_text"]
        if not text.endswith((".", "?", "!")):
            text += "."
        clear = replace_exact(text, pair.get("clear", ()), pair["row"])
        ambiguous = replace_exact(clear, pair["ambiguous"], pair["row"])
        pair_id = f"AMB-{index:03d}"
        for label, variant in (("clear", clear), ("ambiguous", ambiguous)):
            examples.append({
                "id": f"{pair_id}-{label}", "pair_id": pair_id, "source_row": pair["row"],
                "project_id": source["project_id"], "focus": pair["focus"],
                "challenge": pair.get("challenge", ""), "label": label, "requirement": variant,
            })

    if len(recipe["pairs"]) != 50 or len({case["source_row"] for case in examples}) != 50:
        raise ValueError("Expected 50 pairs from distinct source requirements.")
    if len({case["requirement"] for case in examples}) != len(examples):
        raise ValueError("Curated requirement text must be unique across cases.")
    return recipe, examples


def write_dataset(examples):
    with DATASET_PATH.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(examples)


def main():
    recipe, examples = build_examples()
    write_dataset(examples)
    print(f"{recipe['dataset']}: {dict(Counter(case['label'] for case in examples))}")
    print(f"Saved {len(examples)} cases to {DATASET_PATH}")


if __name__ == "__main__":
    main()
