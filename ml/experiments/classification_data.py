"""Read the frozen Phase 2 splits for the classification baselines."""

import csv
import hashlib
import io

from ml.data.preprocess_promise import LABEL_NAMES, PROCESSED_DIR, Requirement, binary_label, split_overlap

PHASE2_COMMIT = "9df2a9373b5481fec522f0b9a70781f4f0331741"

# Fingerprints of the authoritative local Phase 2 outputs.
SPLIT_INFO = {
    "train": {"rows": 677, "sha256": "353b6fc002713a08fbe503c592bf7d513767102f224901a2fa2fbc6ed2d82c10"},
    "validation": {"rows": 143, "sha256": "ce1a5f6c48900dd73813c66eb9a241742b4422ae1fff70ca52eab2ea65aeb2d1"},
    "test": {"rows": 148, "sha256": "363996f48f77606b8fa8dfb55eab98a2c21105586646a253f32e93fe4be4c039"},
}

TASK_TARGETS = {"binary": "binary_label", "multiclass": "original_label"}
TASK_LABELS = {
    "binary": {"Functional": "Functional", "Non-Functional": "Non-Functional"},
    "multiclass": LABEL_NAMES,
}


def read_split(name: str) -> list[Requirement]:
    path = PROCESSED_DIR / f"{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Restore the existing Phase 2 split at {path}.")

    data = path.read_bytes()
    expected = SPLIT_INFO[name]
    if hashlib.sha256(data).hexdigest() != expected["sha256"]:
        raise ValueError(f"The Phase 2 split checksum differs at {path}.")

    reader = csv.DictReader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
    if reader.fieldnames != list(Requirement._fields):
        raise ValueError(f"Unexpected processed columns in {path}.")

    rows = []
    for record in reader:
        source_row = int(record.pop("source_row"))
        rows.append(Requirement(source_row=source_row, **record))

    if len(rows) != expected["rows"] or len({row.source_row for row in rows}) != len(rows):
        raise ValueError(f"Unexpected row count or repeated source row in {path}.")
    if {row.original_label for row in rows} != LABEL_NAMES.keys():
        raise ValueError(f"Unexpected requirement categories in {path}.")
    for row in rows:
        if not (row.requirement_text.strip() and row.clean_text.strip() and row.project_id.strip()):
            raise ValueError(f"Blank text or project ID in {path} at source row {row.source_row}.")
        if row.label_name != LABEL_NAMES[row.original_label] or row.binary_label != binary_label(row.original_label):
            raise ValueError(f"Inconsistent labels in {path} at source row {row.source_row}.")
    return rows


def ensure_disjoint(splits: dict[str, list[Requirement]]) -> None:
    for field in ("source_row", "project_id", "clean_text"):
        if any(split_overlap(splits, field).values()):
            raise ValueError(f"Phase 2 partitions share {field} values.")


def target_labels(rows: list[Requirement], task: str) -> list[str]:
    return [getattr(row, TASK_TARGETS[task]) for row in rows]
