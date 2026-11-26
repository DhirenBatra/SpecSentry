"""Validate PROMISE_exp and generate classification data and research outputs."""

import csv
import io
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from itertools import accumulate, combinations
from pathlib import Path
from typing import NamedTuple

from .download_promise import (
    DATA_DIR,
    RAW_PATH,
    SOURCE_COMMIT,
    SOURCE_PATH,
    SOURCE_REPOSITORY,
    SOURCE_SHA256,
    SOURCE_URL,
    verify_source,
)

PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR.parents[1] / "research" / "results"
RAW_COLUMNS = ("ProjectID", "RequirementText", "_class_")
LABEL_REFERENCE = f"{SOURCE_REPOSITORY}/blob/{SOURCE_COMMIT}/README.md#data-set-overview"
LABEL_NAMES = {
    "A": "Availability",
    "F": "Functional Requirement",
    "FT": "Fault Tolerance",
    "L": "Legal",
    "LF": "Look-and-feel",
    "MN": "Maintainability",
    "O": "Operability",
    "PE": "Performance",
    "PO": "Portability",
    "SC": "Scalability",
    "SE": "Security",
    "US": "Usability",
}

SPLIT_NAMES = ("train", "validation", "test")
SPLIT_PERCENTAGES = (70, 15, 15)
SEED = 42
PROJECT_TRIALS = 5_000


class Requirement(NamedTuple):
    source_row: int
    project_id: str
    requirement_text: str
    clean_text: str
    original_label: str
    label_name: str
    binary_label: str


def read_raw() -> list[list[str]]:
    if not RAW_PATH.is_file():
        raise FileNotFoundError("Run python -m ml.data.download_promise first.")
    data = RAW_PATH.read_bytes()
    verify_source(data)

    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
    header = next(reader, None)
    if header != list(RAW_COLUMNS):
        raise ValueError(f"Unexpected CSV columns {header!r}. Expected {RAW_COLUMNS!r}.")
    rows = list(reader)
    if not rows or any(len(row) != len(RAW_COLUMNS) for row in rows):
        raise ValueError("The CSV must contain records with exactly three fields.")

    found_labels = {label for _, _, label in rows if label.strip()}
    if found_labels != LABEL_NAMES.keys():
        raise ValueError(f"Unexpected PROMISE_exp label set {sorted(found_labels)!r}.")
    return rows


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def binary_label(code: str) -> str:
    return "Functional" if code == "F" else "Non-Functional"


def clean_requirements(raw_rows: list[list[str]]) -> tuple[list[Requirement], dict]:
    requirements = []
    missing_rows = []
    duplicate_rows = []
    first_occurrence = {}
    whitespace_changes = 0
    for source_row, (project_id, text, code) in enumerate(raw_rows, start=1):
        clean_text = normalize_whitespace(text)
        whitespace_changes += text != clean_text
        if not (project_id.strip() and clean_text and code.strip()):
            missing_rows.append(source_row)
            continue

        key = (project_id, code, clean_text)
        if key in first_occurrence:
            duplicate_rows.append({
                "kept_source_row": first_occurrence[key],
                "removed_source_row": source_row,
            })
            continue

        first_occurrence[key] = source_row
        requirements.append(Requirement(
            source_row, project_id, text, clean_text, code, LABEL_NAMES[code], binary_label(code)
        ))

    return requirements, {
        "rows_with_whitespace_changes": whitespace_changes,
        "dropped_missing_source_rows": missing_rows,
        "dropped_same_project_and_label_duplicates": duplicate_rows,
    }


def largest_remainder_counts(total: int) -> list[int]:
    """Round the requested percentages so the counts sum to the class size."""
    proportions = [divmod(total * rate, 100) for rate in SPLIT_PERCENTAGES]
    counts = [whole for whole, _ in proportions]
    remainder_order = sorted(range(len(counts)), key=lambda index: -proportions[index][1])
    for index in remainder_order[:total - sum(counts)]:
        counts[index] += 1
    return counts


def stratified_split(rows: list[Requirement]) -> dict[str, list[Requirement]]:
    """Ordinary row-level stratification, computed for comparison only."""
    by_label = defaultdict(list)
    for row in rows:
        by_label[row.original_label].append(row)

    rng = random.Random(SEED)
    splits = {name: [] for name in SPLIT_NAMES}
    for code in sorted(by_label):
        category = by_label[code]
        rng.shuffle(category)
        start = 0
        for name, count in zip(SPLIT_NAMES, largest_remainder_counts(len(category)), strict=True):
            splits[name].extend(category[start:start + count])
            start += count
    return splits


def project_aware_split(rows: list[Requirement]) -> dict[str, list[Requirement]]:
    """Find a label-balanced partition while keeping each project together."""
    by_project = defaultdict(list)
    for row in rows:
        by_project[row.project_id].append(row)
    projects = sorted(by_project)
    if len(projects) < len(SPLIT_NAMES):
        raise ValueError("At least three projects are required for a project-aware split.")

    label_totals = Counter(row.original_label for row in rows)
    project_counts = {
        project: Counter(row.original_label for row in group)
        for project, group in by_project.items()
    }
    ratios = [percentage / 100 for percentage in SPLIT_PERCENTAGES]
    rng = random.Random(SEED)
    best_score = None
    best_groups = None

    for _ in range(PROJECT_TRIALS):
        rng.shuffle(projects)
        totals = list(accumulate((len(by_project[project]) for project in projects), initial=0))
        validation_end = min(
            range(1, len(projects) - 1),
            key=lambda index: abs(totals[index] - len(rows) * ratios[1]),
        )
        test_end = min(
            range(validation_end + 1, len(projects)),
            key=lambda index: abs(totals[index] - totals[validation_end] - len(rows) * ratios[2]),
        )
        groups = (projects[test_end:], projects[:validation_end], projects[validation_end:test_end])

        counts = []
        for group in groups:
            labels = Counter()
            for project in group:
                labels.update(project_counts[project])
            counts.append(labels)

        missing = [len(label_totals.keys() - group.keys()) for group in counts]
        size_error = sum(
            abs(sum(group.values()) / len(rows) - ratio)
            for group, ratio in zip(counts, ratios, strict=True)
        )
        label_error = sum(
            abs(group[code] / total - ratio)
            for group, ratio in zip(counts, ratios, strict=True)
            for code, total in label_totals.items()
        ) / len(label_totals)
        score = (missing[0], sum(missing[1:]), 2 * size_error + label_error)
        if best_score is None or score < best_score:
            best_score, best_groups = score, groups

    if best_groups is None or best_score[0]:
        raise ValueError("No project split could place every category in training.")

    return {
        name: sorted(
            (row for project in group for row in by_project[project]),
            key=lambda row: row.source_row,
        )
        for name, group in zip(SPLIT_NAMES, best_groups, strict=True)
    }


def split_overlap(splits: dict[str, list[Requirement]], field: str) -> dict[str, int]:
    values = {name: {getattr(row, field) for row in rows} for name, rows in splits.items()}
    return {
        f"{left}_vs_{right}": len(values[left] & values[right])
        for left, right in combinations(splits, 2)
    }


def validate_project_split(rows: list[Requirement], splits: dict[str, list[Requirement]]) -> None:
    expected = Counter(row.source_row for row in rows)
    assigned = Counter(row.source_row for group in splits.values() for row in group)
    if assigned != expected:
        raise ValueError("The split must assign every processed requirement exactly once.")
    if any(split_overlap(splits, "project_id").values()):
        raise ValueError("The split shares projects between partitions.")
    if any(split_overlap(splits, "clean_text").values()):
        raise ValueError("The split shares identical clean text between partitions.")


def word_statistics(lengths: list[int]) -> dict:
    return {
        "mean": statistics.mean(lengths),
        "median": statistics.median(lengths),
        "minimum": min(lengths),
        "maximum": max(lengths),
    }


def duplicate_statistics(texts: list[str]) -> dict[str, int]:
    repeated = [count for count in Counter(texts).values() if count > 1]
    return {"groups": len(repeated), "rows_beyond_first": sum(count - 1 for count in repeated)}


def summarize_split(splits: dict[str, list[Requirement]], total: int) -> dict:
    partitions = {}
    for name, rows in splits.items():
        counts = Counter(row.original_label for row in rows)
        partitions[name] = {
            "requirements": len(rows),
            "percentage": round(100 * len(rows) / total, 2),
            "projects": len({row.project_id for row in rows}),
            "class_counts": {code: counts[code] for code in LABEL_NAMES},
            "binary_counts": dict(Counter(row.binary_label for row in rows)),
        }
    return {
        "partitions": partitions,
        "project_overlap": split_overlap(splits, "project_id"),
        "clean_text_overlap": split_overlap(splits, "clean_text"),
    }


def build_statistics(raw_rows, rows, cleaning, ordinary_splits, chosen_splits) -> dict:
    raw_texts = [text for _, text, _ in raw_rows if text.strip()]
    raw_labels = [code for _, _, code in raw_rows if code.strip()]
    return {
        "source": {
            "dataset": "PROMISE_exp",
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "path": SOURCE_PATH,
            "download_url": SOURCE_URL,
            "sha256": SOURCE_SHA256,
            "encoding": "UTF-8",
            "delimiter": ",",
            "label_reference": LABEL_REFERENCE,
        },
        "label_names": LABEL_NAMES,
        "raw": {
            "requirements": len(raw_rows),
            "columns": RAW_COLUMNS,
            "projects": len({project for project, _, _ in raw_rows if project.strip()}),
            "missing_values": {
                column: sum(not row[index].strip() for row in raw_rows)
                for index, column in enumerate(RAW_COLUMNS)
            },
            "exact_duplicate_text": duplicate_statistics(raw_texts),
            "whitespace_normalized_duplicate_text": duplicate_statistics(
                [normalize_whitespace(text) for text in raw_texts]
            ),
            "class_counts": dict(Counter(raw_labels)),
            "binary_counts": dict(Counter(map(binary_label, raw_labels))),
            "word_lengths": word_statistics([len(text.split()) for text in raw_texts]),
        },
        "cleaning": cleaning,
        "processed": {
            "requirements": len(rows),
            "columns": Requirement._fields,
            "projects": len({row.project_id for row in rows}),
            "class_counts": dict(Counter(row.original_label for row in rows)),
            "binary_counts": dict(Counter(row.binary_label for row in rows)),
            "word_lengths": word_statistics([len(row.clean_text.split()) for row in rows]),
            "projects_per_class": {
                code: len({row.project_id for row in rows if row.original_label == code})
                for code in LABEL_NAMES
            },
        },
        "splits": {
            "seed": SEED,
            "target_percentages": dict(zip(SPLIT_NAMES, SPLIT_PERCENTAGES, strict=True)),
            "selected_method": "project_aware",
            "project_aware_trials": PROJECT_TRIALS,
            "row_stratified": summarize_split(ordinary_splits, len(rows)),
            "project_aware": summarize_split(chosen_splits, len(rows)),
        },
    }


def write_csv(path: Path, rows: list[Requirement]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(Requirement._fields)
        writer.writerows(rows)


def main() -> None:
    try:
        raw_rows = read_raw()
        rows, cleaning = clean_requirements(raw_rows)
        ordinary_splits = stratified_split(rows)
        chosen_splits = project_aware_split(rows)
        validate_project_split(rows, chosen_splits)
        results = build_statistics(raw_rows, rows, cleaning, ordinary_splits, chosen_splits)

        from .plot_promise import write_figures

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        write_csv(PROCESSED_DIR / "requirements.csv", rows)
        for name, group in chosen_splits.items():
            write_csv(PROCESSED_DIR / f"{name}.csv", group)
        (RESULTS_DIR / "dataset_statistics.json").write_text(
            json.dumps(results, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
        write_figures(
            RESULTS_DIR,
            LABEL_NAMES,
            results["processed"]["class_counts"],
            results["processed"]["binary_counts"],
            [len(row.clean_text.split()) for row in rows],
        )
    except (OSError, ValueError, csv.Error) as error:
        sys.exit(f"PROMISE_exp preprocessing stopped. {error}")

    sizes = ", ".join(f"{name}={len(group)}" for name, group in chosen_splits.items())
    print(f"Verified {len(raw_rows)} raw records. Wrote {len(rows)} clean records ({sizes}).")
    print(f"Statistics and figures written to {RESULTS_DIR}.")


if __name__ == "__main__":
    main()
