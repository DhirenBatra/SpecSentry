import csv
import unittest
from collections import Counter
from itertools import combinations
from operator import itemgetter

from ml.data.preprocess_promise import (
    PROCESSED_DIR,
    clean_requirements,
    normalize_whitespace,
    project_aware_split,
    read_raw,
)

EXPECTED_LABELS = {
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
EXPECTED_COLUMNS = [
    "source_row",
    "project_id",
    "requirement_text",
    "clean_text",
    "original_label",
    "label_name",
    "binary_label",
]


def read_processed(filename):
    with (PROCESSED_DIR / filename).open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return reader.fieldnames, list(reader)


class PreprocessPromiseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = read_raw()
        cls.cleaned, cls.cleaning = clean_requirements(cls.raw)
        cls.columns, cls.processed = read_processed("requirements.csv")

    def test_whitespace_normalization_preserves_case_and_punctuation(self):
        self.assertEqual(
            normalize_whitespace("\t  The system SHALL keep\r\n  version 2.1! "),
            "The system SHALL keep version 2.1!",
        )

    def test_category_names_match_upstream(self):
        self.assertEqual(
            {row["original_label"]: row["label_name"] for row in self.processed},
            EXPECTED_LABELS,
        )

    def test_binary_labels_match_source_categories(self):
        self.assertEqual(
            Counter(row["binary_label"] for row in self.processed),
            {"Functional": 444, "Non-Functional": 524},
        )
        for row in self.processed:
            expected = "Functional" if row["original_label"] == "F" else "Non-Functional"
            self.assertEqual(row["binary_label"], expected)

    def test_csv_schema_and_original_text_preservation(self):
        self.assertEqual(self.columns, EXPECTED_COLUMNS)
        self.assertEqual(len(self.processed), 968)
        for row in self.processed:
            project, text, label = self.raw[int(row["source_row"]) - 1]
            self.assertEqual(
                (row["project_id"], row["requirement_text"], row["original_label"]),
                (project, text, label),
            )
            self.assertEqual(row["clean_text"], normalize_whitespace(text))

    def test_requirement_text_is_populated(self):
        self.assertTrue(self.processed)
        for row in self.processed:
            self.assertTrue(row["requirement_text"].strip())
            self.assertTrue(row["clean_text"].strip())

    def test_only_the_same_project_whitespace_duplicate_is_removed(self):
        self.assertEqual(
            self.cleaning["dropped_same_project_and_label_duplicates"],
            [{"kept_source_row": 498, "removed_source_row": 503}],
        )
        self.assertEqual(self.cleaning["dropped_missing_source_rows"], [])
        kept, removed = self.raw[497], self.raw[502]
        self.assertNotEqual(kept[1], removed[1])
        self.assertEqual(normalize_whitespace(kept[1]), normalize_whitespace(removed[1]))
        self.assertEqual((kept[0], kept[2]), (removed[0], removed[2]))
        self.assertCountEqual(
            [int(row["source_row"]) for row in self.processed],
            set(range(1, len(self.raw) + 1)) - {503},
        )

    def test_saved_splits_partition_rows_projects_and_text(self):
        partitions = {}
        for name in ("train", "validation", "test"):
            columns, partitions[name] = read_processed(f"{name}.csv")
            self.assertEqual(columns, EXPECTED_COLUMNS)
            self.assertEqual({row["original_label"] for row in partitions[name]}, set(EXPECTED_LABELS))

        for left, right in combinations(partitions.values(), 2):
            for field in ("project_id", "clean_text"):
                self.assertFalse({row[field] for row in left} & {row[field] for row in right})

        assigned = [row for partition in partitions.values() for row in partition]
        by_source_row = itemgetter("source_row")
        self.assertEqual(sorted(assigned, key=by_source_row), sorted(self.processed, key=by_source_row))

        regenerated = project_aware_split(self.cleaned)
        for name, rows in partitions.items():
            self.assertEqual(
                [int(row["source_row"]) for row in rows],
                [row.source_row for row in regenerated[name]],
            )


if __name__ == "__main__":
    unittest.main()
