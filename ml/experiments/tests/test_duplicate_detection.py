"""Check the duplicate detector without requiring a model download."""

import os
from collections import Counter

import numpy as np
import pytest

from backend.app.nlp.duplicates import (
    DuplicateDetector,
    fit_tfidf,
    load_sbert_model,
    sbert_similarity,
)
from ml.data.duplicates.build_curated_set import build_pairs, check_disjoint
from ml.experiments.run_duplicate_evaluation import select_threshold, threshold_labels

ANCHOR = "Administrators shall delete inactive accounts."
HARD_NEGATIVE = "Administrators shall view inactive accounts."
UNRELATED = "Players may decorate doughnuts."


@pytest.fixture
def lexical_detector():
    scorer = fit_tfidf([ANCHOR, HARD_NEGATIVE, UNRELATED])
    return DuplicateDetector(scorer, 0.5)


def test_identical_requirement_returns_a_structured_result(lexical_detector):
    result = lexical_detector.analyze_duplicate(ANCHOR, ANCHOR)
    assert result == {
        "is_duplicate": True,
        "similarity": pytest.approx(1.0),
        "threshold": 0.5,
        "method": "tfidf",
        "requirement_a": ANCHOR,
        "requirement_b": ANCHOR,
    }


def test_lexical_overlap_keeps_a_hard_negative_harder_than_an_unrelated_pair(lexical_detector):
    easy = lexical_detector.analyze_duplicate(ANCHOR, UNRELATED)
    hard = lexical_detector.analyze_duplicate(ANCHOR, HARD_NEGATIVE)
    assert easy["similarity"] == 0.0
    assert not easy["is_duplicate"]
    assert 0.5 < hard["similarity"] < 1.0
    assert hard["is_duplicate"]


def test_tfidf_ignores_vocabulary_absent_from_fit_texts(lexical_detector):
    unseen = "Motor temperature limit"
    result = lexical_detector.analyze_duplicate(unseen, unseen)
    assert result["similarity"] == 0.0
    assert not result["is_duplicate"]


class RecordingEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **options):
        self.calls.append(list(texts))
        assert options["batch_size"] > 1
        assert options["normalize_embeddings"]
        vectors = {
            "Users must be able to save a copy.": [1.0, 0.0],
            "Users can keep a separate copy.": [0.8, 0.6],
            "Tabs display team names.": [0.0, 1.0],
        }
        return np.array([vectors[text] for text in texts])


def test_document_search_encodes_each_text_once_and_includes_the_cutoff():
    anchor = "Users must be able to save a copy."
    paraphrase = "Users can keep a separate copy."
    other = "Tabs display team names."
    model = RecordingEncoder()
    detector = DuplicateDetector(sbert_similarity(model), 0.8)
    matches = detector.find_duplicates([anchor, paraphrase, other, anchor])

    assert model.calls == [[anchor, paraphrase, other]]
    assert [(row["index_a"], row["index_b"]) for row in matches] == [(0, 1), (0, 3), (1, 3)]
    assert [row["similarity"] for row in matches] == pytest.approx([0.8, 1.0, 0.8])
    assert all(row["method"] == "sbert" and row["is_duplicate"] for row in matches)
    assert detector.analyze_duplicate(anchor, paraphrase)["is_duplicate"]


def test_lexical_document_search_preserves_zero_based_positions(lexical_detector):
    matches = lexical_detector.find_duplicates([UNRELATED, ANCHOR, HARD_NEGATIVE])
    match, = matches
    assert (match["index_a"], match["index_b"]) == (1, 2)
    assert match["requirement_b"] == HARD_NEGATIVE


@pytest.mark.parametrize("invalid", ["", " \t ", None, 10])
def test_invalid_requirement_is_rejected(lexical_detector, invalid):
    with pytest.raises(ValueError, match="nonempty"):
        lexical_detector.analyze_duplicate(ANCHOR, invalid)
    with pytest.raises(ValueError, match="nonempty"):
        lexical_detector.find_duplicates([ANCHOR, invalid])


def test_empty_document_and_out_of_range_threshold(lexical_detector):
    assert lexical_detector.find_duplicates([]) == []
    assert lexical_detector.find_duplicates([ANCHOR]) == []
    with pytest.raises(TypeError, match="collection"):
        lexical_detector.find_duplicates(ANCHOR)
    with pytest.raises(ValueError, match="threshold"):
        DuplicateDetector(lexical_detector.scorer, 1.1)


def test_threshold_selection_breaks_development_f1_ties_toward_higher_cutoff():
    truth = ["duplicate", "non_duplicate", "non_duplicate", "duplicate"]
    threshold, f1 = select_threshold(truth, [0.9, 0.8, 0.5, 0.3])
    assert threshold == 0.9
    assert f1 == pytest.approx(2 / 3)
    assert threshold_labels([0.8999, 0.9, 1.0], threshold) == ["non_duplicate", "duplicate", "duplicate"]


def test_curated_split_balances_classes_and_keeps_related_texts_together():
    pairs, development, evaluation = build_pairs()
    assert len(pairs) == len({pair["pair_id"] for pair in pairs}) == 120
    assert len(pairs) == len(development) + len(evaluation)
    check_disjoint(development, evaluation)
    for partition in (development, evaluation):
        assert Counter(pair["true_label"] for pair in partition) == {"duplicate": 30, "non_duplicate": 30}
        assert Counter(pair["subtype"] for pair in partition) == {
            "paraphrase": 30, "unrelated": 5, "actor": 5, "action": 5, "negation": 5, "numeric": 5, "intent": 5,
        }


@pytest.mark.skipif(os.getenv("SPECSENTRY_TEST_SBERT") != "1", reason="Requires the cached MiniLM model")
def test_cached_sbert_scores_a_real_paraphrase_above_an_unrelated_requirement(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    scorer = sbert_similarity(load_sbert_model())
    original = "The system shall email a receipt to the buyer after each purchase."
    variant = "After every purchase, the buyer must receive a receipt by email."
    different = "Administrators shall archive logs after six months."
    identical, paraphrase, unrelated = scorer.score_pairs(
        [(original, original), (original, variant), (original, different)]
    )
    assert identical == pytest.approx(1.0)
    assert paraphrase > unrelated
