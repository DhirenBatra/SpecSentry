"""Verify contradiction semantics and inference wiring with one optional model smoke test."""

import os
from collections import Counter
from types import SimpleNamespace

import pytest

from backend.app.nlp.contradiction_rules import explicit_conflict
from backend.app.nlp.contradictions import (
    BATCH_SIZE,
    MODEL_NAME,
    ContradictionDetector,
    load_nli_model,
    verify_label_mapping,
)
from ml.data.contradictions.build_curated_set import build_pairs
from ml.experiments.run_contradiction_evaluation import select_directionality

ALERT = "The service must send every alert by email."
NEGATIVE = "The service must never send an alert by email."
PARAPHRASE = "The service is required to email all alerts."
UNRELATED = "The application shall open its settings panel in two seconds."
SCORES = {
    "contradiction": {"contradiction": 0.9, "neutral": 0.08, "entailment": 0.02},
    "entailment": {"contradiction": 0.03, "neutral": 0.07, "entailment": 0.9},
    "neutral": {"contradiction": 0.1, "neutral": 0.85, "entailment": 0.05},
}


@pytest.mark.parametrize("second,expected", [
    (NEGATIVE, "contradiction"), (PARAPHRASE, "entailment"), (UNRELATED, "neutral"),
])
def test_pair_output_uses_the_largest_model_score(second, expected):
    detector = ContradictionDetector(lambda pairs: [SCORES[expected]] * len(pairs), "forward")
    result = detector.analyze_contradiction(ALERT, second)
    assert result == {
        "is_contradiction": expected == "contradiction",
        "label": expected,
        **{f"{label}_score": score for label, score in SCORES[expected].items()},
        "model": MODEL_NAME,
        "directionality": "forward",
        "requirement_a": ALERT,
        "requirement_b": second,
    }


@pytest.mark.parametrize("positive,negative", [
    ("The system shall send alerts.", "The system shall not send alerts."),
    ("An operator may restart the service.", "An operator may not restart the service."),
    ("The system must export invoices.", "The system is forbidden to export invoices."),
])
def test_explicit_opposite_obligations(positive, negative):
    assert explicit_conflict(positive, negative) == "opposite_obligations"
    assert explicit_conflict(negative, positive) == "opposite_obligations"


@pytest.mark.parametrize("first_bound,second_bound,expected", [
    ("at least 90 days", "at most 30 days", "disjoint_numeric_bounds"),
    ("at least 90 days", "less than 90 days", "disjoint_numeric_bounds"),
    ("at least 90 days", "at most 90 days", None),
    ("at most 30 days", "at most 90 days", None),
    ("at least 0.5 seconds", "at most 0.25 seconds", "disjoint_numeric_bounds"),
    ("exactly 30 days", "exactly 31 days", "disjoint_numeric_bounds"),
    ("at least 90 days", "at most 3 months", None),
])
def test_numeric_intervals_keep_unit_and_boundary_semantics(first_bound, second_bound, expected):
    first = f"The service shall retain each record for {first_bound}."
    second = f"The service must retain each record for {second_bound}."
    assert explicit_conflict(first, second) == expected


def test_rules_leave_different_actors_and_conditional_ranges_unflagged():
    assert explicit_conflict("Users may export data.", "Guests may not export data.") is None
    assert explicit_conflict(
        "The service shall alert an operator if at least 5 users are online.",
        "The service shall alert an operator if at most 2 users are online.",
    ) is None


def test_directionality_uses_the_winning_full_vector_and_original_text_order():
    calls = []

    def scorer(pairs):
        calls.extend(pairs)
        return [SCORES["neutral"] if first == ALERT else SCORES["contradiction"] for first, _ in pairs]

    reverse = ContradictionDetector(scorer, "reverse").analyze_contradiction(ALERT, NEGATIVE)
    assert calls == [(NEGATIVE, ALERT)]
    calls.clear()
    combined = ContradictionDetector(scorer, "max_contradiction").analyze_contradiction(ALERT, NEGATIVE)
    assert calls == [(ALERT, NEGATIVE), (NEGATIVE, ALERT)]
    assert combined["label"] == reverse["label"] == "contradiction"
    assert combined["neutral_score"] == SCORES["contradiction"]["neutral"]
    assert combined["requirement_a"] == ALERT and combined["requirement_b"] == NEGATIVE


def test_select_directionality_prefers_one_pass_without_material_gain():
    assert select_directionality({"forward": .60, "reverse": .60, "max_contradiction": .64}) == "forward"
    assert select_directionality({"forward": .60, "reverse": .70, "max_contradiction": .74}) == "reverse"
    assert select_directionality({"forward": .60, "reverse": .70, "max_contradiction": .76}) == "max_contradiction"


def test_document_search_batches_unordered_pairs():
    texts = [f"Requirement {index}." for index in range(7)]
    batch_sizes = []

    def score(pairs):
        batch_sizes.append(len(pairs))
        return [SCORES["contradiction"] if pair == (texts[0], texts[6]) else SCORES["neutral"] for pair in pairs]

    matches = ContradictionDetector(score, "forward").find_contradictions(texts)
    assert batch_sizes == [BATCH_SIZE, 21 - BATCH_SIZE]
    assert [(match["index_a"], match["index_b"]) for match in matches] == [(0, 6)]


@pytest.mark.parametrize("invalid", ["", " \t ", None, 42])
def test_invalid_requirements_are_rejected_before_prediction(invalid):
    detector = ContradictionDetector(lambda _: pytest.fail("Called model for invalid input"), "forward")
    with pytest.raises(ValueError, match="nonempty"):
        detector.analyze_contradiction(ALERT, invalid)
    with pytest.raises(ValueError, match="nonempty"):
        detector.find_contradictions([invalid])
    with pytest.raises(ValueError, match="nonempty"):
        explicit_conflict(ALERT, invalid)


def test_empty_document_and_invalid_strategy():
    scorer = lambda _: pytest.fail("Called model for empty input")
    detector = ContradictionDetector(scorer, "forward")
    assert detector.find_contradictions([]) == []
    with pytest.raises(TypeError, match="collection"):
        detector.find_contradictions(ALERT)
    with pytest.raises(ValueError, match="strategy"):
        ContradictionDetector(scorer, "unknown")


def test_label_mapping_reads_config_indices_and_rejects_disagreement():
    config = SimpleNamespace(
        id2label={0: "NEUTRAL", 1: "CONTRADICTION", 2: "ENTAILMENT"},
        label2id={"CONTRADICTION": 1, "NEUTRAL": 0, "ENTAILMENT": 2},
    )
    assert verify_label_mapping(config) == {"contradiction": 1, "neutral": 0, "entailment": 2}
    config.label2id["CONTRADICTION"] = 0
    with pytest.raises(ValueError, match="mappings disagree"):
        verify_label_mapping(config)


def test_curated_groups_preserve_class_and_family_balance():
    pairs, development, evaluation = build_pairs()
    assert len(pairs) == len({pair["pair_id"] for pair in pairs}) == 120
    assert {pair["group_id"] for pair in development}.isdisjoint(pair["group_id"] for pair in evaluation)
    for partition in (development, evaluation):
        assert Counter(pair["true_label"] for pair in partition) == dict.fromkeys(SCORES, 20)
        assert Counter(pair["family"] for pair in partition) == dict.fromkeys(
            ("obligation", "role", "numeric", "action", "condition"), 12,
        )


@pytest.mark.skipif(os.getenv("SPECSENTRY_TEST_NLI") != "1", reason="Requires the cached NLI checkpoint")
def test_cached_nli_model_outputs_all_scores_and_handles_clear_opposition_and_agreement():
    encoder = load_nli_model(device="cpu", local_files_only=True)
    assert encoder.label_ids == {"contradiction": 0, "entailment": 1, "neutral": 2}
    assert not encoder.model.training
    detector = ContradictionDetector(encoder.score_pairs, "forward")
    results = detector.analyze_pairs([(ALERT, second) for second in (NEGATIVE, PARAPHRASE, UNRELATED)])
    assert [result["label"] for result in results[:2]] == ["contradiction", "entailment"]
    assert len(results) == 3
    for result in results:
        assert sum(result[f"{label}_score"] for label in SCORES) == pytest.approx(1.0)
