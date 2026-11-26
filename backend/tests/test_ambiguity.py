import pytest

from app.nlp.ambiguity import detect_ambiguity


def test_measurable_response_requirement_is_clear():
    result = detect_ambiguity("The system shall respond within 2 seconds.")
    assert result == {"ambiguous": False, "score": 0, "severity": "low", "issues": []}


def test_vague_response_time_returns_explanation_and_medium_severity():
    result = detect_ambiguity("The system should respond quickly.")
    assert result["ambiguous"] is True
    assert (result["score"], result["severity"]) == (3, "medium")
    assert [(issue["text"], issue["category"]) for issue in result["issues"]] == [
        ("should", "weak_obligation"), ("quickly", "vague_term"),
    ]
    assert "response-time threshold" in result["issues"][1]["explanation"]


def test_multiple_subjective_signals_keep_their_original_spans():
    text = "The interface should be Intuitive and user-friendly."
    result = detect_ambiguity(text)
    assert result["score"] == 5
    assert [issue["text"] for issue in result["issues"]] == ["should", "Intuitive", "user-friendly"]
    for issue in result["issues"]:
        assert text[issue["start"]:issue["end"]] == issue["text"]
        assert issue["weight"] > 0


def test_compound_phrases_own_their_spans():
    result = detect_ambiguity("The service may retry where appropriate and log it as soon as possible.")
    assert [issue["text"] for issue in result["issues"]] == [
        "may", "where appropriate", "as soon as possible",
    ]
    assert result["score"] == 5


def test_repeated_cues_score_once_and_total_caps_at_ten():
    duplicated = detect_ambiguity("The service shall work quickly and QUICKLY.")
    assert len(duplicated["issues"]) == 2
    assert duplicated["score"] == 2
    combined = detect_ambiguity(
        "It should usually work quickly and efficiently as appropriate, if possible, and so on."
    )
    assert combined["score"] == 10
    assert combined["severity"] == "high"
    assert {issue["category"] for issue in combined["issues"]} == {
        "unclear_reference", "weak_obligation", "weak_quantifier", "vague_term", "open_ended_phrase",
    }


def test_undefined_comparison_and_opening_reference_are_flagged():
    result = detect_ambiguity("When it fails, the service shall switch to a better connection.")
    assert [(issue["text"], issue["category"]) for issue in result["issues"]] == [
        ("it", "unclear_reference"), ("better", "undefined_comparative"),
    ]


@pytest.mark.parametrize("text", [
    'The payment service shall write "Retry quickly if possible" to the audit log.',
    "The scheduler shall read the SomeQueue setting at startup.",
    "The system shall run the batch on 1 May 2026.",
    "Authenticated users may download their own invoices.",
    "The service shall respond quickly, within 2 seconds.",
    "The ADC shall sample approximately 10 Hz (± 1 Hz).",
    "The error rate shall be lower than 5%.",
    "It shall be possible to export the report as CSV.",
    "The service shall record the result and send it to the auditor.",
    "The API shall return the header as required by RFC 9110.",
])
def test_false_positive_guards(text):
    assert detect_ambiguity(text)["ambiguous"] is False


def test_vague_only_ablation_uses_same_span_resolution():
    full = detect_ambiguity("The service should respond quickly as appropriate.")
    vague = detect_ambiguity("The service should respond quickly as appropriate.", vague_only=True)
    assert vague["issues"] == [issue for issue in full["issues"] if issue["category"] == "vague_term"]
    assert vague["score"] == 2
    assert detect_ambiguity("The service should respond.", vague_only=True)["ambiguous"] is False


@pytest.mark.parametrize("text", ["", " \t\n"])
def test_empty_requirement_is_rejected(text):
    with pytest.raises(ValueError, match="blank"):
        detect_ambiguity(text)


@pytest.mark.parametrize("text", [None, 27, ["The system shall respond."]])
def test_wrong_type_is_rejected(text):
    with pytest.raises(TypeError, match="string"):
        detect_ambiguity(text)
