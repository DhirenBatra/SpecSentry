import json
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from time import sleep
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api import analyze
from app.main import app
from app.schemas.analysis import MAX_REQUIREMENTS, MAX_TEXT_LENGTH
from app.services import analysis_models
from app.services.analysis import AnalysisEngine, calculate_quality

client = TestClient(app)
PAYLOAD = {
    "requirements": [
        {"id": " R1 ", "text": "  The interface should load quickly.  "},
        {"id": "R2", "text": "The system shall email every receipt."},
        {"id": "R3", "text": "The system must send each receipt by email."},
        {"id": "R4", "text": "The system must never email a receipt."},
    ],
}


class StubClassifier:
    def __init__(self, predictions):
        self.predictions = predictions
        self.calls = []

    def predict(self, texts):
        self.calls.append(texts)
        return self.predictions[:len(texts)]


@pytest.fixture
def fake_engine(monkeypatch):
    def duplicate_matches(texts):
        return [
            {
                "index_a": first,
                "index_b": second,
                "requirement_a": texts[first],
                "requirement_b": texts[second],
                "method": "sbert",
                "is_duplicate": True,
                "similarity": 0.81,
                "threshold": analysis_models.DUPLICATE_THRESHOLD,
            }
            for first, second in ((0, 2), (1, 2))
            if second < len(texts)
        ]

    def contradiction_matches(texts):
        return [
            {
                "index_a": first,
                "index_b": second,
                "requirement_a": texts[first],
                "requirement_b": texts[second],
                "label": "contradiction",
                "is_contradiction": True,
                "contradiction_score": 0.91,
                "neutral_score": 0.07,
                "entailment_score": 0.02,
                "model": "cross-encoder/nli-MiniLM2-L6-H768",
                "directionality": "forward",
            }
            for first, second in ((1, 3), (2, 3))
            if second < len(texts)
        ]

    engine = AnalysisEngine(
        StubClassifier(["Non-Functional", "Functional", "Functional", "Functional"]),
        StubClassifier(["US", "F", "F", "F"]),
        {"US": "Usability", "F": "Functional Requirement"},
        duplicate_matches,
        contradiction_matches,
    )
    monkeypatch.setattr(analyze, "get_analysis_engine", lambda: engine)
    return engine


def test_analyze_returns_classifications_and_evidence(fake_engine):
    response = client.post("/api/analyze", json=PAYLOAD)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["summary"] == {
        "total_requirements": 4,
        "quality_score": 35.0,
        "ambiguous_requirements": 1,
        "potential_duplicate_pairs": 2,
        "potential_contradiction_pairs": 2,
        "requirements_in_duplicate_pairs": 3,
        "requirements_in_contradiction_pairs": 3,
    }
    assert result["score_breakdown"] == {
        "ambiguity_penalty": 5.0,
        "duplicate_penalty": 22.5,
        "contradiction_penalty": 37.5,
    }
    first = result["requirements"][0]
    assert first["id"] == "R1"
    assert first["text"] == PAYLOAD["requirements"][0]["text"]
    assert (first["category_code"], first["category"], first["binary_category"]) == ("US", "Usability", "Non-Functional")
    assert first["ambiguity"]["ambiguous"]
    assert {issue["text"] for issue in first["ambiguity"]["issues"]} == {"should", "quickly"}
    for issue in first["ambiguity"]["issues"]:
        assert first["text"][issue["start"]:issue["end"]] == issue["text"]
    assert result["duplicate_pairs"][1] == {
        "requirement_a_id": "R2",
        "requirement_b_id": "R3",
        "is_duplicate": True,
        "method": "sbert",
        "similarity": 0.81,
        "threshold": analysis_models.DUPLICATE_THRESHOLD,
    }
    assert result["contradiction_pairs"][0] == {
        "requirement_a_id": "R2",
        "requirement_b_id": "R4",
        "is_contradiction": True,
        "label": "contradiction",
        "contradiction_score": 0.91,
        "neutral_score": 0.07,
        "entailment_score": 0.02,
        "model": "cross-encoder/nli-MiniLM2-L6-H768",
        "directionality": "forward",
    }
    expected_batch = [[item["text"].strip() for item in PAYLOAD["requirements"]]]
    assert fake_engine.binary_classifier.calls == fake_engine.multiclass_classifier.calls == expected_batch


def test_single_clear_requirement_has_no_quality_penalty_for_its_category(fake_engine):
    response = client.post("/api/analyze", json={
        "requirements": [{"id": "R1", "text": "The status page shall open within 2 seconds."}],
    })
    assert response.status_code == 200
    result = response.json()
    assert result["requirements"][0]["binary_category"] == "Non-Functional"
    assert result["summary"]["quality_score"] == 100.0
    assert result["duplicate_pairs"] == result["contradiction_pairs"] == []
    assert sum(result["score_breakdown"].values()) == 0.0


@pytest.mark.parametrize("total,affected,expected", [
    (1, (0, 0, 0), 100.0),
    (3, (3, 3, 3), 0.0),
    (3, (1, 2, 2), 40.0),
    (6, (2, 4, 4), 40.0),
])
def test_quality_uses_rounded_normalized_penalties(total, affected, expected):
    score, breakdown = calculate_quality(total, *affected)
    assert 0 <= score <= 100
    assert score == expected
    assert score == round(100 - sum(breakdown.model_dump().values()), 2)


@pytest.mark.parametrize("payload", [
    {},
    {"requirements": []},
    {"requirements": [{"id": "R1", "text": "Shall respond."}, {"id": " R1 ", "text": "Shall log."}]},
    {"requirements": [{"id": "   ", "text": "Shall respond."}]},
    {"requirements": [{"id": "R1", "text": ""}]},
    {"requirements": [{"id": "R1", "text": " \n\t "}]},
    {"requirements": [{"id": "R1", "text": "x" * (MAX_TEXT_LENGTH + 1)}]},
    {"requirements": [{"id": str(index), "text": "Shall respond."} for index in range(MAX_REQUIREMENTS + 1)]},
])
def test_invalid_payloads_are_rejected_before_model_loading(monkeypatch, payload):
    monkeypatch.setattr(analyze, "get_analysis_engine", lambda: pytest.fail("Invalid request tried to load models."))
    assert client.post("/api/analyze", json=payload).status_code == 422


def test_missing_request_body_is_rejected():
    assert client.post("/api/analyze").status_code == 422


def test_missing_classifier_returns_useful_503(monkeypatch, tmp_path):
    monkeypatch.setattr(analysis_models, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(analyze, "get_analysis_engine", analysis_models._load_analysis_engine)
    response = client.post("/api/analyze", json=PAYLOAD)
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Missing Phase 3 classifier artifacts" in detail
    assert analysis_models.BINARY_ARTIFACT in detail
    assert analysis_models.MULTICLASS_ARTIFACT in detail
    assert str(tmp_path) not in detail
    assert client.get("/api/health").json() == {"status": "ok", "service": "SpecSentry API"}


def test_corrupted_classifier_is_rejected_before_deserialization(monkeypatch, tmp_path):
    for name in (analysis_models.BINARY_ARTIFACT, analysis_models.MULTICLASS_ARTIFACT):
        (tmp_path / name).write_bytes(b"invalid classifier")
    monkeypatch.setattr(analysis_models, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(analyze, "get_analysis_engine", analysis_models._load_analysis_engine)
    response = client.post("/api/analyze", json=PAYLOAD)
    assert response.status_code == 503
    assert "Its checksum differs." in response.json()["detail"]


@pytest.mark.parametrize("component", ["SBERT", "NLI"])
def test_pretrained_model_load_failure_returns_safe_503(component, monkeypatch, caplog):
    def fail():
        raise RuntimeError("internal model load details")

    monkeypatch.setattr(analysis_models, "_load_classifiers", lambda: (object(), object(), {}))
    monkeypatch.setattr(analyze, "get_analysis_engine", analysis_models._load_analysis_engine)
    monkeypatch.setitem(sys.modules, "app.nlp.duplicates", SimpleNamespace(
        load_sbert_model=fail if component == "SBERT" else object,
        sbert_similarity=lambda model: model,
        DuplicateDetector=lambda scorer, threshold: object(),
    ))
    monkeypatch.setitem(sys.modules, "app.nlp.contradictions", SimpleNamespace(
        load_nli_model=fail,
        ContradictionDetector=object,
    ))
    response = client.post("/api/analyze", json=PAYLOAD)
    assert response.status_code == 503
    assert component in response.json()["detail"]
    assert "internal model load details" not in response.text
    assert "internal model load details" in caplog.text


def test_inference_failure_is_logged_and_returns_safe_500(fake_engine, monkeypatch, caplog):
    def fail(_):
        raise RuntimeError("internal error details")

    monkeypatch.setattr(fake_engine.binary_classifier, "predict", fail)
    response = client.post("/api/analyze", json=PAYLOAD)
    assert response.status_code == 500
    assert response.json() == {"detail": "Analysis failed. Check the server logs."}
    assert "internal error details" in caplog.text


def test_concurrent_requests_share_the_cached_engine(monkeypatch):
    engine = object()
    loads = []
    start = Barrier(4, timeout=5)

    def fake_load():
        loads.append(True)
        sleep(0.02)
        return engine

    def get_engine(_):
        start.wait()
        return analysis_models.get_analysis_engine()

    analysis_models._cached_engine.cache_clear()
    monkeypatch.setattr(analysis_models, "_load_analysis_engine", fake_load)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            outcomes = list(executor.map(get_engine, range(4)))
        assert outcomes == [engine] * 4
        assert loads == [True]
    finally:
        analysis_models._cached_engine.cache_clear()


def test_frozen_pair_settings_match_previous_results():
    results = analysis_models.ROOT / "research" / "results"
    duplicate = json.loads((results / "duplicate_detection_results.json").read_text(encoding="utf-8"))
    nli = json.loads((results / "contradiction_detection_results.json").read_text(encoding="utf-8"))
    assert analysis_models.DUPLICATE_THRESHOLD == duplicate["methods"]["sbert"]["threshold"]
    assert analysis_models.NLI_DIRECTION == nli["directionality"]["selected"]


def test_openapi_documents_analysis_and_local_cors_preflight():
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    operation = schema["paths"]["/api/analyze"]["post"]
    assert {"200", "422", "500", "503"} <= operation["responses"].keys()
    assert operation["requestBody"]["content"]["application/json"]["schema"]["$ref"].endswith("/AnalysisRequest")

    preflight = client.options("/api/analyze", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "http://localhost:5173"
