"""Load the frozen classifier artifacts and pair detectors on first use."""

import hashlib
import io
import json
from functools import cache
from pathlib import Path
from threading import Lock

from app.services.analysis import AnalysisEngine

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "ml" / "artifacts"
CLASSIFICATION_RESULTS = ROOT / "research" / "results" / "classification_baseline_results.json"
BINARY_ARTIFACT = "binary_logistic_regression.joblib"
MULTICLASS_ARTIFACT = "multiclass_linear_svm.joblib"
# The held-out experiment used the full-precision cutoff selected on development.
DUPLICATE_THRESHOLD = 0.5158489942550659
NLI_DIRECTION = "forward"
_LOAD_LOCK = Lock()


class AnalysisUnavailable(RuntimeError):
    """A safe explanation for an analysis dependency that failed to load."""


def _restore_classifier(path: Path, metadata: dict):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != metadata["artifact_sha256"]:
        raise AnalysisUnavailable(f"Restore the original Phase 3 {path.name} in ml/artifacts. Its checksum differs.")

    import joblib

    classifier = joblib.load(io.BytesIO(data))
    if set(classifier.classes_) != set(metadata["label_names"]):
        raise ValueError(f"Unexpected classifier labels for {path.name}.")
    return classifier


def _load_classifiers():
    binary_path, multiclass_path = (ARTIFACTS / name for name in (BINARY_ARTIFACT, MULTICLASS_ARTIFACT))
    missing = [path.name for path in (binary_path, multiclass_path) if not path.is_file()]
    if missing:
        raise AnalysisUnavailable(f"Missing Phase 3 classifier artifacts. Restore {', '.join(missing)} in ml/artifacts.")
    try:
        tasks = json.loads(CLASSIFICATION_RESULTS.read_text(encoding="utf-8"))["tasks"]
        binary_metadata = tasks["binary"]["models"]["logistic_regression"]
        multiclass_metadata = tasks["multiclass"]["models"]["linear_svm"]
        binary = _restore_classifier(binary_path, binary_metadata)
        multiclass = _restore_classifier(multiclass_path, multiclass_metadata)
    except AnalysisUnavailable:
        raise
    except Exception as error:
        raise AnalysisUnavailable("Phase 3 classifiers could not load. Check the analysis dependencies and artifact files.") from error
    return binary, multiclass, multiclass_metadata["label_names"]


def _load_analysis_engine() -> AnalysisEngine:
    binary, multiclass, category_names = _load_classifiers()
    try:
        from app.nlp.duplicates import DuplicateDetector, load_sbert_model, sbert_similarity

        duplicates = DuplicateDetector(sbert_similarity(load_sbert_model()), DUPLICATE_THRESHOLD)
    except Exception as error:
        raise AnalysisUnavailable("The SBERT encoder could not load. Check the analysis dependencies and Hugging Face cache.") from error
    try:
        from app.nlp.contradictions import ContradictionDetector, load_nli_model

        contradictions = ContradictionDetector(load_nli_model().score_pairs, NLI_DIRECTION)
    except Exception as error:
        raise AnalysisUnavailable("The NLI model could not load. Check the analysis dependencies and Hugging Face cache.") from error
    return AnalysisEngine(binary, multiclass, category_names, duplicates.find_duplicates, contradictions.find_contradictions)


@cache
def _cached_engine() -> AnalysisEngine:
    return _load_analysis_engine()


def get_analysis_engine() -> AnalysisEngine:
    """Serialize initial loading so concurrent first requests share one engine."""
    with _LOAD_LOCK:
        return _cached_engine()
