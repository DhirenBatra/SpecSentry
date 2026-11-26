"""Compare requirement text with fitted TF-IDF or frozen sentence embeddings."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import TypedDict

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
SBERT_BATCH_SIZE = 32
SBERT_CACHE = Path(__file__).resolve().parents[3] / "ml" / "artifacts" / "duplicate_detection" / "huggingface"
TFIDF_OPTIONS = {"ngram_range": (1, 1), "token_pattern": r"(?u)\b\w+\b", "norm": "l2"}


class DuplicateResult(TypedDict):
    is_duplicate: bool
    similarity: float
    threshold: float
    method: str
    requirement_a: str
    requirement_b: str


class DocumentDuplicate(DuplicateResult):
    index_a: int
    index_b: int


def _texts(requirements: Iterable[str]) -> list[str]:
    if isinstance(requirements, (str, bytes)):
        raise TypeError("Pass a collection of requirement strings.")
    texts = list(requirements)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Each requirement must be a nonempty string.")
    return texts


def _row_cosines(left, right) -> np.ndarray:
    # Both encoders produce unit rows. Out-of-vocabulary TF-IDF rows stay zero.
    if issparse(left):
        scores = np.asarray(left.multiply(right).sum(axis=1)).ravel()
    else:
        scores = np.einsum("ij,ij->i", left, right)
    return np.clip(scores.astype(float, copy=False), -1.0, 1.0)


@dataclass(frozen=True)
class SimilarityScorer:
    method: str
    encode: Callable[[list[str]], np.ndarray | csr_matrix]

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[float]:
        """Score explicit pairs, encoding each distinct text once for the batch."""
        pairs = list(pairs)
        if not pairs:
            return []
        texts = _texts(text for left, right in pairs for text in (left, right))
        lookup = {text: index for index, text in enumerate(dict.fromkeys(texts))}
        vectors = self.encode(list(lookup))
        left = vectors[[lookup[first] for first, _ in pairs]]
        right = vectors[[lookup[second] for _, second in pairs]]
        return _row_cosines(left, right).tolist()


def fit_tfidf(development_texts: Iterable[str]) -> SimilarityScorer:
    texts = _texts(development_texts)
    vectorizer = TfidfVectorizer(**TFIDF_OPTIONS)
    vectorizer.fit(list(dict.fromkeys(texts)))
    return SimilarityScorer("tfidf", vectorizer.transform)


@lru_cache(maxsize=2)
def load_sbert_model(*, device: str | None = None):
    """Load the pinned public model once in a process for the requested device."""
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer(
        SBERT_MODEL,
        revision=SBERT_REVISION,
        cache_folder=str(SBERT_CACHE),
        device=device,
        token=False,
        model_kwargs={"use_safetensors": True},
    )


def sbert_similarity(model) -> SimilarityScorer:
    encode = partial(
        model.encode,
        batch_size=SBERT_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return SimilarityScorer("sbert", encode)


@dataclass(frozen=True)
class DuplicateDetector:
    scorer: SimilarityScorer
    threshold: float

    def __post_init__(self):
        if not -1.0 <= self.threshold <= 1.0:
            raise ValueError("The cosine threshold must be between -1 and 1.")

    def _result(self, first: str, second: str, similarity: float) -> DuplicateResult:
        return {
            "is_duplicate": similarity >= self.threshold,
            "similarity": similarity,
            "threshold": self.threshold,
            "method": self.scorer.method,
            "requirement_a": first,
            "requirement_b": second,
        }

    def analyze_duplicate(self, requirement_a: str, requirement_b: str) -> DuplicateResult:
        similarity, = self.scorer.score_pairs([(requirement_a, requirement_b)])
        return self._result(requirement_a, requirement_b, similarity)

    def find_duplicates(self, requirements: Iterable[str]) -> list[DocumentDuplicate]:
        """Return matching pairs with zero-based document indices in input order."""
        texts = _texts(requirements)
        if len(texts) < 2:
            return []
        lookup = {text: index for index, text in enumerate(dict.fromkeys(texts))}
        vectors = self.scorer.encode(list(lookup))
        positions = np.array([lookup[text] for text in texts])
        matches = []
        for first, position in enumerate(positions[:-1]):
            later = positions[first + 1:]
            scores = _row_cosines(vectors[np.full(later.size, position)], vectors[later])
            for offset in np.flatnonzero(scores >= self.threshold):
                second = first + 1 + int(offset)
                result = self._result(texts[first], texts[second], float(scores[offset]))
                matches.append(DocumentDuplicate(**result, index_a=first, index_b=second))
        return matches
