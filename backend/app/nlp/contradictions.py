"""Run frozen NLI on software-requirement pairs."""

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from itertools import batched, combinations
from pathlib import Path
from typing import TypedDict

MODEL_NAME = "cross-encoder/nli-MiniLM2-L6-H768"
MODEL_REVISION = "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d"
CACHE_DIR = Path(__file__).resolve().parents[3] / "ml" / "artifacts" / "contradiction_detection" / "huggingface"
LABELS = ("contradiction", "neutral", "entailment")
DIRECTIONS = ("forward", "reverse", "max_contradiction")
BATCH_SIZE = 16
MAX_LENGTH = 256


class ContradictionResult(TypedDict):
    is_contradiction: bool
    label: str
    contradiction_score: float
    neutral_score: float
    entailment_score: float
    model: str
    directionality: str
    requirement_a: str
    requirement_b: str


class DocumentContradiction(ContradictionResult):
    index_a: int
    index_b: int


def checked_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Each requirement must be a nonempty string.")
    return text


def verify_label_mapping(config) -> dict[str, int]:
    """Read output indices from the checkpoint and check its inverse mapping."""
    if not config.id2label or not config.label2id:
        raise ValueError("Model configuration must provide both NLI label mappings.")
    indices = {label.casefold(): int(index) for index, label in config.id2label.items()}
    inverse = {label.casefold(): int(index) for label, index in config.label2id.items()}
    if len(config.id2label) != 3 or set(indices) != set(LABELS) or set(indices.values()) != {0, 1, 2}:
        raise ValueError(f"Expected three NLI labels in the model configuration, got {config.id2label}.")
    if len(config.label2id) != 3 or indices != inverse:
        raise ValueError("Model id2label and label2id mappings disagree.")
    return indices


@dataclass(frozen=True)
class NLIEncoder:
    tokenizer: object
    model: object
    label_ids: dict[str, int]
    device: str

    def score_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[dict[str, float]]:
        import torch

        output = []
        with torch.inference_mode():
            for batch in batched(pairs, BATCH_SIZE):
                first, second = zip(*batch, strict=True)
                inputs = self.tokenizer(
                    [checked_text(text) for text in first],
                    [checked_text(text) for text in second],
                    padding=True,
                    truncation="longest_first",
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self.device)
                probabilities = self.model(**inputs).logits.float().softmax(dim=-1).cpu().tolist()
                output.extend({label: row[index] for label, index in self.label_ids.items()} for row in probabilities)
        return output


@lru_cache(maxsize=2)
def load_nli_model(*, device: str | None = None, local_files_only: bool = False) -> NLIEncoder:
    """Load and validate the pinned model once for this process and device."""
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    options = {
        "revision": MODEL_REVISION,
        "cache_dir": CACHE_DIR,
        "token": False,
        "local_files_only": local_files_only,
    }
    config = AutoConfig.from_pretrained(MODEL_NAME, **options)
    label_ids = verify_label_mapping(config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **options)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=config, use_safetensors=True, **options,
    )
    model.to(device).eval()
    return NLIEncoder(tokenizer, model, label_ids, device)


def max_contradiction_scores(forward, reverse):
    """Keep the entire score vector from the direction with higher contradiction."""
    return reverse if reverse["contradiction"] > forward["contradiction"] else forward


def predicted_label(scores: dict[str, float]) -> str:
    return max(LABELS, key=scores.__getitem__)


@dataclass(frozen=True)
class ContradictionDetector:
    score_pairs: Callable[[Iterable[tuple[str, str]]], list[dict[str, float]]]
    directionality: str

    def __post_init__(self):
        if self.directionality not in DIRECTIONS:
            raise ValueError(f"Unknown directionality strategy: {self.directionality}.")

    def _analyze_pairs(self, pairs: list[tuple[str, str]]) -> list[ContradictionResult]:
        pairs = [(checked_text(first), checked_text(second)) for first, second in pairs]
        if self.directionality == "forward":
            scores = self.score_pairs(pairs)
        else:
            reversed_pairs = [(second, first) for first, second in pairs]
            if self.directionality == "reverse":
                scores = self.score_pairs(reversed_pairs)
            else:
                forward = self.score_pairs(pairs)
                reverse = self.score_pairs(reversed_pairs)
                scores = [max_contradiction_scores(a, b) for a, b in zip(forward, reverse, strict=True)]
        results = []
        for (first, second), probabilities in zip(pairs, scores, strict=True):
            label = predicted_label(probabilities)
            results.append({
                "is_contradiction": label == "contradiction",
                "label": label,
                "contradiction_score": probabilities["contradiction"],
                "neutral_score": probabilities["neutral"],
                "entailment_score": probabilities["entailment"],
                "model": MODEL_NAME,
                "directionality": self.directionality,
                "requirement_a": first,
                "requirement_b": second,
            })
        return results

    def analyze_pairs(self, pairs: Iterable[tuple[str, str]]) -> list[ContradictionResult]:
        return self._analyze_pairs(list(pairs))

    def analyze_contradiction(self, requirement_a: str, requirement_b: str) -> ContradictionResult:
        result, = self._analyze_pairs([(requirement_a, requirement_b)])
        return result

    def find_contradictions(self, requirements: Iterable[str]) -> list[DocumentContradiction]:
        """Scan unordered pairs in bounded batches and return zero-based positions."""
        if isinstance(requirements, (str, bytes)):
            raise TypeError("Pass a collection of requirement strings.")
        texts = [checked_text(text) for text in requirements]
        matches = []
        for batch in batched(combinations(range(len(texts)), 2), BATCH_SIZE):
            results = self._analyze_pairs([(texts[first], texts[second]) for first, second in batch])
            for (first, second), result in zip(batch, results, strict=True):
                if result["is_contradiction"]:
                    matches.append(DocumentContradiction(**result, index_a=first, index_b=second))
        return matches
