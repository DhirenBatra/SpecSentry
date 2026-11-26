"""Combine the validated NLP components and summarize document quality."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING

from app.nlp.ambiguity import detect_ambiguity
from app.schemas.analysis import (
    AnalysisResponse,
    AnalysisSummary,
    AnalyzedRequirement,
    ContradictionPair,
    DuplicatePair,
    PairEvidence,
    RequirementInput,
    ScoreBreakdown,
)

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

    from app.nlp.contradictions import DocumentContradiction
    from app.nlp.duplicates import DocumentDuplicate


def _affected_requirements(pairs: Iterable[PairEvidence]) -> int:
    return len({identifier for pair in pairs for identifier in (pair.requirement_a_id, pair.requirement_b_id)})


def calculate_quality(
    total: int, ambiguous: int, in_duplicates: int, in_contradictions: int,
) -> tuple[float, ScoreBreakdown]:
    """Penalize the fraction of requirements affected by each kind of issue."""
    penalties = ScoreBreakdown(
        ambiguity_penalty=round(20 * ambiguous / total, 2),
        duplicate_penalty=round(30 * in_duplicates / total, 2),
        contradiction_penalty=round(50 * in_contradictions / total, 2),
    )
    score = 100 - sum(penalties.model_dump().values())
    return round(max(0.0, min(100.0, score)), 2), penalties


@dataclass(frozen=True)
class AnalysisEngine:
    binary_classifier: Pipeline
    multiclass_classifier: Pipeline
    category_names: dict[str, str]
    find_duplicates: Callable[[list[str]], list[DocumentDuplicate]]
    find_contradictions: Callable[[list[str]], list[DocumentContradiction]]
    _inference_lock: Lock = field(default_factory=Lock, init=False, repr=False, compare=False)

    def analyze_requirements(self, requirements: list[RequirementInput]) -> AnalysisResponse:
        texts = [requirement.text for requirement in requirements]
        classifier_texts = [" ".join(text.split()) for text in texts]
        with self._inference_lock:
            binary_labels = self.binary_classifier.predict(classifier_texts)
            category_codes = self.multiclass_classifier.predict(classifier_texts)
            duplicates = self.find_duplicates(texts)
            contradictions = self.find_contradictions(texts)

        analyzed = [
            AnalyzedRequirement(
                id=requirement.id,
                text=requirement.text,
                category_code=code,
                category=self.category_names[code],
                binary_category=binary,
                ambiguity=detect_ambiguity(requirement.text),
            )
            for requirement, code, binary in zip(requirements, category_codes, binary_labels, strict=True)
        ]
        duplicate_pairs = [
            DuplicatePair(
                requirement_a_id=requirements[match["index_a"]].id,
                requirement_b_id=requirements[match["index_b"]].id,
                method=match["method"],
                is_duplicate=match["is_duplicate"],
                similarity=match["similarity"],
                threshold=match["threshold"],
            )
            for match in duplicates
        ]
        contradiction_pairs = [
            ContradictionPair(
                requirement_a_id=requirements[match["index_a"]].id,
                requirement_b_id=requirements[match["index_b"]].id,
                label=match["label"],
                is_contradiction=match["is_contradiction"],
                contradiction_score=match["contradiction_score"],
                neutral_score=match["neutral_score"],
                entailment_score=match["entailment_score"],
                model=match["model"],
                directionality=match["directionality"],
            )
            for match in contradictions
        ]
        ambiguous = sum(requirement.ambiguity.ambiguous for requirement in analyzed)
        in_duplicates = _affected_requirements(duplicate_pairs)
        in_contradictions = _affected_requirements(contradiction_pairs)
        total = len(requirements)
        score, penalties = calculate_quality(total, ambiguous, in_duplicates, in_contradictions)
        return AnalysisResponse(
            summary=AnalysisSummary(
                total_requirements=total,
                quality_score=score,
                ambiguous_requirements=ambiguous,
                potential_duplicate_pairs=len(duplicate_pairs),
                potential_contradiction_pairs=len(contradiction_pairs),
                requirements_in_duplicate_pairs=in_duplicates,
                requirements_in_contradiction_pairs=in_contradictions,
            ),
            requirements=analyzed,
            duplicate_pairs=duplicate_pairs,
            contradiction_pairs=contradiction_pairs,
            score_breakdown=penalties,
        )
