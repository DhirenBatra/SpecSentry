"""Request and evidence schemas for document analysis."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

MAX_REQUIREMENTS = 20
MAX_TEXT_LENGTH = 500
RequirementId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=64)]


class RequirementInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: RequirementId
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH, description="Requirement text with original spacing preserved.")

    @field_validator("text")
    @classmethod
    def reject_blank_text(cls, text: str) -> str:
        if not text.strip():
            raise ValueError("Requirement text must not be blank.")
        return text


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirements: list[RequirementInput] = Field(
        min_length=1, max_length=MAX_REQUIREMENTS,
        description="Requirements with unique IDs. Pair analysis follows input order.",
    )

    @field_validator("requirements")
    @classmethod
    def require_unique_ids(cls, requirements: list[RequirementInput]) -> list[RequirementInput]:
        if len({requirement.id for requirement in requirements}) != len(requirements):
            raise ValueError("Requirement IDs must be unique.")
        return requirements


class AmbiguityIssue(BaseModel):
    text: str
    category: str
    explanation: str
    start: int
    end: int
    weight: int


class AmbiguityFinding(BaseModel):
    ambiguous: bool
    score: int = Field(ge=0, le=10)
    severity: Literal["low", "medium", "high"]
    issues: list[AmbiguityIssue]


class AnalyzedRequirement(RequirementInput):
    category_code: str
    category: str
    binary_category: Literal["Functional", "Non-Functional"]
    ambiguity: AmbiguityFinding


class PairEvidence(BaseModel):
    requirement_a_id: str
    requirement_b_id: str


class DuplicatePair(PairEvidence):
    method: Literal["sbert"]
    is_duplicate: Literal[True]
    similarity: float = Field(ge=-1, le=1)
    threshold: float = Field(ge=-1, le=1)


class ContradictionPair(PairEvidence):
    label: Literal["contradiction"]
    is_contradiction: Literal[True]
    contradiction_score: float = Field(ge=0, le=1)
    neutral_score: float = Field(ge=0, le=1)
    entailment_score: float = Field(ge=0, le=1)
    model: str
    directionality: Literal["forward"]


class ScoreBreakdown(BaseModel):
    ambiguity_penalty: float = Field(ge=0, le=20)
    duplicate_penalty: float = Field(ge=0, le=30)
    contradiction_penalty: float = Field(ge=0, le=50)


class AnalysisSummary(BaseModel):
    total_requirements: int
    quality_score: float = Field(ge=0, le=100, description="Heuristic score based on affected requirement fractions.")
    ambiguous_requirements: int
    potential_duplicate_pairs: int
    potential_contradiction_pairs: int
    requirements_in_duplicate_pairs: int
    requirements_in_contradiction_pairs: int


class AnalysisResponse(BaseModel):
    summary: AnalysisSummary
    requirements: list[AnalyzedRequirement]
    duplicate_pairs: list[DuplicatePair]
    contradiction_pairs: list[ContradictionPair]
    score_breakdown: ScoreBreakdown


class AnalysisError(BaseModel):
    detail: str
