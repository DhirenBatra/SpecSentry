"""Typed requests and suggestions for one previously detected finding."""

from math import isclose
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.schemas.analysis import MAX_TEXT_LENGTH, AmbiguityIssue, ContradictionPair, RequirementId


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("Text must not be blank.")
    return value


RequirementText = Annotated[str, Field(min_length=1, max_length=MAX_TEXT_LENGTH), AfterValidator(_nonblank)]


class AmbiguityEvidence(AmbiguityIssue):
    model_config = ConfigDict(extra="forbid")

    text: RequirementText
    category: Literal[
        "vague_term", "weak_quantifier", "weak_obligation", "undefined_comparative",
        "unclear_reference", "open_ended_phrase",
    ]
    explanation: Annotated[str, Field(min_length=1, max_length=250), AfterValidator(_nonblank)]
    start: int = Field(ge=0, lt=MAX_TEXT_LENGTH)
    end: int = Field(gt=0, le=MAX_TEXT_LENGTH)
    weight: int = Field(ge=1, le=3)


class AmbiguityRewriteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["rewrite_ambiguity"]
    requirement: RequirementText
    issues: list[AmbiguityEvidence] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_issue_spans(self) -> Self:
        for issue in self.issues:
            if (
                not issue.start < issue.end <= len(self.requirement)
                or self.requirement[issue.start:issue.end] != issue.text
            ):
                raise ValueError("Each issue span must match the supplied requirement.")
        return self


class ContradictionEvidence(ContradictionPair):
    model_config = ConfigDict(extra="forbid")

    requirement_a_id: RequirementId
    requirement_b_id: RequirementId
    model: Annotated[str, Field(min_length=1, max_length=120), AfterValidator(_nonblank)]

    @model_validator(mode="after")
    def validate_pair_evidence(self) -> Self:
        scores = (self.contradiction_score, self.neutral_score, self.entailment_score)
        if self.requirement_a_id == self.requirement_b_id:
            raise ValueError("Contradiction evidence must refer to distinct requirement IDs.")
        if self.contradiction_score < max(scores) or not isclose(sum(scores), 1, abs_tol=0.001):
            raise ValueError("Contradiction evidence must contain normalized scores with contradiction highest.")
        return self


class ContradictionExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal["explain_contradiction"]
    requirement_a: RequirementText
    requirement_b: RequirementText
    evidence: ContradictionEvidence


AssistanceRequest = AmbiguityRewriteRequest | ContradictionExplanationRequest


class AssistanceSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    explanation: str

    @field_validator("*")
    @classmethod
    def validate_suggestion_text(cls, value: str) -> str:
        text = value.strip()
        if not 1 <= len(text) <= 800:
            raise ValueError("Each response field must contain 1 to 800 characters.")
        return text


class AmbiguityRewrite(AssistanceSuggestion):
    suggested_rewrite: str


class ContradictionExplanation(AssistanceSuggestion):
    suggested_resolution: str


AssistanceResponse = AmbiguityRewrite | ContradictionExplanation
