"""Run SpecSentry analysis on a validated requirements document."""

import logging

from fastapi import APIRouter, HTTPException

from app.schemas.analysis import AnalysisError, AnalysisRequest, AnalysisResponse
from app.services.analysis_models import AnalysisUnavailable, get_analysis_engine

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Analysis"])


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze a set of software requirements",
    responses={
        500: {"model": AnalysisError, "description": "Analysis failed during inference"},
        503: {"model": AnalysisError, "description": "A required model or artifact is unavailable"},
    },
)
def analyze_document(payload: AnalysisRequest) -> AnalysisResponse:
    """Return classifications, ambiguity spans, potential duplicate and contradiction pairs, and a heuristic quality score."""
    try:
        return get_analysis_engine().analyze_requirements(payload.requirements)
    except AnalysisUnavailable as error:
        logger.exception("SpecSentry analysis models are unavailable.")
        raise HTTPException(status_code=503, detail=str(error)) from None
    except Exception:
        logger.exception("SpecSentry analysis failed for %d requirements.", len(payload.requirements))
        raise HTTPException(status_code=500, detail="Analysis failed. Check the server logs.") from None
