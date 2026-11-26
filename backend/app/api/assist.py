"""Expose explicit user requests for LLM-assisted requirement feedback."""

import logging
from typing import Annotated

from fastapi import APIRouter, Body, HTTPException

from app.schemas.analysis import AnalysisError
from app.schemas.assistance import AssistanceRequest, AssistanceResponse
from app.services.assistance import AssistanceError, get_assistance_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Optional AI assistance"])


@router.post(
    "/ai/assist",
    response_model=AssistanceResponse,
    summary="Explain an existing finding and suggest an improvement",
    responses={
        500: {"model": AnalysisError, "description": "Assistance failed"},
        502: {"model": AnalysisError, "description": "Provider failure or invalid suggestion"},
        503: {"model": AnalysisError, "description": "Optional assistance is unavailable"},
        504: {"model": AnalysisError, "description": "Provider timeout"},
    },
)
def assist(payload: Annotated[AssistanceRequest, Body(discriminator="action")]) -> AssistanceResponse:
    """Send selected requirement text and existing evidence to the configured provider on demand."""
    try:
        return get_assistance_service().assist(payload)
    except AssistanceError as error:
        raise HTTPException(status_code=error.status_code, detail=str(error)) from None
    except Exception as error:
        logger.error("Unexpected AI assistance error (%s).", type(error).__name__)
        raise HTTPException(status_code=500, detail="AI assistance failed. Check the server logs.") from None
