from fastapi import APIRouter

from app.core.config import SERVICE_NAME
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=SERVICE_NAME)
