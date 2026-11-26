from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analyze import router as analysis_router
from app.api.health import router as health_router
from app.core.config import FRONTEND_ORIGINS, SERVICE_NAME

app = FastAPI(title=SERVICE_NAME, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_methods=["GET", "POST"],
)
app.include_router(health_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
