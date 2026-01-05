"""Health and model information endpoints."""

from fastapi import HTTPException, status

from ..model_loader import get_model_info, is_model_loaded
from ..models import HealthResponse, ModelInfoResponse


async def health_check():
    """Health check endpoint."""
    model_loaded = is_model_loaded()
    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        message="Service is running" if model_loaded else "Model not loaded",
    )


async def model_info():
    """Model information endpoint."""
    if not is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    
    info = get_model_info()
    return ModelInfoResponse(**info)

