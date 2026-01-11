"""FastAPI application for NER predictions."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import APIConfig
from .startup import startup_event, shutdown_event
from .exception_handlers import register_exception_handlers
from .routes import health, predictions

app = FastAPI(
    title="Resume NER API",
    description="Named Entity Recognition API for resume parsing",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=APIConfig.CORS_ORIGINS,
    allow_credentials=APIConfig.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup():
    """Startup event handler."""
    startup_event(app)

@app.on_event("shutdown")
async def shutdown():
    """Shutdown event handler."""
    shutdown_event(app)

# Register exception handlers
register_exception_handlers(app)

# Register routes
from .models import HealthResponse, ModelInfoResponse, PredictionResponse, BatchPredictionResponse

app.add_api_route("/health", health.health_check, methods=["GET"], response_model=HealthResponse)
app.add_api_route("/info", health.model_info, methods=["GET"], response_model=ModelInfoResponse)
app.add_api_route("/predict/debug", predictions.predict_debug, methods=["POST"])
app.add_api_route("/predict", predictions.predict, methods=["POST"], response_model=PredictionResponse)
app.add_api_route("/predict/batch", predictions.predict_batch, methods=["POST"], response_model=BatchPredictionResponse)
app.add_api_route("/predict/file", predictions.predict_file, methods=["POST"], response_model=PredictionResponse)
app.add_api_route("/predict/file/batch", predictions.predict_file_batch, methods=["POST"], response_model=BatchPredictionResponse)
