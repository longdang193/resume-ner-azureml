"""Pydantic models for API requests and responses."""

from typing import List, Optional
from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Entity output model."""

    text: str = Field(..., description="The extracted entity text")
    label: str = Field(..., description="Entity type (e.g., SKILL, NAME, EMAIL)")
    start: int = Field(..., description="Start character position in original text")
    end: int = Field(..., description="End character position in original text")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")


class TextRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(..., description="Input text to extract entities from", min_length=1)


class BatchTextRequest(BaseModel):
    """Batch text prediction request."""

    texts: List[str] = Field(..., description="List of input texts", min_items=1, max_items=32)


class PredictionResponse(BaseModel):
    """Single prediction response."""

    entities: List[Entity] = Field(..., description="List of extracted entities")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    extracted_text: Optional[str] = Field(None, description="Extracted text (for file uploads)")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_processing_time_ms: float = Field(..., description="Total processing time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    message: str = Field(..., description="Status message")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    backbone: str = Field(..., description="Model backbone name")
    entity_types: List[str] = Field(..., description="Supported entity types")
    max_sequence_length: int = Field(..., description="Maximum sequence length")
    version: str = Field(..., description="Model version")


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")


