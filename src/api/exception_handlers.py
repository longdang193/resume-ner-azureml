"""Exception handlers for FastAPI application."""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import status

from .exceptions import (
    APIException,
    ModelNotLoadedError,
    TextExtractionError,
    InvalidFileTypeError,
    FileSizeExceededError,
)
from .models import ErrorResponse


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers for the FastAPI application."""
    
    @app.exception_handler(APIException)
    async def api_exception_handler(request, exc: APIException):
        """Handle API exceptions."""
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=str(exc),
            ).dict(),
        )
    
    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request, exc: ModelNotLoadedError):
        """Handle model not loaded errors."""
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(
                error="ModelNotLoadedError",
                message=str(exc),
            ).dict(),
        )
    
    @app.exception_handler(TextExtractionError)
    async def text_extraction_handler(request, exc: TextExtractionError):
        """Handle text extraction errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="TextExtractionError",
                message=str(exc),
            ).dict(),
        )
    
    @app.exception_handler(InvalidFileTypeError)
    async def invalid_file_type_handler(request, exc: InvalidFileTypeError):
        """Handle invalid file type errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="InvalidFileTypeError",
                message=str(exc),
            ).dict(),
        )
    
    @app.exception_handler(FileSizeExceededError)
    async def file_size_exceeded_handler(request, exc: FileSizeExceededError):
        """Handle file size exceeded errors."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error="FileSizeExceededError",
                message=str(exc),
            ).dict(),
        )

