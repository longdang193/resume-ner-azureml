"""Custom exceptions for the API."""


class APIException(Exception):
    """Base exception for API errors."""

    pass


class ModelNotLoadedError(APIException):
    """Raised when model is not loaded."""

    pass


class TextExtractionError(APIException):
    """Raised when text extraction fails."""

    pass


class InvalidFileTypeError(APIException):
    """Raised when file type is not supported."""

    pass


class FileSizeExceededError(APIException):
    """Raised when file size exceeds limit."""

    pass


class InferenceError(APIException):
    """Raised when inference fails."""

    pass


