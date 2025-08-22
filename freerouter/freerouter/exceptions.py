"""
Custom exceptions for FreeRouter.

This module defines all custom exceptions used throughout the FreeRouter
system for better error handling and debugging.
"""

from typing import Optional, Dict, Any


class FreeRouterError(Exception):
    """Base exception for all FreeRouter errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class ConfigurationError(FreeRouterError):
    """Raised when there's a configuration problem."""
    pass


class APIError(FreeRouterError):
    """Base class for API-related errors."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict] = None):
        super().__init__(message, error_code="API_ERROR")
        self.status_code = status_code
        self.response_data = response_data or {}


class OpenRouterError(APIError):
    """Raised when OpenRouter API returns an error."""
    pass


class RateLimitError(OpenRouterError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str, retry_after: Optional[int] = None):
        super().__init__(message, error_code="RATE_LIMIT")
        self.retry_after = retry_after


class ModelNotFoundError(OpenRouterError):
    """Raised when a requested model is not found."""
    
    def __init__(self, model_id: str):
        super().__init__(f"Model not found: {model_id}", error_code="MODEL_NOT_FOUND")
        self.model_id = model_id


class ModelUnavailableError(FreeRouterError):
    """Raised when no models are available for routing."""
    
    def __init__(self, reason: str = "No models available"):
        super().__init__(reason, error_code="NO_MODELS_AVAILABLE")


class RoutingError(FreeRouterError):
    """Raised when query routing fails."""
    
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(message, error_code="ROUTING_ERROR")
        self.query = query


class EvaluationError(FreeRouterError):
    """Raised when model evaluation fails."""
    
    def __init__(self, message: str, model_id: Optional[str] = None):
        super().__init__(message, error_code="EVALUATION_ERROR")
        self.model_id = model_id


class ModelTestError(FreeRouterError):
    """Raised when model testing fails."""
    
    def __init__(self, message: str, model_id: str):
        super().__init__(message, error_code="MODEL_TEST_ERROR")
        self.model_id = model_id


class QueryAnalysisError(FreeRouterError):
    """Raised when query analysis fails."""
    
    def __init__(self, message: str, query: Optional[str] = None):
        super().__init__(message, error_code="QUERY_ANALYSIS_ERROR")
        self.query = query


class TimeoutError(FreeRouterError):
    """Raised when an operation times out."""
    
    def __init__(self, operation: str, timeout: float):
        super().__init__(f"Operation '{operation}' timed out after {timeout}s", error_code="TIMEOUT")
        self.operation = operation
        self.timeout = timeout


class ValidationError(FreeRouterError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        super().__init__(message, error_code="VALIDATION_ERROR")
        self.field = field


class CacheError(FreeRouterError):
    """Raised when cache operations fail."""
    
    def __init__(self, message: str, operation: str):
        super().__init__(message, error_code="CACHE_ERROR")
        self.operation = operation


class NetworkError(FreeRouterError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, url: Optional[str] = None):
        super().__init__(message, error_code="NETWORK_ERROR")
        self.url = url


# Error handler utility functions

def handle_api_error(response_status: int, response_data: Dict[str, Any]) -> None:
    """
    Handle API errors and raise appropriate exceptions.
    
    Args:
        response_status: HTTP status code
        response_data: Response data from API
        
    Raises:
        Appropriate exception based on error type
    """
    error_message = response_data.get('error', {}).get('message', 'Unknown API error')
    
    if response_status == 401:
        raise OpenRouterError("Authentication failed. Check your API key.", response_status, response_data)
    elif response_status == 403:
        raise OpenRouterError("Access forbidden. Check your permissions.", response_status, response_data)
    elif response_status == 404:
        raise ModelNotFoundError(error_message)
    elif response_status == 429:
        retry_after = response_data.get('retry_after')
        raise RateLimitError(error_message, retry_after)
    elif response_status >= 500:
        raise OpenRouterError(f"Server error: {error_message}", response_status, response_data)
    else:
        raise OpenRouterError(error_message, response_status, response_data)


def format_error_details(error: Exception) -> Dict[str, Any]:
    """
    Format error details for logging or debugging.
    
    Args:
        error: Exception to format
        
    Returns:
        Dictionary with error details
    """
    details = {
        "type": type(error).__name__,
        "message": str(error),
    }
    
    if isinstance(error, FreeRouterError):
        details["error_code"] = error.error_code
        details["details"] = error.details
        
        if hasattr(error, 'status_code'):
            details["status_code"] = error.status_code
        if hasattr(error, 'model_id'):
            details["model_id"] = error.model_id
        if hasattr(error, 'query'):
            details["query"] = error.query
        if hasattr(error, 'retry_after'):
            details["retry_after"] = error.retry_after
    
    return details


class ErrorContext:
    """Context manager for error handling with additional context."""
    
    def __init__(self, operation: str, **context):
        self.operation = operation
        self.context = context
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and issubclass(exc_type, Exception):
            # Add context to FreeRouter errors
            if isinstance(exc_val, FreeRouterError):
                exc_val.details.update({
                    "operation": self.operation,
                    **self.context
                })
            
            # Log the error with context
            import logging
            logger = logging.getLogger(__name__)
            
            error_details = format_error_details(exc_val)
            error_details.update(self.context)
            
            logger.error(f"Error in {self.operation}: {exc_val}", extra={"error_details": error_details})
        
        return False  # Don't suppress the exception