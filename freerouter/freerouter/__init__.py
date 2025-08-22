"""
FreeRouter Core Package

This package contains the core components of the FreeRouter system.
"""

from .client import OpenRouterClient
from .router import FreeRouter
from .analyzer import QueryAnalyzer
from .evaluator import ModelEvaluator
from .models import ModelManager
from .config import Config
from .exceptions import (
    FreeRouterError,
    ConfigurationError,
    APIError,
    OpenRouterError,
    RateLimitError,
    ModelNotFoundError,
    ModelUnavailableError,
    RoutingError,
    EvaluationError,
    ModelTestError,
    QueryAnalysisError,
    TimeoutError,
    ValidationError,
    CacheError,
    NetworkError,
)

__all__ = [
    "OpenRouterClient",
    "FreeRouter", 
    "QueryAnalyzer",
    "ModelEvaluator",
    "ModelManager",
    "Config",
    # Exceptions
    "FreeRouterError",
    "ConfigurationError",
    "APIError",
    "OpenRouterError",
    "RateLimitError",
    "ModelNotFoundError",
    "ModelUnavailableError",
    "RoutingError",
    "EvaluationError",
    "ModelTestError",
    "QueryAnalysisError",
    "TimeoutError",
    "ValidationError",
    "CacheError",
    "NetworkError",
]