"""
FreeRouter: Model routing for resource-constrained developers.

The open-source model router that leverages free models from OpenRouter
to provide intelligent routing with cost optimization.
"""

from .controller import Controller
from .matrix_factorization_router import MatrixFactorizationRouter
from .random_forest_router import RandomForestRouter
from .performance_monitor import PerformanceMonitor
from .huggingface_integration import HuggingFaceIntegration
from .exceptions import FreeRouterError, RateLimitError, ModelUnavailableError

__version__ = "0.2.0"
__all__ = [
    "Controller", 
    "MatrixFactorizationRouter", 
    "RandomForestRouter", 
    "PerformanceMonitor",
    "HuggingFaceIntegration",
    "FreeRouterError", 
    "RateLimitError", 
    "ModelUnavailableError"
]