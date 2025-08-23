"""
FreeRouter: Model routing for resource-constrained developers.

The open-source model router that leverages free models from OpenRouter
to provide intelligent routing with cost optimization.
"""

from .controller import Controller
from .exceptions import FreeRouterError, RateLimitError, ModelUnavailableError

__version__ = "0.1.0"
__all__ = ["Controller", "FreeRouterError", "RateLimitError", "ModelUnavailableError"]