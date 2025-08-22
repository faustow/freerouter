"""
FreeRouter - Intelligent Model Router for OpenRouter Free Models

A production-ready Python application that intelligently routes user queries 
to the most appropriate free model available on OpenRouter.
"""

__version__ = "0.1.0"
__author__ = "FreeRouter Team"
__email__ = "contact@freerouter.dev"

from .router import FreeRouter
from .client import OpenRouterClient
from .analyzer import QueryAnalyzer
from .evaluator import ModelEvaluator
from .models import ModelManager

__all__ = [
    "FreeRouter",
    "OpenRouterClient", 
    "QueryAnalyzer",
    "ModelEvaluator",
    "ModelManager",
]