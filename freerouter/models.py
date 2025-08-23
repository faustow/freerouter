"""Data models and configurations for FreeRouter."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class IntentType(Enum):
    """Intent categories for semantic routing."""
    CODING = "coding"
    WRITING = "writing"
    ANALYSIS = "analysis"
    MATHEMATICS = "mathematics"
    GENERAL = "general"


@dataclass
class ModelInfo:
    """Information about a model in the routing pool."""
    id: str
    name: str
    provider: str
    max_tokens: int
    rate_limit_rpm: int  # requests per minute
    rate_limit_rpd: int  # requests per day
    intent_specialization: List[IntentType]
    cost_tier: int = 0  # 0 = free, higher = more expensive
    context_length: int = 4096
    supports_streaming: bool = True


@dataclass
class RouteDecision:
    """Result of a routing decision."""
    model_id: str
    confidence: float
    intent: IntentType
    reasoning: str
    fallback_models: List[str]


@dataclass
class RouterConfig:
    """Configuration for the router."""
    precision_threshold: float = 0.3
    enable_fallbacks: bool = True
    max_fallbacks: int = 3
    intent_detection_threshold: float = 0.7
    rate_limit_buffer: float = 0.1  # Keep 10% buffer for rate limits


# OpenRouter free models pool based on research
FREE_MODELS = {
    "deepseek/deepseek-r1": ModelInfo(
        id="deepseek/deepseek-r1",
        name="DeepSeek R1",
        provider="DeepSeek",
        max_tokens=8192,
        rate_limit_rpm=20,
        rate_limit_rpd=1000,
        intent_specialization=[IntentType.CODING, IntentType.ANALYSIS, IntentType.MATHEMATICS],
        context_length=32768
    ),
    "kimi/moonshot-v1-auto": ModelInfo(
        id="kimi/moonshot-v1-auto", 
        name="Kimi K2 (1T params)",
        provider="Moonshot",
        max_tokens=4096,
        rate_limit_rpm=20,
        rate_limit_rpd=50,
        intent_specialization=[IntentType.WRITING, IntentType.ANALYSIS],
        context_length=200000
    ),
    "google/gemma-2-9b-it:free": ModelInfo(
        id="google/gemma-2-9b-it:free",
        name="Gemma 2 9B IT",
        provider="Google",
        max_tokens=8192,
        rate_limit_rpm=20,
        rate_limit_rpd=200,
        intent_specialization=[IntentType.GENERAL, IntentType.WRITING],
        context_length=8192
    ),
    "meta-llama/llama-3-8b-instruct:free": ModelInfo(
        id="meta-llama/llama-3-8b-instruct:free",
        name="Llama 3 8B Instruct",
        provider="Meta",
        max_tokens=8192,
        rate_limit_rpm=20,
        rate_limit_rpd=200,
        intent_specialization=[IntentType.GENERAL, IntentType.CODING],
        context_length=8192
    ),
    "microsoft/phi-3-medium-128k-instruct:free": ModelInfo(
        id="microsoft/phi-3-medium-128k-instruct:free",
        name="Phi-3 Medium 128K",
        provider="Microsoft",
        max_tokens=4096,
        rate_limit_rpm=20,
        rate_limit_rpd=200,
        intent_specialization=[IntentType.CODING, IntentType.MATHEMATICS],
        context_length=128000
    )
}