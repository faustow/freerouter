"""
Pytest configuration and fixtures for FreeRouter tests.
"""

import asyncio
import pytest
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any, List

from freerouter.freerouter.config import Config
from freerouter.freerouter.client import OpenRouterClient, ModelInfo, ChatResponse
from freerouter.freerouter.models import ModelManager, ModelProfile
from freerouter.freerouter.analyzer import QueryAnalyzer


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = Config(
        openrouter_api_key="test_api_key",
        openrouter_base_url="https://api.test.com",
        model_profiles={
            "test/model-1": {
                "capabilities": {
                    "coding": 4,
                    "reasoning": 3,
                    "creative_writing": 2,
                    "analysis": 3,
                    "conversation": 4,
                    "math": 3
                },
                "specialties": ["coding", "conversation"],
                "context_window": 4096,
                "cost_tier": "free"
            },
            "test/model-2": {
                "capabilities": {
                    "coding": 2,
                    "reasoning": 4,
                    "creative_writing": 5,
                    "analysis": 4,
                    "conversation": 3,
                    "math": 2
                },
                "specialties": ["creative_writing", "analysis"],
                "context_window": 8192,
                "cost_tier": "free"
            }
        },
        query_types={
            "coding": {
                "keywords": ["code", "function", "python", "javascript"],
                "patterns": ["write.*function", "implement.*algorithm"],
                "preferred_models": ["test/model-1"]
            },
            "creative_writing": {
                "keywords": ["story", "poem", "creative", "write"],
                "patterns": ["write.*story", "create.*poem"],
                "preferred_models": ["test/model-2"]
            }
        },
        test_queries={
            "coding": ["Write a Python function", "Implement quicksort"],
            "creative_writing": ["Write a story", "Create a poem"],
            "math": ["Solve 2x + 5 = 15", "Calculate factorial"],
            "conversation": ["Hello", "How are you?"]
        }
    )
    return config


@pytest.fixture
def mock_model_info():
    """Create mock model information."""
    return ModelInfo(
        id="test/model-1",
        name="Test Model 1",
        description="A test model for unit testing",
        pricing={"prompt": "0", "completion": "0"},
        context_length=4096,
        architecture={"tokenizer": "test"},
        top_provider={"name": "test_provider"}
    )


@pytest.fixture
def mock_chat_response():
    """Create mock chat response."""
    return ChatResponse(
        id="test_response_id",
        model="test/model-1",
        choices=[{
            "message": {
                "role": "assistant",
                "content": "This is a test response from the model."
            },
            "finish_reason": "stop"
        }],
        usage={"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
        created=1234567890
    )


@pytest.fixture
def mock_openrouter_client(mock_config, mock_model_info, mock_chat_response):
    """Create a mock OpenRouter client."""
    client = Mock(spec=OpenRouterClient)
    client.config = mock_config
    
    # Mock async methods
    client.get_models = AsyncMock(return_value=[mock_model_info])
    client.get_free_models = AsyncMock(return_value=[mock_model_info])
    client.chat_completion = AsyncMock(return_value=mock_chat_response)
    client.test_model = AsyncMock(return_value=True)
    client.get_model_info = AsyncMock(return_value=mock_model_info)
    client.close = AsyncMock()
    
    # Mock context manager
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    
    return client


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return {
        "coding": [
            "Write a Python function to calculate factorial",
            "How do I implement a binary search algorithm?",
            "Debug this JavaScript code",
            "Create a REST API endpoint"
        ],
        "creative_writing": [
            "Write a short story about time travel",
            "Create a poem about the ocean",
            "Describe a magical forest",
            "Write dialogue between two characters"
        ],
        "math": [
            "Solve: 2x + 5 = 15",
            "Calculate the area of a circle with radius 7",
            "What is 15% of 240?",
            "Find the derivative of x^2 + 3x + 2"
        ],
        "analysis": [
            "Compare renewable energy sources",
            "Analyze the pros and cons of remote work",
            "Evaluate different programming languages",
            "Compare electric vs gas vehicles"
        ],
        "conversation": [
            "Hello, how are you?",
            "What's the weather like?",
            "Can you help me?",
            "Tell me about yourself"
        ],
        "reasoning": [
            "Why can't penguins fly?",
            "Explain the paradox of omnipotence",
            "Why are manhole covers round?",
            "How do you know if you're in a simulation?"
        ]
    }


@pytest.fixture
def mock_model_profiles():
    """Create mock model profiles for testing."""
    from freerouter.freerouter.models import ModelProfile, PerformanceMetrics, CapabilityScore
    from datetime import datetime
    
    profiles = {}
    
    # Model 1: Good at coding
    profile1 = ModelProfile(model_id="test/model-1")
    profile1.capabilities = {
        "coding": CapabilityScore(score=4.5, confidence=0.9, sample_count=10),
        "reasoning": CapabilityScore(score=3.5, confidence=0.8, sample_count=8),
        "conversation": CapabilityScore(score=4.0, confidence=0.7, sample_count=12)
    }
    profile1.performance = PerformanceMetrics(
        average_response_time=2.5,
        success_rate=0.95,
        total_requests=50,
        successful_requests=47,
        failed_requests=3
    )
    profiles["test/model-1"] = profile1
    
    # Model 2: Good at creative writing
    profile2 = ModelProfile(model_id="test/model-2")
    profile2.capabilities = {
        "creative_writing": CapabilityScore(score=4.8, confidence=0.9, sample_count=15),
        "analysis": CapabilityScore(score=4.2, confidence=0.8, sample_count=10),
        "conversation": CapabilityScore(score=3.8, confidence=0.7, sample_count=8)
    }
    profile2.performance = PerformanceMetrics(
        average_response_time=3.2,
        success_rate=0.92,
        total_requests=40,
        successful_requests=37,
        failed_requests=3
    )
    profiles["test/model-2"] = profile2
    
    return profiles


@pytest.fixture
async def mock_model_manager(mock_config, mock_openrouter_client, mock_model_profiles):
    """Create a mock model manager."""
    manager = Mock(spec=ModelManager)
    manager.config = mock_config
    manager.client = mock_openrouter_client
    manager.profiles = mock_model_profiles
    
    # Mock async methods
    manager.start = AsyncMock()
    manager.stop = AsyncMock()
    manager.discover_models = AsyncMock(return_value=list(mock_model_profiles.keys()))
    manager.test_model = AsyncMock(return_value=True)
    manager.get_available_models = Mock(return_value=list(mock_model_profiles.keys()))
    manager.get_model_profile = Mock(side_effect=lambda model_id: mock_model_profiles.get(model_id))
    manager.get_best_models = Mock(return_value=["test/model-1", "test/model-2"])
    
    # Mock context manager
    manager.__aenter__ = AsyncMock(return_value=manager)
    manager.__aexit__ = AsyncMock(return_value=None)
    
    return manager


@pytest.fixture
def mock_query_analyzer(mock_config):
    """Create a mock query analyzer."""
    analyzer = QueryAnalyzer(mock_config)
    return analyzer


# Test data fixtures

@pytest.fixture
def evaluation_test_data():
    """Test data for evaluation tests."""
    return {
        "queries": [
            ("Write a Python function", "coding"),
            ("Tell me a story", "creative_writing"),
            ("What is 2+2?", "math"),
            ("Hello there", "conversation")
        ],
        "expected_scores": {
            "accuracy": 3.5,
            "relevance": 4.0,
            "clarity": 3.8,
            "completeness": 3.2,
            "efficiency": 4.2
        }
    }


@pytest.fixture
def routing_test_scenarios():
    """Test scenarios for routing tests."""
    return [
        {
            "query": "Write a Python function to sort a list",
            "expected_type": "coding",
            "expected_model": "test/model-1",
            "confidence_threshold": 0.7
        },
        {
            "query": "Create a beautiful poem about nature",
            "expected_type": "creative_writing", 
            "expected_model": "test/model-2",
            "confidence_threshold": 0.8
        },
        {
            "query": "Hello, how are you today?",
            "expected_type": "conversation",
            "expected_model": None,  # Could be either model
            "confidence_threshold": 0.5
        }
    ]