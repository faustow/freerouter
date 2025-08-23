"""Pytest configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import Mock


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_api_key():
    """Sample API key for testing."""
    return "sk-or-test-key-1234567890abcdef"


@pytest.fixture
def sample_messages():
    """Sample conversation messages."""
    return [
        {"role": "user", "content": "Write a Python function to calculate the factorial of a number"}
    ]


@pytest.fixture
def sample_openrouter_response():
    """Sample OpenRouter API response."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1699999999,
        "model": "deepseek/deepseek-r1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 35,
            "total_tokens": 55
        }
    }