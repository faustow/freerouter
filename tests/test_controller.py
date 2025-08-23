"""Tests for the main Controller class."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from freerouter.controller import Controller
from freerouter.exceptions import ModelUnavailableError, RateLimitError
from freerouter.models import IntentType


class TestController:
    @pytest.fixture
    def mock_client(self):
        """Mock OpenRouter client."""
        client = Mock()
        client.get_available_models = Mock(return_value=[
            "deepseek/deepseek-r1",
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3-8b-instruct:free"
        ])
        client.chat_completion = AsyncMock(return_value={
            "choices": [{"message": {"content": "Test response"}}],
            "usage": {"total_tokens": 50}
        })
        client.close = AsyncMock()
        return client
    
    @pytest.fixture
    def controller(self, mock_client):
        """Controller with mocked client."""
        with patch('freerouter.controller.OpenRouterClient') as mock_class:
            mock_class.return_value = mock_client
            controller = Controller(api_key="test-key")
            return controller
    
    @pytest.mark.asyncio
    async def test_basic_chat_completion(self, controller, mock_client):
        """Test basic chat completion functionality."""
        messages = [{"role": "user", "content": "Hello, world!"}]
        
        response = await controller.chat_completions_create(messages=messages)
        
        # Should have called the client
        mock_client.chat_completion.assert_called_once()
        
        # Should have routing information
        assert "x_freerouter" in response
        assert "routed_model" in response["x_freerouter"]
        assert "intent" in response["x_freerouter"]
        assert "confidence" in response["x_freerouter"]
    
    @pytest.mark.asyncio
    async def test_preferred_model_routing(self, controller, mock_client):
        """Test routing to a preferred model."""
        messages = [{"role": "user", "content": "Write some code"}]
        preferred_model = "deepseek/deepseek-r1"
        
        response = await controller.chat_completions_create(
            model=preferred_model,
            messages=messages
        )
        
        # Should route to the preferred model
        assert response["x_freerouter"]["routed_model"] == preferred_model
    
    @pytest.mark.asyncio
    async def test_semantic_routing(self, controller, mock_client):
        """Test that semantic routing works correctly."""
        # Test coding query
        messages = [{"role": "user", "content": "def fibonacci(n):"}]
        
        response = await controller.chat_completions_create(messages=messages)
        
        # Should detect coding intent
        assert response["x_freerouter"]["intent"] == "coding"
        
        # Should route to a coding-specialized model
        routed_model = response["x_freerouter"]["routed_model"]
        assert "deepseek" in routed_model.lower() or "phi" in routed_model.lower()
    
    @pytest.mark.asyncio
    async def test_fallback_on_rate_limit(self, controller, mock_client):
        """Test fallback routing when primary model is rate limited."""
        messages = [{"role": "user", "content": "Hello"}]
        
        # Mock rate limit error then success
        mock_client.chat_completion.side_effect = [
            RateLimitError("Rate limited"),
            {
                "choices": [{"message": {"content": "Fallback response"}}],
                "usage": {"total_tokens": 30}
            }
        ]
        
        response = await controller.chat_completions_create(messages=messages)
        
        # Should have used fallback
        assert response["x_freerouter"]["fallback_used"] is True
        assert mock_client.chat_completion.call_count == 2
    
    @pytest.mark.asyncio
    async def test_no_models_available(self, controller, mock_client):
        """Test error when no models are available."""
        mock_client.get_available_models.return_value = []
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(ModelUnavailableError):
            await controller.chat_completions_create(messages=messages)
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, controller, mock_client):
        """Test streaming chat completion."""
        messages = [{"role": "user", "content": "Tell me a story"}]
        
        # Mock streaming response
        mock_chunks = [
            {"choices": [{"delta": {"content": "Once"}}]},
            {"choices": [{"delta": {"content": " upon"}}]},
            {"choices": [{"delta": {"content": " a time"}}]}
        ]
        
        async def mock_stream(*args, **kwargs):
            for chunk in mock_chunks:
                yield chunk
        
        mock_client.stream_chat_completion = mock_stream
        
        chunks = []
        async for chunk in controller.chat_completions_create(
            messages=messages, 
            stream=True
        ):
            chunks.append(chunk)
        
        assert len(chunks) == 3
        # First chunk should have routing info
        if "x_freerouter" in chunks[0]:
            assert "routed_model" in chunks[0]["x_freerouter"]
    
    def test_extract_query_from_messages(self, controller):
        """Test query extraction from message history."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First response"},
            {"role": "user", "content": "Second question"}
        ]
        
        query = controller._extract_query_from_messages(messages)
        assert query == "Second question"
    
    def test_empty_messages_handling(self, controller):
        """Test handling of empty messages."""
        query = controller._extract_query_from_messages([])
        assert query == ""
        
        query = controller._extract_query_from_messages([
            {"role": "system", "content": "System message only"}
        ])
        assert query == "System message only"
    
    def test_statistics_tracking(self, controller):
        """Test that statistics are tracked correctly."""
        stats = controller.get_stats()
        
        assert "total_requests" in stats
        assert "successful_routes" in stats
        assert "fallback_routes" in stats
        assert "rate_limit_hits" in stats
        assert "intent_distribution" in stats
        assert "success_rate" in stats
        
        # Should have entries for all intents
        for intent in IntentType:
            assert intent.value in stats["intent_distribution"]
    
    def test_list_models(self, controller):
        """Test listing available models."""
        models = controller.list_models()
        
        assert len(models) > 0
        
        # Each model should have required fields
        for model in models:
            assert "id" in model
            assert "object" in model
            assert "owned_by" in model
            assert "x_freerouter" in model
            
            freerouter_info = model["x_freerouter"]
            assert "specialization" in freerouter_info
            assert "rate_limit_rpm" in freerouter_info
            assert "rate_limit_rpd" in freerouter_info
            assert "context_length" in freerouter_info
    
    @pytest.mark.asyncio
    async def test_close_cleanup(self, controller, mock_client):
        """Test proper cleanup on close."""
        await controller.close()
        mock_client.close.assert_called_once()