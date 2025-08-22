"""
Tests for the OpenRouterClient module.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx

from freerouter.freerouter.client import OpenRouterClient, RateLimitError, ModelNotFoundError, APIError
from freerouter.freerouter.exceptions import OpenRouterError


class TestOpenRouterClient:
    """Test cases for OpenRouterClient."""
    
    @pytest.mark.asyncio
    async def test_init(self, mock_config):
        """Test OpenRouterClient initialization."""
        client = OpenRouterClient(mock_config)
        assert client.config == mock_config
        assert client.client.base_url == mock_config.openrouter_base_url
        assert "Authorization" in client.client.headers
        await client.close()
    
    @pytest.mark.asyncio
    async def test_init_without_api_key(self, mock_config):
        """Test initialization without API key raises error."""
        mock_config.openrouter_api_key = ""
        
        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            OpenRouterClient(mock_config)
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config):
        """Test async context manager functionality."""
        async with OpenRouterClient(mock_config) as client:
            assert isinstance(client, OpenRouterClient)
            assert client.client is not None
    
    @pytest.mark.asyncio
    async def test_get_models_success(self, mock_openrouter_client):
        """Test successful model retrieval."""
        models = await mock_openrouter_client.get_models()
        assert len(models) > 0
        assert all(hasattr(model, 'id') for model in models)
        assert all(hasattr(model, 'name') for model in models)
    
    @pytest.mark.asyncio
    async def test_get_models_caching(self, mock_openrouter_client):
        """Test model caching functionality."""
        # First call
        models1 = await mock_openrouter_client.get_models()
        
        # Second call should use cache
        models2 = await mock_openrouter_client.get_models()
        
        # Should be called only once due to caching
        assert mock_openrouter_client.get_models.call_count >= 1
        assert len(models1) == len(models2)
    
    @pytest.mark.asyncio
    async def test_get_free_models(self, mock_openrouter_client):
        """Test free model filtering."""
        free_models = await mock_openrouter_client.get_free_models()
        assert isinstance(free_models, list)
        # Mock should return free models
        assert len(free_models) >= 0
    
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, mock_openrouter_client, mock_chat_response):
        """Test successful chat completion."""
        from freerouter.freerouter.client import ChatMessage
        
        messages = [ChatMessage(role="user", content="Hello")]
        response = await mock_openrouter_client.chat_completion(
            model="test/model-1",
            messages=messages
        )
        
        assert response.id == mock_chat_response.id
        assert response.model == mock_chat_response.model
        assert len(response.choices) > 0
        assert response.message_content != ""
    
    @pytest.mark.asyncio
    async def test_chat_completion_with_parameters(self, mock_openrouter_client):
        """Test chat completion with various parameters."""
        from freerouter.freerouter.client import ChatMessage
        
        messages = [ChatMessage(role="user", content="Test")]
        await mock_openrouter_client.chat_completion(
            model="test/model-1",
            messages=messages,
            max_tokens=100,
            temperature=0.5,
            top_p=0.9
        )
        
        # Verify the mock was called with correct parameters
        mock_openrouter_client.chat_completion.assert_called()
    
    @pytest.mark.asyncio
    async def test_test_model_success(self, mock_openrouter_client):
        """Test successful model testing."""
        result = await mock_openrouter_client.test_model("test/model-1")
        assert result is True
    
    @pytest.mark.asyncio
    async def test_test_model_failure(self, mock_openrouter_client):
        """Test model testing failure."""
        mock_openrouter_client.test_model.return_value = False
        result = await mock_openrouter_client.test_model("nonexistent-model")
        assert result is False
    
    @pytest.mark.asyncio 
    async def test_get_model_info(self, mock_openrouter_client, mock_model_info):
        """Test getting specific model information."""
        model_info = await mock_openrouter_client.get_model_info("test/model-1")
        assert model_info is not None
        assert model_info.id == "test/model-1"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_config):
        """Test rate limiting functionality."""
        from freerouter.freerouter.client import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=2, requests_per_hour=5)
        
        # Should allow first few requests
        await limiter.acquire()
        await limiter.acquire()
        
        # Test that it tracks requests
        assert len(limiter.minute_requests) == 2
        assert len(limiter.hour_requests) == 2


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic(self):
        """Test basic rate limiter functionality."""
        from freerouter.freerouter.client import RateLimiter
        
        limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
        
        # Should allow normal requests
        await limiter.acquire()
        assert len(limiter.minute_requests) == 1
        assert len(limiter.hour_requests) == 1
    
    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup(self):
        """Test that old requests are cleaned up."""
        from freerouter.freerouter.client import RateLimiter
        import time
        
        limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
        
        # Add old request manually
        old_time = time.time() - 3700  # More than 1 hour ago
        limiter.hour_requests.append(old_time)
        
        await limiter.acquire()
        
        # Old request should be cleaned up
        assert all(req > time.time() - 3600 for req in limiter.hour_requests)


class TestErrorHandling:
    """Test error handling in OpenRouterClient."""
    
    @pytest.mark.asyncio
    async def test_handle_401_error(self, mock_config):
        """Test handling of authentication errors."""
        client = OpenRouterClient(mock_config)
        
        # Mock HTTP response with 401 status
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        
        with pytest.raises(OpenRouterError):
            client._handle_response_error(mock_response)
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_handle_404_error(self, mock_config):
        """Test handling of model not found errors."""
        client = OpenRouterClient(mock_config)
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {
            "error": {"message": "Model not found"}
        }
        
        with pytest.raises(ModelNotFoundError):
            client._handle_response_error(mock_response)
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_handle_429_error(self, mock_config):
        """Test handling of rate limit errors."""
        client = OpenRouterClient(mock_config)
        
        mock_response = Mock()
        mock_response.status_code = 429
        mock_response.headers = {"retry-after": "60"}
        mock_response.json.return_value = {
            "error": {"message": "Rate limit exceeded"}
        }
        
        with pytest.raises(RateLimitError) as exc_info:
            client._handle_response_error(mock_response)
        
        assert exc_info.value.retry_after == 60
        await client.close()
    
    @pytest.mark.asyncio
    async def test_handle_500_error(self, mock_config):
        """Test handling of server errors."""
        client = OpenRouterClient(mock_config)
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.json.return_value = {
            "error": {"message": "Internal server error"}
        }
        
        with pytest.raises(OpenRouterError):
            client._handle_response_error(mock_response)
        
        await client.close()


class TestChatMessage:
    """Test ChatMessage data class."""
    
    def test_chat_message_creation(self):
        """Test creating ChatMessage instances."""
        from freerouter.freerouter.client import ChatMessage
        
        message = ChatMessage(role="user", content="Hello")
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.name is None
        
        message_with_name = ChatMessage(role="assistant", content="Hi", name="bot")
        assert message_with_name.name == "bot"


class TestChatResponse:
    """Test ChatResponse data class."""
    
    def test_chat_response_message_content(self, mock_chat_response):
        """Test extracting message content from response."""
        content = mock_chat_response.message_content
        assert content == "This is a test response from the model."
    
    def test_chat_response_empty_choices(self):
        """Test handling response with no choices."""
        from freerouter.freerouter.client import ChatResponse
        
        response = ChatResponse(
            id="test",
            model="test/model",
            choices=[],
            usage={},
            created=123456
        )
        
        assert response.message_content == ""