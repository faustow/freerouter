"""
OpenRouter API client with error handling and rate limiting.

This module provides a robust client for interacting with the OpenRouter API,
including automatic retries, rate limiting, and comprehensive error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .config import Config, get_config
from .exceptions import (
    OpenRouterError, 
    RateLimitError, 
    ModelNotFoundError, 
    APIError
)

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an available model."""
    id: str
    name: str
    description: str
    pricing: Dict[str, Any]
    context_length: int
    architecture: Dict[str, Any] = field(default_factory=dict)
    top_provider: Dict[str, Any] = field(default_factory=dict)
    per_request_limits: Optional[Dict[str, Any]] = None


@dataclass 
class ChatMessage:
    """A chat message in the conversation."""
    role: str  # 'user', 'assistant', or 'system'
    content: str
    name: Optional[str] = None


@dataclass
class ChatResponse:
    """Response from a chat completion request."""
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, Any]
    created: int
    
    @property
    def message_content(self) -> str:
        """Get the content of the first choice."""
        if self.choices:
            return self.choices[0].get('message', {}).get('content', '')
        return ''


class RateLimiter:
    """Rate limiter for API requests."""
    
    def __init__(self, requests_per_minute: int = 20, requests_per_hour: int = 200):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests: List[float] = []
        self.hour_requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            now = time.time()
            
            # Clean old requests
            cutoff_minute = now - 60
            cutoff_hour = now - 3600
            
            self.minute_requests = [req for req in self.minute_requests if req > cutoff_minute]
            self.hour_requests = [req for req in self.hour_requests if req > cutoff_hour]
            
            # Check limits
            if len(self.minute_requests) >= self.requests_per_minute:
                sleep_time = 60 - (now - self.minute_requests[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.1f} seconds")
                    await asyncio.sleep(sleep_time)
            
            if len(self.hour_requests) >= self.requests_per_hour:
                sleep_time = 3600 - (now - self.hour_requests[0])
                if sleep_time > 0:
                    logger.info(f"Hourly rate limit reached, sleeping for {sleep_time:.1f} seconds")
                    await asyncio.sleep(sleep_time)
            
            # Record this request
            current_time = time.time()
            self.minute_requests.append(current_time)
            self.hour_requests.append(current_time)


class OpenRouterClient:
    """
    Asynchronous client for the OpenRouter API.
    
    Provides methods for interacting with OpenRouter's chat completions API
    with built-in rate limiting, error handling, and retry logic.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the OpenRouter client.
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        
        if not self.config.openrouter_api_key:
            raise ValueError("OpenRouter API key is required")
        
        # Set up HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.config.openrouter_base_url,
            headers={
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "Content-Type": "application/json",
                "User-Agent": "FreeRouter/0.1.0"
            },
            timeout=self.config.default_config.timeout,
        )
        
        # Set up rate limiter
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.default_config.rate_limit.requests_per_minute,
            requests_per_hour=self.config.default_config.rate_limit.requests_per_hour,
        )
        
        # Cache for model information
        self._models_cache: Optional[List[ModelInfo]] = None
        self._models_cache_time: Optional[datetime] = None
        self._cache_ttl = timedelta(hours=1)
        
        logger.info("OpenRouter client initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
        logger.debug("OpenRouter client closed")
    
    def _handle_response_error(self, response: httpx.Response) -> None:
        """Handle HTTP response errors."""
        if response.status_code == 200:
            return
        
        try:
            error_data = response.json()
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
        except (json.JSONDecodeError, KeyError):
            error_message = f"HTTP {response.status_code}: {response.text}"
        
        if response.status_code == 401:
            raise OpenRouterError(f"Authentication failed: {error_message}")
        elif response.status_code == 403:
            raise OpenRouterError(f"Access forbidden: {error_message}")
        elif response.status_code == 429:
            retry_after = response.headers.get('retry-after')
            retry_after_int = int(retry_after) if retry_after else None
            raise RateLimitError(error_message, retry_after_int)
        elif response.status_code == 404:
            # Extract model ID from error message if possible, otherwise use "unknown"
            model_id = error_message.lower().replace("model not found:", "").strip() or "unknown"
            raise ModelNotFoundError(model_id)
        elif response.status_code >= 500:
            raise OpenRouterError(f"Server error: {error_message}")
        else:
            raise APIError(error_message, status_code=response.status_code)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, RateLimitError)),
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any
    ) -> httpx.Response:
        """Make an HTTP request with rate limiting and retry logic."""
        await self.rate_limiter.acquire()
        
        try:
            response = await self.client.request(method, endpoint, **kwargs)
            self._handle_response_error(response)
            return response
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise
        except RateLimitError as e:
            if e.retry_after:
                logger.info(f"Rate limited, retrying after {e.retry_after} seconds")
                await asyncio.sleep(e.retry_after)
            raise
    
    async def get_models(self, force_refresh: bool = False) -> List[ModelInfo]:
        """
        Get list of available models.
        
        Args:
            force_refresh: Force refresh of cached models
            
        Returns:
            List of available models
        """
        # Check cache
        if (not force_refresh and 
            self._models_cache and 
            self._models_cache_time and
            datetime.now() - self._models_cache_time < self._cache_ttl):
            logger.debug("Returning cached models list")
            return self._models_cache
        
        logger.info("Fetching models from OpenRouter API")
        
        try:
            response = await self._make_request("GET", "/models")
            data = response.json()
            
            models = []
            for model_data in data.get('data', []):
                try:
                    model = ModelInfo(
                        id=model_data['id'],
                        name=model_data.get('name', model_data['id']),
                        description=model_data.get('description', ''),
                        pricing=model_data.get('pricing', {}),
                        context_length=model_data.get('context_length', 4096),
                        architecture=model_data.get('architecture', {}),
                        top_provider=model_data.get('top_provider', {}),
                        per_request_limits=model_data.get('per_request_limits'),
                    )
                    models.append(model)
                except KeyError as e:
                    logger.warning(f"Skipping malformed model data: missing {e}")
            
            # Update cache
            self._models_cache = models
            self._models_cache_time = datetime.now()
            
            logger.info(f"Retrieved {len(models)} models")
            return models
            
        except Exception as e:
            logger.error(f"Error fetching models: {e}")
            # Return cached models if available, otherwise empty list
            return self._models_cache or []
    
    async def get_free_models(self) -> List[ModelInfo]:
        """Get list of free models only."""
        all_models = await self.get_models()
        free_models = []
        
        for model in all_models:
            # Check if model is free (pricing has 0 cost or no cost data)
            pricing = model.pricing
            if not pricing:
                continue
                
            prompt_cost = float(pricing.get('prompt', '0'))
            completion_cost = float(pricing.get('completion', '0'))
            
            if prompt_cost == 0 and completion_cost == 0:
                free_models.append(model)
        
        logger.info(f"Found {len(free_models)} free models")
        return free_models
    
    async def chat_completion(
        self,
        model: str,
        messages: List[Union[ChatMessage, Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 1.0,
        stream: bool = False,
        **kwargs: Any
    ) -> ChatResponse:
        """
        Create a chat completion.
        
        Args:
            model: Model to use for completion
            messages: List of messages in the conversation
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Chat completion response
        """
        # Convert messages to dict format if needed
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                msg_dict = {"role": msg.role, "content": msg.content}
                if msg.name:
                    msg_dict["name"] = msg.name
                formatted_messages.append(msg_dict)
            else:
                formatted_messages.append(msg)
        
        payload = {
            "model": model,
            "messages": formatted_messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        logger.debug(f"Making chat completion request to model: {model}")
        
        try:
            response = await self._make_request("POST", "/chat/completions", json=payload)
            data = response.json()
            
            chat_response = ChatResponse(
                id=data['id'],
                model=data['model'],
                choices=data['choices'],
                usage=data.get('usage', {}),
                created=data['created']
            )
            
            logger.debug(f"Chat completion successful, usage: {chat_response.usage}")
            return chat_response
            
        except Exception as e:
            logger.error(f"Chat completion error with model {model}: {e}")
            raise
    
    async def test_model(self, model: str, test_prompt: str = "Hello, how are you?") -> bool:
        """
        Test if a model is working and responsive.
        
        Args:
            model: Model to test
            test_prompt: Test prompt to send
            
        Returns:
            True if model responds successfully, False otherwise
        """
        try:
            messages = [ChatMessage(role="user", content=test_prompt)]
            response = await self.chat_completion(
                model=model,
                messages=messages,
                max_tokens=50,
                temperature=0.1
            )
            
            return bool(response.message_content.strip())
            
        except Exception as e:
            logger.warning(f"Model {model} test failed: {e}")
            return False
    
    async def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a specific model."""
        models = await self.get_models()
        for model in models:
            if model.id == model_id:
                return model
        return None