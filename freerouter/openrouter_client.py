"""OpenRouter API client with rate limiting and error handling."""

import asyncio
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass, field
from collections import defaultdict, deque

import httpx
from pydantic import BaseModel

from .exceptions import RateLimitError, ModelUnavailableError, FreeRouterError
from .models import ModelInfo, FREE_MODELS


@dataclass
class RateLimitTracker:
    """Track rate limits for a model."""
    requests_per_minute: deque = field(default_factory=deque)
    requests_per_day: deque = field(default_factory=deque)
    last_request: float = 0
    
    def can_make_request(self, model: ModelInfo, buffer: float = 0.1) -> bool:
        """Check if we can make a request without hitting rate limits."""
        now = time.time()
        
        # Clean old requests
        cutoff_minute = now - 60
        while self.requests_per_minute and self.requests_per_minute[0] < cutoff_minute:
            self.requests_per_minute.popleft()
            
        cutoff_day = now - 86400  # 24 hours
        while self.requests_per_day and self.requests_per_day[0] < cutoff_day:
            self.requests_per_day.popleft()
        
        # Check limits with buffer
        rpm_limit = int(model.rate_limit_rpm * (1 - buffer))
        rpd_limit = int(model.rate_limit_rpd * (1 - buffer))
        
        return (len(self.requests_per_minute) < rpm_limit and 
                len(self.requests_per_day) < rpd_limit)
    
    def record_request(self) -> None:
        """Record a new request."""
        now = time.time()
        self.requests_per_minute.append(now)
        self.requests_per_day.append(now)
        self.last_request = now


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    stop: Optional[List[str]] = None


class OpenRouterClient:
    """Client for OpenRouter API with rate limiting."""
    
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.rate_trackers: Dict[str, RateLimitTracker] = defaultdict(RateLimitTracker)
        self.client = httpx.AsyncClient(
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/freerouter/freerouter",
                "X-Title": "FreeRouter"
            },
            timeout=httpx.Timeout(60.0)
        )
    
    async def chat_completion(
        self, 
        request: ChatCompletionRequest,
        rate_limit_buffer: float = 0.1
    ) -> Dict[str, Any]:
        """Make a chat completion request with rate limiting."""
        model = FREE_MODELS.get(request.model)
        if not model:
            raise ModelUnavailableError(f"Model {request.model} not found in free pool")
        
        tracker = self.rate_trackers[request.model]
        if not tracker.can_make_request(model, rate_limit_buffer):
            raise RateLimitError(f"Rate limit exceeded for model {request.model}")
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": request.model,
                    "messages": request.messages,
                    "max_tokens": request.max_tokens or model.max_tokens,
                    "temperature": request.temperature,
                    "stream": request.stream,
                    "stop": request.stop
                }
            )
            
            if response.status_code == 429:
                raise RateLimitError(f"API rate limit hit for model {request.model}")
            elif response.status_code >= 400:
                raise FreeRouterError(f"API error {response.status_code}: {response.text}")
            
            tracker.record_request()
            return response.json()
            
        except httpx.RequestError as e:
            raise FreeRouterError(f"Request failed: {str(e)}")
    
    async def stream_chat_completion(
        self, 
        request: ChatCompletionRequest,
        rate_limit_buffer: float = 0.1
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream a chat completion request."""
        model = FREE_MODELS.get(request.model)
        if not model:
            raise ModelUnavailableError(f"Model {request.model} not found in free pool")
        
        tracker = self.rate_trackers[request.model]
        if not tracker.can_make_request(model, rate_limit_buffer):
            raise RateLimitError(f"Rate limit exceeded for model {request.model}")
        
        request.stream = True
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json={
                    "model": request.model,
                    "messages": request.messages,
                    "max_tokens": request.max_tokens or model.max_tokens,
                    "temperature": request.temperature,
                    "stream": True,
                    "stop": request.stop
                }
            ) as response:
                if response.status_code == 429:
                    raise RateLimitError(f"API rate limit hit for model {request.model}")
                elif response.status_code >= 400:
                    raise FreeRouterError(f"API error {response.status_code}")
                
                tracker.record_request()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]  # Remove "data: " prefix
                        if data == "[DONE]":
                            break
                        try:
                            yield eval(data)  # Parse JSON chunk
                        except:
                            continue
                            
        except httpx.RequestError as e:
            raise FreeRouterError(f"Stream request failed: {str(e)}")
    
    def get_available_models(self, rate_limit_buffer: float = 0.1) -> List[str]:
        """Get list of models that are currently available (not rate limited)."""
        available = []
        for model_id, model_info in FREE_MODELS.items():
            tracker = self.rate_trackers[model_id]
            if tracker.can_make_request(model_info, rate_limit_buffer):
                available.append(model_id)
        return available
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()