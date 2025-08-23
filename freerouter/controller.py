"""Main Controller class providing OpenAI-compatible interface."""

import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
import json

from .openrouter_client import OpenRouterClient, ChatCompletionRequest
from .semantic_router import SemanticRouter
from .models import RouterConfig, RouteDecision, IntentType, FREE_MODELS
from .exceptions import FreeRouterError, ModelUnavailableError, RateLimitError


class Controller:
    """Main FreeRouter controller with OpenAI-compatible interface."""
    
    def __init__(
        self,
        api_key: str,
        routing_model: str = "semantic",
        model_pool: str = "openrouter-free",
        precision_threshold: float = 0.3,
        **kwargs
    ):
        """Initialize the FreeRouter controller.
        
        Args:
            api_key: OpenRouter API key
            routing_model: Routing strategy ("semantic", "mf", "rf")
            model_pool: Model pool to use ("openrouter-free")
            precision_threshold: Routing precision threshold (0.1-0.9)
        """
        self.client = OpenRouterClient(api_key)
        self.semantic_router = SemanticRouter()
        self.config = RouterConfig(
            precision_threshold=precision_threshold,
            **kwargs
        )
        self.routing_model = routing_model
        self.model_pool = model_pool
        
        # Statistics tracking
        self.routing_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "fallback_routes": 0,
            "rate_limit_hits": 0,
            "intent_distribution": {intent.value: 0 for intent in IntentType}
        }
    
    async def chat_completions_create(
        self,
        model: Optional[str] = None,
        messages: List[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Union[Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """OpenAI-compatible chat completions endpoint."""
        if not messages:
            raise ValueError("Messages are required")
        
        self.routing_stats["total_requests"] += 1
        
        # Extract query for routing
        query = self._extract_query_from_messages(messages)
        
        # Route the request
        route_decision = await self._route_request(query, model)
        
        # Create request
        request = ChatCompletionRequest(
            model=route_decision.model_id,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            stop=stop
        )
        
        try:
            if stream:
                return self._stream_with_routing_info(request, route_decision)
            else:
                response = await self.client.chat_completion(
                    request, 
                    self.config.rate_limit_buffer
                )
                
                # Add routing information to response
                response["x_freerouter"] = {
                    "routed_model": route_decision.model_id,
                    "intent": route_decision.intent.value,
                    "confidence": route_decision.confidence,
                    "reasoning": route_decision.reasoning,
                    "fallbacks_available": len(route_decision.fallback_models)
                }
                
                self.routing_stats["successful_routes"] += 1
                return response
                
        except RateLimitError:
            # Try fallback models
            return await self._handle_fallback(request, route_decision)
    
    async def _route_request(
        self, 
        query: str, 
        preferred_model: Optional[str] = None
    ) -> RouteDecision:
        """Route a request to the best available model."""
        available_models = self.client.get_available_models(
            self.config.rate_limit_buffer
        )
        
        if not available_models:
            raise ModelUnavailableError("No models available due to rate limits")
        
        # If preferred model is available, use it
        if preferred_model and preferred_model in available_models:
            intent, confidence = self.semantic_router.classify_intent(
                query, 
                self.config.intent_detection_threshold
            )
            self.routing_stats["intent_distribution"][intent.value] += 1
            
            return RouteDecision(
                model_id=preferred_model,
                confidence=1.0,
                intent=intent,
                reasoning=f"User-specified model {preferred_model}",
                fallback_models=[m for m in available_models if m != preferred_model]
            )
        
        # Use semantic routing
        if self.routing_model == "semantic":
            model_id, intent, confidence = self.semantic_router.route_query(
                query, 
                available_models,
                self.config.precision_threshold
            )
            
            self.routing_stats["intent_distribution"][intent.value] += 1
            
            reasoning = self.semantic_router.explain_routing(
                query, model_id, intent, confidence
            )
            
            fallback_models = [m for m in available_models if m != model_id]
            
            return RouteDecision(
                model_id=model_id,
                confidence=confidence,
                intent=intent,
                reasoning=reasoning,
                fallback_models=fallback_models
            )
        
        # Fallback to first available model
        model_id = available_models[0]
        intent = IntentType.GENERAL
        
        return RouteDecision(
            model_id=model_id,
            confidence=0.5,
            intent=intent,
            reasoning=f"Default routing to {model_id}",
            fallback_models=available_models[1:]
        )
    
    async def _handle_fallback(
        self, 
        request: ChatCompletionRequest, 
        original_decision: RouteDecision
    ) -> Dict[str, Any]:
        """Handle fallback routing when primary model is rate limited."""
        self.routing_stats["rate_limit_hits"] += 1
        
        for fallback_model in original_decision.fallback_models[:self.config.max_fallbacks]:
            available_models = self.client.get_available_models(
                self.config.rate_limit_buffer
            )
            
            if fallback_model not in available_models:
                continue
            
            request.model = fallback_model
            try:
                response = await self.client.chat_completion(
                    request, 
                    self.config.rate_limit_buffer
                )
                
                response["x_freerouter"] = {
                    "routed_model": fallback_model,
                    "intent": original_decision.intent.value,
                    "confidence": original_decision.confidence * 0.8,  # Reduced confidence
                    "reasoning": f"Fallback to {fallback_model} due to rate limits",
                    "fallback_used": True
                }
                
                self.routing_stats["fallback_routes"] += 1
                return response
                
            except RateLimitError:
                continue
        
        raise ModelUnavailableError("All models are rate limited")
    
    async def _stream_with_routing_info(
        self, 
        request: ChatCompletionRequest, 
        route_decision: RouteDecision
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream response with routing information."""
        try:
            async for chunk in self.client.stream_chat_completion(
                request, 
                self.config.rate_limit_buffer
            ):
                # Add routing info to first chunk
                if "choices" in chunk and chunk["choices"]:
                    if not hasattr(self, '_routing_info_sent'):
                        chunk["x_freerouter"] = {
                            "routed_model": route_decision.model_id,
                            "intent": route_decision.intent.value,
                            "confidence": route_decision.confidence,
                            "reasoning": route_decision.reasoning
                        }
                        self._routing_info_sent = True
                
                yield chunk
                
            self.routing_stats["successful_routes"] += 1
            
        except RateLimitError:
            # Handle fallback for streaming
            fallback_request = request
            for fallback_model in route_decision.fallback_models[:self.config.max_fallbacks]:
                fallback_request.model = fallback_model
                try:
                    async for chunk in self.client.stream_chat_completion(
                        fallback_request, 
                        self.config.rate_limit_buffer
                    ):
                        if "choices" in chunk and chunk["choices"]:
                            if not hasattr(self, '_routing_info_sent'):
                                chunk["x_freerouter"] = {
                                    "routed_model": fallback_model,
                                    "intent": route_decision.intent.value,
                                    "confidence": route_decision.confidence * 0.8,
                                    "reasoning": f"Fallback to {fallback_model}",
                                    "fallback_used": True
                                }
                                self._routing_info_sent = True
                        
                        yield chunk
                    
                    self.routing_stats["fallback_routes"] += 1
                    return
                    
                except RateLimitError:
                    continue
            
            raise ModelUnavailableError("All models are rate limited")
    
    def _extract_query_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Extract the main query from conversation messages."""
        if not messages:
            return ""
        
        # Get the last user message
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if user_messages:
            return user_messages[-1].get("content", "")
        
        # Fallback to last message
        return messages[-1].get("content", "")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            **self.routing_stats,
            "available_models": self.client.get_available_models(),
            "success_rate": (
                self.routing_stats["successful_routes"] / 
                max(self.routing_stats["total_requests"], 1)
            )
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models in OpenAI format."""
        models = []
        for model_id, model_info in FREE_MODELS.items():
            models.append({
                "id": model_id,
                "object": "model",
                "owned_by": model_info.provider,
                "permission": [],
                "x_freerouter": {
                    "specialization": [intent.value for intent in model_info.intent_specialization],
                    "rate_limit_rpm": model_info.rate_limit_rpm,
                    "rate_limit_rpd": model_info.rate_limit_rpd,
                    "context_length": model_info.context_length,
                    "max_tokens": model_info.max_tokens
                }
            })
        return models
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()