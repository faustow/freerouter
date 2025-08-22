"""
Core routing logic for intelligent model selection.

This module implements the main FreeRouter class that combines query analysis,
model management, and routing logic to automatically select the best model
for user queries.
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime

from .client import OpenRouterClient, ChatMessage, ChatResponse
from .analyzer import QueryAnalyzer, QueryAnalysis, QueryType, QueryComplexity
from .models import ModelManager, ModelProfile, ModelStatus
from .config import Config, get_config

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """Represents a routing decision with explanation."""
    selected_model: str
    confidence: float
    reasoning: str
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    query_analysis: Optional[QueryAnalysis] = None
    fallback_used: bool = False


@dataclass
class RouteResponse:
    """Response from a routed query."""
    response: ChatResponse
    routing_decision: RoutingDecision
    execution_time: float
    model_used: str


class RoutingStrategy:
    """Base class for routing strategies."""
    
    def select_model(
        self,
        query_analysis: QueryAnalysis,
        available_models: List[str],
        model_manager: ModelManager
    ) -> RoutingDecision:
        """Select the best model for a query."""
        raise NotImplementedError


class CapabilityBasedStrategy(RoutingStrategy):
    """Routing strategy based on model capabilities."""
    
    def select_model(
        self,
        query_analysis: QueryAnalysis,
        available_models: List[str],
        model_manager: ModelManager
    ) -> RoutingDecision:
        """Select model based on capability scores."""
        if not available_models:
            return RoutingDecision(
                selected_model="",
                confidence=0.0,
                reasoning="No available models"
            )
        
        # Get capability name for primary query type
        capability_map = {
            QueryType.CODING: "coding",
            QueryType.CREATIVE_WRITING: "creative_writing",
            QueryType.ANALYSIS: "analysis",
            QueryType.MATH: "math",
            QueryType.REASONING: "reasoning",
            QueryType.CONVERSATION: "conversation"
        }
        
        primary_capability = capability_map.get(query_analysis.primary_type, "conversation")
        
        # Score each available model
        model_scores = []
        
        for model_id in available_models:
            profile = model_manager.get_model_profile(model_id)
            if not profile or not profile.is_healthy():
                continue
            
            # Base score from primary capability
            primary_score = profile.get_capability_score(primary_capability)
            total_score = primary_score * 1.0
            
            # Add secondary capability scores
            for secondary_type in query_analysis.secondary_types:
                secondary_capability = capability_map.get(secondary_type, "conversation")
                secondary_score = profile.get_capability_score(secondary_capability)
                total_score += secondary_score * 0.3
            
            # Performance adjustments
            perf = profile.performance
            if perf.total_requests > 5:
                # Boost for good success rate
                if perf.success_rate > 0.9:
                    total_score *= 1.1
                elif perf.success_rate < 0.8:
                    total_score *= 0.9
                
                # Boost for fast response time
                if perf.average_response_time < 2.0:
                    total_score *= 1.05
                elif perf.average_response_time > 10.0:
                    total_score *= 0.95
            
            # Complexity adjustments
            if query_analysis.complexity == QueryComplexity.COMPLEX:
                # Prefer models with higher context windows for complex queries
                if profile.context_window > 8192:
                    total_score *= 1.1
            elif query_analysis.complexity == QueryComplexity.SIMPLE:
                # For simple queries, prefer faster models
                if perf.average_response_time < 3.0:
                    total_score *= 1.05
            
            model_scores.append((model_id, total_score))
        
        if not model_scores:
            # Fallback to first available model
            return RoutingDecision(
                selected_model=available_models[0],
                confidence=0.1,
                reasoning="Fallback: no model profiles available",
                fallback_used=True
            )
        
        # Sort by score
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected_model, best_score = model_scores[0]
        alternatives = model_scores[1:4]  # Top 3 alternatives
        
        # Calculate confidence based on score gap
        if len(model_scores) > 1:
            score_gap = best_score - model_scores[1][1]
            confidence = min(0.9, 0.5 + (score_gap / best_score) * 0.4)
        else:
            confidence = 0.8
        
        # Generate reasoning
        profile = model_manager.get_model_profile(selected_model)
        reasoning_parts = [
            f"Selected for {primary_capability} capability (score: {profile.get_capability_score(primary_capability):.1f})"
        ]
        
        if query_analysis.secondary_types:
            secondary_caps = [capability_map.get(t, "conversation") for t in query_analysis.secondary_types]
            reasoning_parts.append(f"Also handles: {', '.join(secondary_caps)}")
        
        if profile.performance.total_requests > 5:
            reasoning_parts.append(f"Success rate: {profile.performance.success_rate:.1%}")
        
        reasoning = ". ".join(reasoning_parts)
        
        return RoutingDecision(
            selected_model=selected_model,
            confidence=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            query_analysis=query_analysis
        )


class FreeRouter:
    """
    Main FreeRouter class for intelligent model routing.
    
    Automatically selects the best available free model for user queries
    based on query analysis, model capabilities, and performance metrics.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize FreeRouter.
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        
        # Initialize components
        self.client = OpenRouterClient(self.config)
        self.analyzer = QueryAnalyzer(self.config)
        self.model_manager = ModelManager(self.client, self.config)
        
        # Routing strategy
        self.routing_strategy = CapabilityBasedStrategy()
        
        # Fallback models (preferred order)
        self.fallback_models = [
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku",
            "google/gemma-7b-it"
        ]
        
        # Statistics
        self.routing_stats: Dict[str, int] = {}
        self.total_requests = 0
        
        logger.info("FreeRouter initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start the router and all components."""
        await self.model_manager.start()
        logger.info("FreeRouter started")
    
    async def stop(self) -> None:
        """Stop the router and all components."""
        await self.model_manager.stop()
        await self.client.close()
        logger.info("FreeRouter stopped")
    
    async def route_query(
        self,
        query: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        preferred_models: Optional[List[str]] = None,
        **kwargs: Any
    ) -> RouteResponse:
        """
        Route a query to the best available model.
        
        Args:
            query: User query to route
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            preferred_models: List of preferred models (optional)
            **kwargs: Additional parameters for chat completion
            
        Returns:
            RouteResponse with the response and routing information
        """
        start_time = time.time()
        self.total_requests += 1
        
        logger.info(f"Routing query: {query[:100]}...")
        
        try:
            # Analyze the query
            query_analysis = self.analyzer.analyze(query)
            logger.debug(f"Query classified as: {query_analysis.primary_type.value} (confidence: {query_analysis.confidence:.2f})")
            
            # Get available models
            available_models = self.model_manager.get_available_models()
            
            # Apply preferred models filter if provided
            if preferred_models:
                available_models = [m for m in available_models if m in preferred_models]
            
            # Make routing decision
            routing_decision = self.routing_strategy.select_model(
                query_analysis, available_models, self.model_manager
            )
            
            # Fallback if no model selected
            if not routing_decision.selected_model:
                routing_decision = await self._fallback_selection(available_models)
            
            selected_model = routing_decision.selected_model
            
            if not selected_model:
                raise ValueError("No available models found")
            
            logger.info(f"Selected model: {selected_model} (confidence: {routing_decision.confidence:.2f})")
            
            # Update routing stats
            self.routing_stats[selected_model] = self.routing_stats.get(selected_model, 0) + 1
            
            # Execute the query
            messages = [ChatMessage(role="user", content=query)]
            
            response_start = time.time()
            response = await self.client.chat_completion(
                model=selected_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            response_time = time.time() - response_start
            
            # Update model performance
            model_profile = self.model_manager.get_model_profile(selected_model)
            if model_profile:
                success = bool(response.message_content.strip())
                model_profile.update_performance(response_time, success)
            
            total_time = time.time() - start_time
            
            logger.info(f"Query completed in {total_time:.2f}s using {selected_model}")
            
            return RouteResponse(
                response=response,
                routing_decision=routing_decision,
                execution_time=total_time,
                model_used=selected_model
            )
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            
            # Try fallback
            try:
                fallback_decision = await self._fallback_selection(available_models)
                if fallback_decision.selected_model:
                    messages = [ChatMessage(role="user", content=query)]
                    response = await self.client.chat_completion(
                        model=fallback_decision.selected_model,
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        **kwargs
                    )
                    
                    total_time = time.time() - start_time
                    
                    return RouteResponse(
                        response=response,
                        routing_decision=fallback_decision,
                        execution_time=total_time,
                        model_used=fallback_decision.selected_model
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
            
            raise e
    
    async def _fallback_selection(self, available_models: List[str]) -> RoutingDecision:
        """Select a fallback model when primary routing fails."""
        # Try fallback models in order
        for fallback_model in self.fallback_models:
            if fallback_model in available_models:
                # Test the model quickly
                if await self.model_manager.test_model(fallback_model, quick_test=True):
                    return RoutingDecision(
                        selected_model=fallback_model,
                        confidence=0.3,
                        reasoning=f"Fallback model: {fallback_model}",
                        fallback_used=True
                    )
        
        # Last resort: use any available model
        if available_models:
            return RoutingDecision(
                selected_model=available_models[0],
                confidence=0.1,
                reasoning=f"Last resort: {available_models[0]}",
                fallback_used=True
            )
        
        return RoutingDecision(
            selected_model="",
            confidence=0.0,
            reasoning="No models available"
        )
    
    async def get_model_recommendations(self, query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
        """
        Get model recommendations for a query without executing it.
        
        Args:
            query: Query to analyze
            top_k: Number of recommendations to return
            
        Returns:
            List of (model_id, confidence, reasoning) tuples
        """
        query_analysis = self.analyzer.analyze(query)
        available_models = self.model_manager.get_available_models()
        
        routing_decision = self.routing_strategy.select_model(
            query_analysis, available_models, self.model_manager
        )
        
        recommendations = []
        
        # Add primary recommendation
        if routing_decision.selected_model:
            recommendations.append((
                routing_decision.selected_model,
                routing_decision.confidence,
                routing_decision.reasoning
            ))
        
        # Add alternatives
        for model_id, score in routing_decision.alternatives[:top_k-1]:
            profile = self.model_manager.get_model_profile(model_id)
            if profile:
                confidence = min(0.8, score / 5.0)  # Normalize score to confidence
                reasoning = f"Alternative with capability score: {score:.1f}"
                recommendations.append((model_id, confidence, reasoning))
        
        return recommendations[:top_k]
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_requests": self.total_requests,
            "model_usage": self.routing_stats.copy(),
            "available_models": self.model_manager.get_available_models(),
            "model_count": len(self.model_manager.profiles)
        }
    
    async def refresh_models(self) -> List[str]:
        """Refresh the list of available models."""
        return await self.model_manager.discover_models()
    
    async def test_routing(self, test_queries: List[str]) -> Dict[str, Any]:
        """
        Test routing decisions for a list of queries without executing them.
        
        Args:
            test_queries: List of test queries
            
        Returns:
            Dictionary with test results
        """
        results = {
            "total_queries": len(test_queries),
            "routing_decisions": [],
            "type_distribution": {},
            "model_usage": {}
        }
        
        for query in test_queries:
            try:
                query_analysis = self.analyzer.analyze(query)
                available_models = self.model_manager.get_available_models()
                
                routing_decision = self.routing_strategy.select_model(
                    query_analysis, available_models, self.model_manager
                )
                
                result = {
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "query_type": query_analysis.primary_type.value,
                    "complexity": query_analysis.complexity.value,
                    "selected_model": routing_decision.selected_model,
                    "confidence": routing_decision.confidence,
                    "reasoning": routing_decision.reasoning
                }
                
                results["routing_decisions"].append(result)
                
                # Update statistics
                query_type = query_analysis.primary_type.value
                results["type_distribution"][query_type] = (
                    results["type_distribution"].get(query_type, 0) + 1
                )
                
                if routing_decision.selected_model:
                    model = routing_decision.selected_model
                    results["model_usage"][model] = (
                        results["model_usage"].get(model, 0) + 1
                    )
                
            except Exception as e:
                logger.error(f"Error testing query '{query[:50]}...': {e}")
                results["routing_decisions"].append({
                    "query": query[:100] + "..." if len(query) > 100 else query,
                    "error": str(e)
                })
        
        return results


# Convenience function for simple usage
async def route_query(
    query: str,
    config: Optional[Config] = None,
    **kwargs: Any
) -> RouteResponse:
    """
    Convenience function to route a single query.
    
    Args:
        query: Query to route
        config: Optional configuration
        **kwargs: Additional parameters for chat completion
        
    Returns:
        RouteResponse with the result
    """
    async with FreeRouter(config) as router:
        return await router.route_query(query, **kwargs)