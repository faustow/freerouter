"""
Model management system for discovering and profiling models.

This module handles model discovery, capability profiling, performance tracking,
and maintains model metadata for intelligent routing decisions.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum

import aiofiles

from .client import OpenRouterClient, ModelInfo, ChatMessage
from .analyzer import QueryType, QueryComplexity
from .config import Config, get_config

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    UNTESTED = "untested"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a model."""
    average_response_time: float = 0.0
    success_rate: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class CapabilityScore:
    """Capability scoring for a model in a specific area."""
    score: float = 0.0
    confidence: float = 0.0
    sample_count: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class ModelProfile:
    """Comprehensive profile of a model's capabilities and performance."""
    model_id: str
    model_info: Optional[ModelInfo] = None
    status: ModelStatus = ModelStatus.UNTESTED
    
    # Capability scores (1-5 scale)
    capabilities: Dict[str, CapabilityScore] = field(default_factory=dict)
    
    # Performance metrics
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    
    # Configuration and metadata
    cost_tier: str = "unknown"
    context_window: int = 4096
    specialties: List[str] = field(default_factory=list)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_tested: Optional[datetime] = None
    
    def update_capability(self, capability: str, score: float, confidence: float = 1.0) -> None:
        """Update capability score for this model."""
        if capability not in self.capabilities:
            self.capabilities[capability] = CapabilityScore()
        
        cap_score = self.capabilities[capability]
        
        # Weighted average with existing score
        if cap_score.sample_count > 0:
            total_weight = cap_score.sample_count * cap_score.confidence + confidence
            new_score = (
                (cap_score.score * cap_score.sample_count * cap_score.confidence + score * confidence) / 
                total_weight
            )
            new_confidence = total_weight / (cap_score.sample_count + 1)
        else:
            new_score = score
            new_confidence = confidence
        
        cap_score.score = new_score
        cap_score.confidence = min(new_confidence, 1.0)
        cap_score.sample_count += 1
        cap_score.last_updated = datetime.now()
        
        self.updated_at = datetime.now()
    
    def get_capability_score(self, capability: str) -> float:
        """Get capability score for a specific capability."""
        if capability in self.capabilities:
            return self.capabilities[capability].score
        return 0.0
    
    def update_performance(self, response_time: float, success: bool, error_type: Optional[str] = None) -> None:
        """Update performance metrics."""
        self.performance.total_requests += 1
        
        if success:
            self.performance.successful_requests += 1
            self.performance.last_success = datetime.now()
            
            # Update average response time
            if self.performance.successful_requests == 1:
                self.performance.average_response_time = response_time
            else:
                # Exponential moving average
                alpha = 0.1
                self.performance.average_response_time = (
                    alpha * response_time + 
                    (1 - alpha) * self.performance.average_response_time
                )
        else:
            self.performance.failed_requests += 1
            self.performance.last_failure = datetime.now()
            
            if error_type:
                self.performance.error_types[error_type] = (
                    self.performance.error_types.get(error_type, 0) + 1
                )
        
        # Update success rate
        self.performance.success_rate = (
            self.performance.successful_requests / self.performance.total_requests
        )
        
        self.updated_at = datetime.now()
    
    def is_healthy(self) -> bool:
        """Check if model is healthy and performant."""
        if self.status != ModelStatus.AVAILABLE:
            return False
        
        # Check success rate
        if self.performance.total_requests > 5 and self.performance.success_rate < 0.8:
            return False
        
        # Check recent failures
        if (self.performance.last_failure and 
            self.performance.last_failure > datetime.now() - timedelta(minutes=30)):
            return False
        
        return True


class ModelManager:
    """
    Manages model discovery, profiling, and performance tracking.
    
    Automatically discovers available models, tests their capabilities,
    and maintains performance metrics for intelligent routing.
    """
    
    def __init__(self, client: Optional[OpenRouterClient] = None, config: Optional[Config] = None):
        """
        Initialize the model manager.
        
        Args:
            client: OpenRouter client. If None, creates a new one.
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        self.client = client
        
        # Model profiles storage
        self.profiles: Dict[str, ModelProfile] = {}
        
        # Cache settings
        self.cache_file = Path("model_profiles.json")
        self.cache_ttl = timedelta(hours=24)
        
        # Discovery settings
        self.discovery_interval = timedelta(hours=6)
        self.last_discovery: Optional[datetime] = None
        
        # Background tasks
        self._discovery_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Model manager initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if self.client is None:
            self.client = OpenRouterClient(self.config)
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
        if self.client:
            await self.client.close()
    
    async def start(self) -> None:
        """Start the model manager."""
        if self._running:
            return
        
        self._running = True
        
        # Load cached profiles
        await self.load_profiles()
        
        # Start discovery if no recent discovery
        if (not self.last_discovery or 
            datetime.now() - self.last_discovery > self.discovery_interval):
            await self.discover_models()
        
        # Start background discovery task
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        
        logger.info("Model manager started")
    
    async def stop(self) -> None:
        """Stop the model manager."""
        if not self._running:
            return
        
        self._running = False
        
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
        
        # Save profiles
        await self.save_profiles()
        
        logger.info("Model manager stopped")
    
    async def _discovery_loop(self) -> None:
        """Background loop for periodic model discovery."""
        while self._running:
            try:
                await asyncio.sleep(self.discovery_interval.total_seconds())
                if self._running:
                    await self.discover_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def discover_models(self) -> List[str]:
        """
        Discover available models from OpenRouter.
        
        Returns:
            List of discovered model IDs
        """
        logger.info("Starting model discovery")
        
        if not self.client:
            self.client = OpenRouterClient(self.config)
        
        try:
            # Get free models from OpenRouter
            free_models = await self.client.get_free_models()
            discovered_models = []
            
            for model_info in free_models:
                model_id = model_info.id
                discovered_models.append(model_id)
                
                # Create or update profile
                if model_id not in self.profiles:
                    self.profiles[model_id] = ModelProfile(model_id=model_id)
                
                profile = self.profiles[model_id]
                profile.model_info = model_info
                profile.context_window = model_info.context_length
                profile.cost_tier = "free"
                
                # Update from configuration if available
                config_profile = self.config.get_model_config(model_id)
                if config_profile:
                    # Use model_dump() for Pydantic models instead of asdict for dataclasses
                    capabilities_dict = config_profile.capabilities.model_dump()
                    for capability, score in capabilities_dict.items():
                        profile.update_capability(capability, score, confidence=0.5)
                    profile.specialties = config_profile.specialties
                
                profile.updated_at = datetime.now()
            
            self.last_discovery = datetime.now()
            
            logger.info(f"Discovered {len(discovered_models)} models")
            
            # Test newly discovered models
            await self._test_models(discovered_models)
            
            return discovered_models
            
        except Exception as e:
            logger.error(f"Error during model discovery: {e}")
            return []
    
    async def _test_models(self, model_ids: List[str], max_concurrent: int = 3) -> None:
        """Test multiple models concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def test_single_model(model_id: str) -> None:
            async with semaphore:
                await self.test_model(model_id)
        
        tasks = [test_single_model(model_id) for model_id in model_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def test_model(self, model_id: str, quick_test: bool = True) -> bool:
        """
        Test a model's availability and basic functionality.
        
        Args:
            model_id: Model to test
            quick_test: If True, run only basic availability test
            
        Returns:
            True if model is working, False otherwise
        """
        if model_id not in self.profiles:
            self.profiles[model_id] = ModelProfile(model_id=model_id)
        
        profile = self.profiles[model_id]
        
        if not self.client:
            self.client = OpenRouterClient(self.config)
        
        start_time = time.time()
        
        try:
            # Basic availability test
            success = await self.client.test_model(model_id, "Hello! Can you respond?")
            response_time = time.time() - start_time
            
            if success:
                profile.status = ModelStatus.AVAILABLE
                profile.update_performance(response_time, True)
                
                if not quick_test:
                    # Extended testing for capability assessment
                    await self._assess_capabilities(model_id)
            else:
                profile.status = ModelStatus.UNAVAILABLE
                profile.update_performance(response_time, False, "no_response")
            
            profile.last_tested = datetime.now()
            
            logger.debug(f"Model {model_id} test: {'passed' if success else 'failed'}")
            return success
            
        except Exception as e:
            response_time = time.time() - start_time
            error_type = type(e).__name__
            
            profile.status = ModelStatus.ERROR
            profile.update_performance(response_time, False, error_type)
            profile.last_tested = datetime.now()
            
            logger.warning(f"Model {model_id} test failed: {e}")
            return False
    
    async def _assess_capabilities(self, model_id: str) -> None:
        """Assess model capabilities through targeted tests."""
        if not self.client:
            return
        
        profile = self.profiles[model_id]
        
        # Test different capabilities
        capability_tests = {
            "coding": "Write a simple Python function to calculate factorial of a number.",
            "creative_writing": "Write a short, creative story about a robot learning to paint.",
            "analysis": "Compare the advantages and disadvantages of renewable energy sources.",
            "math": "Solve this equation: 2x + 5 = 15. Show your work.",
            "reasoning": "Explain why correlation doesn't necessarily imply causation.",
            "conversation": "Hi! How are you doing today? What's your favorite hobby?"
        }
        
        for capability, test_prompt in capability_tests.items():
            try:
                start_time = time.time()
                
                messages = [ChatMessage(role="user", content=test_prompt)]
                response = await self.client.chat_completion(
                    model=model_id,
                    messages=messages,
                    max_tokens=200,
                    temperature=0.7
                )
                
                response_time = time.time() - start_time
                
                if response.message_content.strip():
                    # Simple scoring based on response quality
                    score = self._score_response(test_prompt, response.message_content, capability)
                    confidence = 0.7  # Medium confidence for single test
                    
                    profile.update_capability(capability, score, confidence)
                    profile.update_performance(response_time, True)
                    
                    logger.debug(f"Model {model_id} {capability} score: {score:.2f}")
                else:
                    profile.update_performance(response_time, False, "empty_response")
                
                # Small delay between tests
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.warning(f"Capability test {capability} failed for {model_id}: {e}")
                profile.update_performance(0, False, type(e).__name__)
    
    def _score_response(self, prompt: str, response: str, capability: str) -> float:
        """Score a response for a given capability (simple heuristic)."""
        if not response or len(response.strip()) < 10:
            return 1.0
        
        score = 3.0  # Base score
        
        # Length-based adjustments
        if len(response) > 100:
            score += 0.5
        if len(response) > 300:
            score += 0.5
        
        # Capability-specific scoring
        response_lower = response.lower()
        
        if capability == "coding":
            if any(word in response_lower for word in ["def", "function", "return", "if", "for"]):
                score += 1.0
            if "```" in response or "python" in response_lower:
                score += 0.5
        
        elif capability == "math":
            if any(word in response_lower for word in ["=", "solve", "answer", "solution"]):
                score += 1.0
            if any(char in response for char in "0123456789"):
                score += 0.5
        
        elif capability == "creative_writing":
            if len(response) > 200:
                score += 1.0
            if any(word in response_lower for word in ["story", "character", "scene"]):
                score += 0.5
        
        elif capability == "analysis":
            if any(word in response_lower for word in ["advantage", "disadvantage", "benefit", "drawback"]):
                score += 1.0
            if "however" in response_lower or "on the other hand" in response_lower:
                score += 0.5
        
        return min(score, 5.0)
    
    def get_models_by_capability(self, capability: str, min_score: float = 3.0) -> List[Tuple[str, float]]:
        """Get models sorted by capability score."""
        models = []
        
        for model_id, profile in self.profiles.items():
            if profile.status == ModelStatus.AVAILABLE and profile.is_healthy():
                score = profile.get_capability_score(capability)
                if score >= min_score:
                    models.append((model_id, score))
        
        return sorted(models, key=lambda x: x[1], reverse=True)
    
    def get_best_models(self, query_type: QueryType, max_models: int = 3) -> List[str]:
        """Get the best models for a specific query type."""
        capability_map = {
            QueryType.CODING: "coding",
            QueryType.CREATIVE_WRITING: "creative_writing",
            QueryType.ANALYSIS: "analysis",
            QueryType.MATH: "math",
            QueryType.REASONING: "reasoning",
            QueryType.CONVERSATION: "conversation"
        }
        
        capability = capability_map.get(query_type, "conversation")
        models_with_scores = self.get_models_by_capability(capability)
        
        return [model_id for model_id, _ in models_with_scores[:max_models]]
    
    def get_model_profile(self, model_id: str) -> Optional[ModelProfile]:
        """Get profile for a specific model."""
        return self.profiles.get(model_id)
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models."""
        return [
            model_id for model_id, profile in self.profiles.items()
            if profile.status == ModelStatus.AVAILABLE and profile.is_healthy()
        ]
    
    async def save_profiles(self) -> None:
        """Save model profiles to cache file."""
        try:
            # Convert profiles to serializable format
            serializable_profiles = {}
            
            for model_id, profile in self.profiles.items():
                profile_dict = {
                    "model_id": profile.model_id,
                    "status": profile.status.value,
                    "capabilities": {
                        cap: {
                            "score": cap_score.score,
                            "confidence": cap_score.confidence,
                            "sample_count": cap_score.sample_count,
                            "last_updated": cap_score.last_updated.isoformat() if cap_score.last_updated else None
                        }
                        for cap, cap_score in profile.capabilities.items()
                    },
                    "performance": {
                        "average_response_time": profile.performance.average_response_time,
                        "success_rate": profile.performance.success_rate,
                        "total_requests": profile.performance.total_requests,
                        "successful_requests": profile.performance.successful_requests,
                        "failed_requests": profile.performance.failed_requests,
                        "last_success": profile.performance.last_success.isoformat() if profile.performance.last_success else None,
                        "last_failure": profile.performance.last_failure.isoformat() if profile.performance.last_failure else None,
                        "error_types": profile.performance.error_types
                    },
                    "cost_tier": profile.cost_tier,
                    "context_window": profile.context_window,
                    "specialties": profile.specialties,
                    "created_at": profile.created_at.isoformat(),
                    "updated_at": profile.updated_at.isoformat(),
                    "last_tested": profile.last_tested.isoformat() if profile.last_tested else None
                }
                serializable_profiles[model_id] = profile_dict
            
            cache_data = {
                "profiles": serializable_profiles,
                "last_discovery": self.last_discovery.isoformat() if self.last_discovery else None,
                "version": "1.0"
            }
            
            async with aiofiles.open(self.cache_file, 'w') as f:
                await f.write(json.dumps(cache_data, indent=2))
            
            logger.debug(f"Saved {len(self.profiles)} model profiles to cache")
            
        except Exception as e:
            logger.error(f"Error saving model profiles: {e}")
    
    async def load_profiles(self) -> None:
        """Load model profiles from cache file."""
        try:
            if not self.cache_file.exists():
                logger.debug("No profile cache file found")
                return
            
            async with aiofiles.open(self.cache_file, 'r') as f:
                content = await f.read()
                cache_data = json.loads(content)
            
            # Check cache age
            if "last_discovery" in cache_data and cache_data["last_discovery"]:
                last_discovery = datetime.fromisoformat(cache_data["last_discovery"])
                if datetime.now() - last_discovery > self.cache_ttl:
                    logger.info("Profile cache is stale, will refresh")
                    return
                self.last_discovery = last_discovery
            
            # Load profiles
            for model_id, profile_data in cache_data.get("profiles", {}).items():
                profile = ModelProfile(model_id=model_id)
                
                # Basic fields
                profile.status = ModelStatus(profile_data.get("status", "untested"))
                profile.cost_tier = profile_data.get("cost_tier", "unknown")
                profile.context_window = profile_data.get("context_window", 4096)
                profile.specialties = profile_data.get("specialties", [])
                
                # Timestamps
                if profile_data.get("created_at"):
                    profile.created_at = datetime.fromisoformat(profile_data["created_at"])
                if profile_data.get("updated_at"):
                    profile.updated_at = datetime.fromisoformat(profile_data["updated_at"])
                if profile_data.get("last_tested"):
                    profile.last_tested = datetime.fromisoformat(profile_data["last_tested"])
                
                # Capabilities
                for cap, cap_data in profile_data.get("capabilities", {}).items():
                    cap_score = CapabilityScore(
                        score=cap_data.get("score", 0.0),
                        confidence=cap_data.get("confidence", 0.0),
                        sample_count=cap_data.get("sample_count", 0)
                    )
                    if cap_data.get("last_updated"):
                        cap_score.last_updated = datetime.fromisoformat(cap_data["last_updated"])
                    profile.capabilities[cap] = cap_score
                
                # Performance
                perf_data = profile_data.get("performance", {})
                profile.performance.average_response_time = perf_data.get("average_response_time", 0.0)
                profile.performance.success_rate = perf_data.get("success_rate", 0.0)
                profile.performance.total_requests = perf_data.get("total_requests", 0)
                profile.performance.successful_requests = perf_data.get("successful_requests", 0)
                profile.performance.failed_requests = perf_data.get("failed_requests", 0)
                profile.performance.error_types = perf_data.get("error_types", {})
                
                if perf_data.get("last_success"):
                    profile.performance.last_success = datetime.fromisoformat(perf_data["last_success"])
                if perf_data.get("last_failure"):
                    profile.performance.last_failure = datetime.fromisoformat(perf_data["last_failure"])
                
                self.profiles[model_id] = profile
            
            logger.info(f"Loaded {len(self.profiles)} model profiles from cache")
            
        except Exception as e:
            logger.error(f"Error loading model profiles: {e}")
            self.profiles = {}