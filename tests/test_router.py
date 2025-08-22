"""
Tests for the FreeRouter core routing logic.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from freerouter.freerouter.router import FreeRouter, CapabilityBasedStrategy, RoutingDecision
from freerouter.freerouter.analyzer import QueryType, QueryComplexity


class TestCapabilityBasedStrategy:
    """Test cases for CapabilityBasedStrategy."""
    
    def test_select_model_coding_query(self, mock_config, mock_model_manager):
        """Test model selection for coding queries."""
        from freerouter.freerouter.analyzer import QueryAnalysis, QueryFeatures
        
        strategy = CapabilityBasedStrategy()
        
        # Create a coding query analysis
        analysis = QueryAnalysis(
            query="Write a Python function",
            primary_type=QueryType.CODING,
            complexity=QueryComplexity.MODERATE,
            confidence=0.8,
            features=QueryFeatures(50, 8, 1, 0, True, False, False, False, False)
        )
        
        available_models = ["test/model-1", "test/model-2"]
        decision = strategy.select_model(analysis, available_models, mock_model_manager)
        
        assert decision.selected_model in available_models
        assert decision.confidence > 0
        assert "coding" in decision.reasoning.lower()
    
    def test_select_model_creative_query(self, mock_config, mock_model_manager):
        """Test model selection for creative writing queries."""
        from freerouter.freerouter.analyzer import QueryAnalysis, QueryFeatures
        
        strategy = CapabilityBasedStrategy()
        
        analysis = QueryAnalysis(
            query="Write a story",
            primary_type=QueryType.CREATIVE_WRITING,
            complexity=QueryComplexity.MODERATE,
            confidence=0.9,
            features=QueryFeatures(30, 5, 1, 0, False, False, True, False, False)
        )
        
        available_models = ["test/model-1", "test/model-2"]
        decision = strategy.select_model(analysis, available_models, mock_model_manager)
        
        assert decision.selected_model == "test/model-2"  # Better at creative writing
        assert decision.confidence > 0
    
    def test_select_model_no_models(self, mock_config, mock_model_manager):
        """Test model selection when no models available."""
        from freerouter.freerouter.analyzer import QueryAnalysis, QueryFeatures
        
        strategy = CapabilityBasedStrategy()
        
        analysis = QueryAnalysis(
            query="Test query",
            primary_type=QueryType.CONVERSATION,
            complexity=QueryComplexity.SIMPLE,
            confidence=0.5,
            features=QueryFeatures(10, 2, 1, 0, False, False, False, False, False)
        )
        
        decision = strategy.select_model(analysis, [], mock_model_manager)
        
        assert decision.selected_model == ""
        assert decision.confidence == 0.0
        assert "No available models" in decision.reasoning
    
    def test_select_model_with_secondary_types(self, mock_config, mock_model_manager):
        """Test model selection with secondary query types."""
        from freerouter.freerouter.analyzer import QueryAnalysis, QueryFeatures
        
        strategy = CapabilityBasedStrategy()
        
        analysis = QueryAnalysis(
            query="Analyze different sorting algorithms and implement the best one",
            primary_type=QueryType.ANALYSIS,
            secondary_types=[QueryType.CODING],
            complexity=QueryComplexity.COMPLEX,
            confidence=0.8,
            features=QueryFeatures(100, 12, 2, 0, True, False, False, True, False)
        )
        
        available_models = ["test/model-1", "test/model-2"]
        decision = strategy.select_model(analysis, available_models, mock_model_manager)
        
        assert decision.selected_model in available_models
        assert decision.confidence > 0


class TestFreeRouter:
    """Test cases for FreeRouter."""
    
    @pytest.mark.asyncio
    async def test_init(self, mock_config):
        """Test FreeRouter initialization."""
        router = FreeRouter(mock_config)
        assert router.config == mock_config
        assert hasattr(router, 'client')
        assert hasattr(router, 'analyzer')
        assert hasattr(router, 'model_manager')
        assert hasattr(router, 'routing_strategy')
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_config):
        """Test async context manager functionality."""
        with patch.object(FreeRouter, 'start') as mock_start, \
             patch.object(FreeRouter, 'stop') as mock_stop:
            mock_start.return_value = AsyncMock()
            mock_stop.return_value = AsyncMock()
            
            async with FreeRouter(mock_config) as router:
                assert isinstance(router, FreeRouter)
    
    @pytest.mark.asyncio
    async def test_route_query_success(self, mock_config):
        """Test successful query routing."""
        router = FreeRouter(mock_config)
        
        # Mock dependencies
        router.analyzer = Mock()
        router.analyzer.analyze.return_value = Mock(
            primary_type=QueryType.CODING,
            secondary_types=[],
            complexity=QueryComplexity.MODERATE,
            confidence=0.8
        )
        
        router.model_manager = Mock()
        router.model_manager.get_available_models.return_value = ["test/model-1"]
        
        router.routing_strategy = Mock()
        router.routing_strategy.select_model.return_value = RoutingDecision(
            selected_model="test/model-1",
            confidence=0.8,
            reasoning="Test routing decision"
        )
        
        router.client = Mock()
        mock_response = Mock()
        mock_response.message_content = "Test response"
        router.client.chat_completion = AsyncMock(return_value=mock_response)
        
        result = await router.route_query("Test query")
        
        assert result.model_used == "test/model-1"
        assert result.response == mock_response
        assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_route_query_with_preferred_models(self, mock_config):
        """Test query routing with preferred models."""
        router = FreeRouter(mock_config)
        
        # Mock dependencies
        router.analyzer = Mock()
        router.analyzer.analyze.return_value = Mock(
            primary_type=QueryType.CODING,
            secondary_types=[],
            complexity=QueryComplexity.MODERATE,
            confidence=0.8
        )
        
        router.model_manager = Mock()
        router.model_manager.get_available_models.return_value = ["test/model-1", "test/model-2"]
        
        router.routing_strategy = Mock()
        router.routing_strategy.select_model.return_value = RoutingDecision(
            selected_model="test/model-2",
            confidence=0.9,
            reasoning="Preferred model selected"
        )
        
        router.client = Mock()
        mock_response = Mock()
        mock_response.message_content = "Test response"
        router.client.chat_completion = AsyncMock(return_value=mock_response)
        
        # Route with preferred models
        result = await router.route_query("Test query", preferred_models=["test/model-2"])
        
        assert result.model_used == "test/model-2"
    
    @pytest.mark.asyncio
    async def test_route_query_fallback(self, mock_config):
        """Test query routing fallback mechanism."""
        router = FreeRouter(mock_config)
        
        # Mock dependencies
        router.analyzer = Mock()
        router.analyzer.analyze.return_value = Mock(
            primary_type=QueryType.CODING,
            secondary_types=[],
            complexity=QueryComplexity.MODERATE,
            confidence=0.8
        )
        
        router.model_manager = Mock()
        router.model_manager.get_available_models.return_value = ["test/model-1"]
        router.model_manager.test_model = AsyncMock(return_value=True)
        
        # First routing fails
        router.routing_strategy = Mock()
        router.routing_strategy.select_model.return_value = RoutingDecision(
            selected_model="",  # No model selected
            confidence=0.0,
            reasoning="No suitable model"
        )
        
        router.client = Mock()
        mock_response = Mock()
        mock_response.message_content = "Fallback response"
        router.client.chat_completion = AsyncMock(return_value=mock_response)
        
        result = await router.route_query("Test query")
        
        # Should use fallback
        assert result.routing_decision.fallback_used
    
    @pytest.mark.asyncio
    async def test_get_model_recommendations(self, mock_config):
        """Test getting model recommendations."""
        router = FreeRouter(mock_config)
        
        # Mock dependencies
        router.analyzer = Mock()
        router.analyzer.analyze.return_value = Mock(
            primary_type=QueryType.CODING,
            secondary_types=[],
            complexity=QueryComplexity.MODERATE,
            confidence=0.8
        )
        
        router.model_manager = Mock()
        router.model_manager.get_available_models.return_value = ["test/model-1", "test/model-2"]
        router.model_manager.get_model_profile.return_value = Mock()
        
        router.routing_strategy = Mock()
        router.routing_strategy.select_model.return_value = RoutingDecision(
            selected_model="test/model-1",
            confidence=0.8,
            reasoning="Best for coding",
            alternatives=[("test/model-2", 0.6)]
        )
        
        recommendations = await router.get_model_recommendations("Write a function")
        
        assert len(recommendations) > 0
        assert all(len(rec) == 3 for rec in recommendations)  # (model, confidence, reasoning)
    
    def test_get_routing_stats(self, mock_config):
        """Test getting routing statistics."""
        router = FreeRouter(mock_config)
        router.total_requests = 10
        router.routing_stats = {"test/model-1": 7, "test/model-2": 3}
        
        # Mock model manager
        router.model_manager = Mock()
        router.model_manager.get_available_models.return_value = ["test/model-1", "test/model-2"]
        router.model_manager.profiles = {"test/model-1": Mock(), "test/model-2": Mock()}
        
        stats = router.get_routing_stats()
        
        assert stats["total_requests"] == 10
        assert stats["model_usage"]["test/model-1"] == 7
        assert stats["model_usage"]["test/model-2"] == 3
        assert len(stats["available_models"]) == 2
        assert stats["model_count"] == 2
    
    @pytest.mark.asyncio
    async def test_refresh_models(self, mock_config):
        """Test refreshing models."""
        router = FreeRouter(mock_config)
        
        router.model_manager = Mock()
        router.model_manager.discover_models = AsyncMock(return_value=["test/model-1", "test/model-2"])
        
        models = await router.refresh_models()
        
        assert len(models) == 2
        assert "test/model-1" in models
        assert "test/model-2" in models
    
    @pytest.mark.asyncio
    async def test_test_routing(self, mock_config, sample_queries):
        """Test routing test functionality."""
        router = FreeRouter(mock_config)
        
        # Mock dependencies
        router.analyzer = Mock()
        router.analyzer.analyze.return_value = Mock(
            primary_type=QueryType.CODING,
            complexity=QueryComplexity.MODERATE
        )
        
        router.model_manager = Mock()
        router.model_manager.get_available_models.return_value = ["test/model-1"]
        
        router.routing_strategy = Mock()
        router.routing_strategy.select_model.return_value = RoutingDecision(
            selected_model="test/model-1",
            confidence=0.8,
            reasoning="Test decision"
        )
        
        test_queries = sample_queries["coding"][:2]  # Test with 2 queries
        results = await router.test_routing(test_queries)
        
        assert results["total_queries"] == 2
        assert len(results["routing_decisions"]) == 2
        assert "type_distribution" in results
        assert "model_usage" in results


class TestRoutingDecision:
    """Test RoutingDecision data class."""
    
    def test_routing_decision_creation(self):
        """Test creating RoutingDecision instances."""
        decision = RoutingDecision(
            selected_model="test/model-1",
            confidence=0.8,
            reasoning="Test reasoning",
            alternatives=[("test/model-2", 0.6)],
            fallback_used=False
        )
        
        assert decision.selected_model == "test/model-1"
        assert decision.confidence == 0.8
        assert decision.reasoning == "Test reasoning"
        assert len(decision.alternatives) == 1
        assert not decision.fallback_used
    
    def test_routing_decision_fallback(self):
        """Test RoutingDecision with fallback."""
        decision = RoutingDecision(
            selected_model="fallback/model",
            confidence=0.3,
            reasoning="Fallback used",
            fallback_used=True
        )
        
        assert decision.fallback_used
        assert decision.confidence == 0.3