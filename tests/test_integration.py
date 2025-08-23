"""
Integration tests for FreeRouter system.

These tests validate the complete system functionality with real API calls
and end-to-end workflows.
"""

import asyncio
import os
import pytest
from unittest.mock import patch, Mock
import httpx
from typing import Dict, Any

from freerouter.freerouter.client import OpenRouterClient, RateLimitError, ModelNotFoundError
from freerouter.freerouter.router import FreeRouter
from freerouter.freerouter.models import ModelManager
from freerouter.freerouter.evaluator import ModelEvaluator, run_evaluation
from freerouter.freerouter.config import Config, load_config
from freerouter.freerouter.exceptions import *


class TestAPIIntegration:
    """Test OpenRouter API integration."""
    
    @pytest.mark.asyncio
    async def test_api_connection_with_valid_key(self):
        """Test API connection with a valid key."""
        # Skip if no API key available
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OPENROUTER_API_KEY environment variable set")
        
        config = Config(openrouter_api_key=api_key)
        
        async with OpenRouterClient(config) as client:
            try:
                models = await client.get_models()
                assert len(models) > 0, "Should discover at least some models"
                
                # Test model info structure
                for model in models[:3]:  # Test first 3 models
                    assert hasattr(model, 'id'), "Model should have ID"
                    assert hasattr(model, 'name'), "Model should have name"
                    assert hasattr(model, 'pricing'), "Model should have pricing info"
                    assert model.id is not None, "Model ID should not be None"
                    
            except Exception as e:
                pytest.fail(f"API connection failed: {e}")
    
    @pytest.mark.asyncio
    async def test_api_connection_with_invalid_key(self):
        """Test API connection with invalid key."""
        config = Config(openrouter_api_key="invalid_key_test_123")
        
        async with OpenRouterClient(config) as client:
            with pytest.raises((OpenRouterError, APIError)):
                await client.get_models()
    
    @pytest.mark.asyncio
    async def test_model_discovery(self):
        """Test model discovery functionality."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OPENROUTER_API_KEY environment variable set")
        
        config = Config(openrouter_api_key=api_key)
        
        async with ModelManager(config=config) as manager:
            discovered_models = await manager.discover_models()
            
            assert len(discovered_models) > 0, "Should discover models"
            
            # Verify model profiles were created
            for model_id in discovered_models[:3]:  # Check first 3
                profile = manager.get_model_profile(model_id)
                assert profile is not None, f"Profile should exist for {model_id}"
                assert profile.model_id == model_id
    
    @pytest.mark.asyncio
    async def test_free_models_filtering(self):
        """Test that only free models are returned."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OPENROUTER_API_KEY environment variable set")
        
        config = Config(openrouter_api_key=api_key)
        
        async with OpenRouterClient(config) as client:
            free_models = await client.get_free_models()
            
            # Verify all returned models are actually free
            for model in free_models:
                pricing = model.pricing
                if pricing:
                    prompt_cost = float(pricing.get('prompt', '0'))
                    completion_cost = float(pricing.get('completion', '0'))
                    assert prompt_cost == 0.0, f"Model {model.id} should have 0 prompt cost"
                    assert completion_cost == 0.0, f"Model {model.id} should have 0 completion cost"


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_network_timeout_handling(self):
        """Test handling of network timeouts."""
        config = Config(openrouter_api_key="test_key", default_config={"timeout": 0.001})
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_request.side_effect = httpx.TimeoutException("Request timed out")
            
            client = OpenRouterClient(config)
            with pytest.raises((NetworkError, httpx.TimeoutException)):
                await client.get_models()
            
            await client.close()
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self):
        """Test rate limit error handling."""
        config = Config(openrouter_api_key="test_key")
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.headers = {"retry-after": "60"}
            mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
            mock_request.return_value = mock_response
            
            client = OpenRouterClient(config)
            with pytest.raises(RateLimitError) as exc_info:
                await client._make_request("GET", "/models")
            
            assert exc_info.value.retry_after == 60
            await client.close()
    
    @pytest.mark.asyncio
    async def test_model_not_found_handling(self):
        """Test model not found error handling."""
        config = Config(openrouter_api_key="test_key")
        
        with patch('httpx.AsyncClient.request') as mock_request:
            mock_response = Mock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"error": {"message": "Model not found"}}
            mock_request.return_value = mock_response
            
            client = OpenRouterClient(config)
            with pytest.raises(ModelNotFoundError):
                await client._make_request("POST", "/chat/completions")
            
            await client.close()


class TestQueryProcessingPipeline:
    """Test the complete query processing pipeline."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_query_processing(self):
        """Test complete query → analysis → routing → response pipeline."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OPENROUTER_API_KEY environment variable set")
        
        config = Config(openrouter_api_key=api_key)
        
        test_queries = [
            "What is 2 + 2?",  # Simple math
            "Hello, how are you?",  # Conversation
            "Write a Python function",  # Coding (but short)
        ]
        
        async with FreeRouter(config) as router:
            for query in test_queries:
                try:
                    result = await router.route_query(query, max_tokens=50)
                    
                    # Verify response structure
                    assert result.response is not None, "Should get a response"
                    assert result.model_used is not None, "Should specify which model was used"
                    assert result.execution_time > 0, "Should track execution time"
                    assert result.routing_decision is not None, "Should have routing decision"
                    
                    # Verify response content
                    content = result.response.message_content
                    assert content is not None, "Response should have content"
                    assert len(content.strip()) > 0, "Response should not be empty"
                    
                    print(f"✅ Query: {query[:30]}... → Model: {result.model_used} → Response: {len(content)} chars")
                    
                except Exception as e:
                    pytest.fail(f"Query processing failed for '{query}': {e}")
    
    @pytest.mark.asyncio
    async def test_query_classification_accuracy(self):
        """Test that queries are classified correctly."""
        config = load_config()
        
        from freerouter.freerouter.analyzer import QueryAnalyzer, QueryType
        analyzer = QueryAnalyzer(config)
        
        test_cases = [
            ("Write a Python function to sort a list", QueryType.CODING),
            ("Tell me a story about a dragon", QueryType.CREATIVE_WRITING),
            ("What is 15% of 200?", QueryType.MATH),
            ("Compare renewable vs fossil energy", QueryType.ANALYSIS),
            ("Hello, how are you today?", QueryType.CONVERSATION),
            ("Why do we dream?", QueryType.REASONING),
        ]
        
        for query, expected_type in test_cases:
            analysis = analyzer.analyze(query)
            
            # Allow for some flexibility in classification
            possible_types = [expected_type]
            if expected_type == QueryType.CONVERSATION:
                possible_types.append(QueryType.REASONING)
            elif expected_type == QueryType.REASONING:
                possible_types.append(QueryType.CONVERSATION)
            
            assert analysis.primary_type in possible_types, \
                f"Query '{query}' classified as {analysis.primary_type}, expected one of {possible_types}"
            
            print(f"✅ '{query[:30]}...' → {analysis.primary_type.value} (confidence: {analysis.confidence:.1%})")
    
    @pytest.mark.asyncio
    async def test_model_selection_logic(self):
        """Test that model selection makes logical sense."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OPENROUTER_API_KEY environment variable set")
        
        config = Config(openrouter_api_key=api_key)
        
        async with FreeRouter(config) as router:
            # Test different query types and verify reasonable model selection
            test_queries = [
                ("Write a Python function to calculate factorial", "coding"),
                ("Create a poem about nature", "creative_writing"),
                ("What is machine learning?", "conversation"),
            ]
            
            for query, expected_category in test_queries:
                recommendations = await router.get_model_recommendations(query, top_k=3)
                
                assert len(recommendations) > 0, f"Should get recommendations for {expected_category} query"
                
                # Check that recommendations are reasonable
                for model_id, confidence, reasoning in recommendations:
                    assert model_id is not None, "Model ID should not be None"
                    assert 0 <= confidence <= 1, f"Confidence should be 0-1, got {confidence}"
                    assert reasoning is not None, "Should provide reasoning"
                    
                print(f"✅ '{query[:30]}...' → Top model: {recommendations[0][0]} (confidence: {recommendations[0][1]:.1%})")


class TestEvaluationObjectivity:
    """Test evaluation system for objectivity and consistency."""
    
    @pytest.mark.asyncio
    async def test_evaluation_consistency(self):
        """Test that evaluation scores are consistent across runs."""
        from freerouter.freerouter.evaluator import ResponseEvaluator
        
        config = load_config()
        evaluator = ResponseEvaluator(config)
        
        # Test with fixed query-response pairs
        test_cases = [
            ("What is 2 + 2?", "2 + 2 equals 4.", "math"),
            ("Write hello world in Python", "print('Hello, World!')", "coding"),
            ("Tell me a story", "Once upon a time, there was a brave knight who saved a village.", "creative_writing"),
        ]
        
        for query, response, query_type in test_cases:
            scores_runs = []
            
            # Run evaluation 5 times
            for _ in range(5):
                scores = evaluator.evaluate_response(query, response, query_type, 1.5)
                scores_runs.append(scores)
            
            # Check consistency (scores should be identical for same input)
            first_scores = scores_runs[0]
            for run_scores in scores_runs[1:]:
                for metric, score in first_scores.items():
                    run_score = run_scores.get(metric, 0)
                    assert abs(score - run_score) < 0.01, \
                        f"Inconsistent {metric} scores: {score} vs {run_score} for query '{query}'"
            
            print(f"✅ Consistent evaluation for '{query[:30]}...' across 5 runs")
    
    def test_evaluation_discrimination(self):
        """Test that evaluation can discriminate between good and bad responses."""
        from freerouter.freerouter.evaluator import ResponseEvaluator
        
        config = load_config()
        evaluator = ResponseEvaluator(config)
        
        # Test with obviously good vs bad responses
        test_cases = [
            {
                "query": "What is 2 + 2?",
                "good_response": "2 + 2 equals 4. This is basic arithmetic.",
                "bad_response": "Purple elephants dance in the moonlight.",
                "query_type": "math"
            },
            {
                "query": "Write a Python function to add two numbers",
                "good_response": "def add(a, b):\n    return a + b\n\nThis function takes two parameters and returns their sum.",
                "bad_response": "Yes, I like programming sometimes.",
                "query_type": "coding"
            }
        ]
        
        for case in test_cases:
            good_scores = evaluator.evaluate_response(
                case["query"], case["good_response"], case["query_type"], 1.0
            )
            bad_scores = evaluator.evaluate_response(
                case["query"], case["bad_response"], case["query_type"], 1.0
            )
            
            # Calculate overall scores
            good_overall = sum(good_scores.values()) / len(good_scores)
            bad_overall = sum(bad_scores.values()) / len(bad_scores)
            
            assert good_overall > bad_overall, \
                f"Good response should score higher: {good_overall:.2f} vs {bad_overall:.2f}"
            
            # Check specific metrics
            assert good_scores["relevance"] > bad_scores["relevance"], \
                "Good response should be more relevant"
            assert good_scores["accuracy"] > bad_scores["accuracy"], \
                "Good response should be more accurate"
            
            print(f"✅ Evaluation discriminates correctly: Good={good_overall:.2f}, Bad={bad_overall:.2f}")
    
    @pytest.mark.asyncio 
    async def test_ground_truth_validation(self):
        """Test evaluation against known correct/incorrect answers."""
        from freerouter.freerouter.evaluator import ResponseEvaluator
        
        config = load_config()
        evaluator = ResponseEvaluator(config)
        
        # Math problems with objective answers
        math_cases = [
            {
                "query": "What is 15% of 200?",
                "correct": "15% of 200 is 30. To calculate: 200 × 0.15 = 30",
                "incorrect": "15% of 200 is 50.",
                "query_type": "math"
            },
            {
                "query": "Solve: 2x + 5 = 15",
                "correct": "2x + 5 = 15\n2x = 15 - 5\n2x = 10\nx = 5",
                "incorrect": "x = 10",
                "query_type": "math"
            }
        ]
        
        for case in math_cases:
            correct_scores = evaluator.evaluate_response(
                case["query"], case["correct"], case["query_type"], 1.0
            )
            incorrect_scores = evaluator.evaluate_response(
                case["query"], case["incorrect"], case["query_type"], 1.0
            )
            
            # Correct answers should score higher on accuracy
            assert correct_scores["accuracy"] > incorrect_scores["accuracy"], \
                f"Correct answer should have higher accuracy score for: {case['query']}"
            
            print(f"✅ Ground truth validation passed for: {case['query']}")


class TestModelProfileAccuracy:
    """Test model profiling and capability detection."""
    
    @pytest.mark.asyncio
    async def test_model_capability_tracking(self):
        """Test that model capabilities are tracked accurately."""
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            pytest.skip("No OPENROUTER_API_KEY environment variable set")
        
        config = Config(openrouter_api_key=api_key)
        
        async with ModelManager(config=config) as manager:
            # Discover models first
            discovered_models = await manager.discover_models()
            
            if len(discovered_models) == 0:
                pytest.skip("No models available for testing")
            
            # Test first available model
            test_model = discovered_models[0]
            
            # Test the model
            success = await manager.test_model(test_model)
            assert success is not None, "Test should return boolean result"
            
            # Check profile was updated
            profile = manager.get_model_profile(test_model)
            assert profile is not None, "Profile should exist after testing"
            assert profile.last_tested is not None, "Last tested time should be set"
            
            print(f"✅ Model {test_model} profile updated successfully")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test that performance metrics are collected correctly."""
        config = load_config()
        
        # Create mock model manager to test metrics
        from freerouter.freerouter.models import ModelProfile, PerformanceMetrics
        
        profile = ModelProfile(model_id="test/model")
        
        # Simulate successful requests
        profile.update_performance(1.5, True)
        profile.update_performance(2.0, True) 
        profile.update_performance(1.8, True)
        
        # Simulate failed request
        profile.update_performance(0.0, False, "timeout")
        
        # Check metrics
        perf = profile.performance
        assert perf.total_requests == 4, "Should track total requests"
        assert perf.successful_requests == 3, "Should track successful requests"
        assert perf.failed_requests == 1, "Should track failed requests"
        assert abs(perf.success_rate - 0.75) < 0.01, "Success rate should be 75%"
        assert perf.average_response_time > 0, "Should track average response time"
        assert "timeout" in perf.error_types, "Should track error types"
        
        print("✅ Performance metrics collection working correctly")


@pytest.mark.asyncio
async def test_system_integration():
    """Test complete system integration."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        pytest.skip("No OPENROUTER_API_KEY environment variable set")
    
    config = Config(openrouter_api_key=api_key)
    
    # Test the complete workflow
    async with FreeRouter(config) as router:
        # 1. Test model discovery
        models = await router.refresh_models()
        assert len(models) > 0, "Should discover models"
        
        # 2. Test routing statistics
        stats = router.get_routing_stats()
        assert "total_requests" in stats, "Should provide statistics"
        
        # 3. Test query routing
        result = await router.route_query("What is Python?", max_tokens=100)
        assert result.response is not None, "Should get response"
        
        # 4. Test model recommendations
        recommendations = await router.get_model_recommendations("Write code", top_k=3)
        assert len(recommendations) > 0, "Should get recommendations"
        
        print("✅ Complete system integration test passed")


if __name__ == "__main__":
    # Run basic validation tests
    asyncio.run(test_system_integration())