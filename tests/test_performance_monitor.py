"""Tests for performance monitoring functionality."""

import pytest
import tempfile
import json
import time
from pathlib import Path

from freerouter.performance_monitor import PerformanceMonitor, PerformanceMetric, RoutingDecision
from freerouter.models import IntentType


class TestPerformanceMonitor:
    @pytest.fixture
    def monitor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            return PerformanceMonitor(data_dir=temp_dir, max_history=100)
    
    def test_record_performance(self, monitor):
        """Test recording performance metrics."""
        query_hash = monitor.record_performance(
            query="Write a Python function",
            model_id="deepseek/deepseek-r1",
            intent=IntentType.CODING,
            response_time=2.5,
            token_count=150,
            user_rating=4.2,
            correctness_score=0.85
        )
        
        assert len(query_hash) == 16  # MD5 hash truncated to 16 chars
        assert len(monitor.performance_metrics) == 1
        
        metric = monitor.performance_metrics[0]
        assert metric.query == "Write a Python function"
        assert metric.model_id == "deepseek/deepseek-r1"
        assert metric.intent == IntentType.CODING
        assert metric.response_time == 2.5
        assert metric.token_count == 150
        assert metric.user_rating == 4.2
        assert metric.correctness_score == 0.85
    
    def test_record_routing_decision(self, monitor):
        """Test recording routing decisions."""
        query_hash = monitor.record_routing_decision(
            query="Debug this code",
            router_type="semantic",
            predicted_model="deepseek/deepseek-r1",
            actual_model="deepseek/deepseek-r1",
            confidence=0.85,
            intent=IntentType.CODING,
            was_fallback=False
        )
        
        assert len(query_hash) == 16
        assert len(monitor.routing_decisions) == 1
        
        decision = monitor.routing_decisions[0]
        assert decision.query == "Debug this code"
        assert decision.router_type == "semantic"
        assert decision.predicted_model == "deepseek/deepseek-r1"
        assert decision.actual_model == "deepseek/deepseek-r1"
        assert decision.confidence == 0.85
        assert decision.intent == IntentType.CODING
        assert not decision.was_fallback
        
        # Check router accuracy stats
        accuracy_stats = monitor.stats["router_accuracy"]["semantic"]
        assert accuracy_stats["total"] == 1
        assert accuracy_stats["correct"] == 1
    
    def test_add_user_feedback(self, monitor):
        """Test adding user feedback to existing metrics."""
        # Record initial metric without rating
        query_hash = monitor.record_performance(
            query="Test query",
            model_id="deepseek/deepseek-r1",
            intent=IntentType.GENERAL,
            response_time=2.0,
            token_count=100
        )
        
        # Add user feedback
        monitor.add_user_feedback(query_hash, 4.5)
        
        # Check that feedback was added
        metric = monitor.performance_metrics[0]
        assert metric.user_rating == 4.5
        
        # Check aggregated stats updated
        model_stats = monitor.stats["model_performance"]["deepseek/deepseek-r1"]
        assert model_stats["rating_count"] == 1
        assert model_stats["avg_rating"] == 4.5
    
    def test_statistics_aggregation(self, monitor):
        """Test that statistics are properly aggregated."""
        # Add multiple metrics
        monitor.record_performance(
            query="Code query 1",
            model_id="deepseek/deepseek-r1",
            intent=IntentType.CODING,
            response_time=2.0,
            token_count=100,
            user_rating=4.0
        )
        
        monitor.record_performance(
            query="Code query 2", 
            model_id="deepseek/deepseek-r1",
            intent=IntentType.CODING,
            response_time=3.0,
            token_count=150,
            user_rating=4.5
        )
        
        monitor.record_performance(
            query="Math query",
            model_id="microsoft/phi-3-medium-128k-instruct:free",
            intent=IntentType.MATHEMATICS,
            response_time=1.5,
            token_count=80,
            user_rating=5.0
        )
        
        # Check model performance stats
        deepseek_stats = monitor.stats["model_performance"]["deepseek/deepseek-r1"]
        assert deepseek_stats["total_requests"] == 2
        assert deepseek_stats["avg_response_time"] == 2.5  # (2.0 + 3.0) / 2
        assert deepseek_stats["avg_rating"] == 4.25  # (4.0 + 4.5) / 2
        assert deepseek_stats["rating_count"] == 2
        assert deepseek_stats["intents"]["coding"] == 2
        
        # Check intent performance stats
        coding_stats = monitor.stats["intent_performance"]["coding"]
        assert coding_stats["total_requests"] == 2
        assert coding_stats["model_distribution"]["deepseek/deepseek-r1"] == 2
        assert coding_stats["avg_rating"] == 4.25
        
        math_stats = monitor.stats["intent_performance"]["mathematics"]
        assert math_stats["total_requests"] == 1
        assert math_stats["avg_rating"] == 5.0
    
    def test_get_training_data_for_mf(self, monitor):
        """Test extracting training data for Matrix Factorization router."""
        # Add metrics with ratings
        monitor.record_performance(
            "Write code", "deepseek/deepseek-r1", IntentType.CODING,
            2.0, 100, user_rating=4.0, correctness_score=0.8
        )
        
        monitor.record_performance(
            "Bad response", "google/gemma-2-9b-it:free", IntentType.GENERAL,
            5.0, 50, user_rating=2.0, correctness_score=0.4
        )
        
        monitor.record_performance(
            "Good math", "microsoft/phi-3-medium-128k-instruct:free", IntentType.MATHEMATICS,
            1.5, 75, user_rating=5.0, correctness_score=0.95
        )
        
        # Get training data with default threshold (3.0)
        training_data = monitor.get_training_data_for_mf(min_rating=3.0)
        
        # Should exclude the rating=2.0 example
        assert len(training_data) == 2
        
        example1 = training_data[0]
        assert example1.query == "Write code"
        assert example1.model_id == "deepseek/deepseek-r1"
        assert example1.correctness == 0.8  # Uses user_rating/5.0 = 4.0/5.0 = 0.8
        assert example1.response_time == 2.0
        assert example1.user_rating == 4.0
    
    def test_get_training_data_for_rf(self, monitor):
        """Test extracting training data for Random Forest router."""
        monitor.record_performance(
            "Good query", "deepseek/deepseek-r1", IntentType.CODING,
            2.0, 100, user_rating=4.5
        )
        
        monitor.record_performance(
            "Bad query", "google/gemma-2-9b-it:free", IntentType.GENERAL,
            5.0, 50, user_rating=2.0
        )
        
        training_data = monitor.get_training_data_for_rf(min_rating=3.0)
        
        assert len(training_data) == 2
        assert training_data[0].success is True  # rating >= 3.0
        assert training_data[1].success is False  # rating < 3.0
    
    def test_get_model_performance_ranking(self, monitor):
        """Test model performance ranking."""
        # Add performance data for multiple models
        monitor.record_performance(
            "Fast good response", "deepseek/deepseek-r1", IntentType.CODING,
            1.0, 100, user_rating=5.0
        )
        
        monitor.record_performance(
            "Slow good response", "google/gemma-2-9b-it:free", IntentType.WRITING,
            10.0, 150, user_rating=4.0
        )
        
        monitor.record_performance(
            "Fast bad response", "meta-llama/llama-3-8b-instruct:free", IntentType.GENERAL,
            2.0, 80, user_rating=2.0
        )
        
        ranking = monitor.get_model_performance_ranking()
        
        assert len(ranking) == 3
        
        # DeepSeek should rank highest (good rating + fast response)
        best_model, best_stats = ranking[0]
        assert best_model == "deepseek/deepseek-r1"
        assert best_stats["composite_score"] > 0.8
        
        # Check that scores are reasonable
        for model_id, stats in ranking:
            assert 0.0 <= stats["composite_score"] <= 1.0
            assert 0.0 <= stats["rating_score"] <= 1.0
            assert 0.0 <= stats["time_score"] <= 1.0
    
    def test_get_intent_insights(self, monitor):
        """Test intent-based insights."""
        monitor.record_performance(
            "Code query", "deepseek/deepseek-r1", IntentType.CODING,
            2.0, 100, user_rating=4.5
        )
        
        monitor.record_performance(
            "Another code query", "microsoft/phi-3-medium-128k-instruct:free", IntentType.CODING,
            1.5, 80, user_rating=4.0
        )
        
        monitor.record_performance(
            "Math query", "microsoft/phi-3-medium-128k-instruct:free", IntentType.MATHEMATICS,
            1.0, 60, user_rating=5.0
        )
        
        insights = monitor.get_intent_insights()
        
        assert "coding" in insights
        assert "mathematics" in insights
        
        coding_insight = insights["coding"]
        assert coding_insight["total_requests"] == 2
        assert coding_insight["avg_rating"] == 4.25  # (4.5 + 4.0) / 2
        assert coding_insight["rating_count"] == 2
        
        # Check model distribution percentages
        percentages = coding_insight["model_percentages"]
        assert percentages["deepseek/deepseek-r1"] == 50.0
        assert percentages["microsoft/phi-3-medium-128k-instruct:free"] == 50.0
    
    def test_get_router_accuracy(self, monitor):
        """Test router accuracy calculation."""
        # Record some routing decisions
        monitor.record_routing_decision(
            "Query 1", "semantic", "model1", "model1", 0.8, IntentType.CODING, False
        )  # Correct
        
        monitor.record_routing_decision(
            "Query 2", "semantic", "model1", "model2", 0.7, IntentType.CODING, True
        )  # Incorrect (fallback)
        
        monitor.record_routing_decision(
            "Query 3", "mf", "model2", "model2", 0.9, IntentType.MATHEMATICS, False
        )  # Correct
        
        accuracy = monitor.get_router_accuracy()
        
        assert accuracy["semantic"] == 0.5  # 1 correct out of 2
        assert accuracy["mf"] == 1.0  # 1 correct out of 1
        assert accuracy["rf"] == 0.0  # No RF decisions
    
    def test_feedback_callbacks(self, monitor):
        """Test feedback callback system."""
        callback_calls = []
        
        def test_callback(metric: PerformanceMetric):
            callback_calls.append(metric.query)
        
        monitor.add_feedback_callback(test_callback)
        
        monitor.record_performance(
            "Test query", "deepseek/deepseek-r1", IntentType.GENERAL,
            2.0, 100, user_rating=4.0
        )
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == "Test query"
    
    def test_export_data(self, monitor):
        """Test data export functionality."""
        # Add some data
        monitor.record_performance(
            "Test query", "deepseek/deepseek-r1", IntentType.CODING,
            2.0, 100, user_rating=4.0
        )
        
        monitor.record_routing_decision(
            "Test query", "semantic", "deepseek/deepseek-r1", "deepseek/deepseek-r1",
            0.8, IntentType.CODING, False
        )
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            result = monitor.export_data(f.name, format="json")
            
            assert "Data exported to" in result
            
            # Verify exported data
            with open(f.name, 'r') as export_file:
                data = json.load(export_file)
                
                assert "performance_metrics" in data
                assert "routing_decisions" in data
                assert "aggregated_stats" in data
                assert "export_timestamp" in data
                
                assert len(data["performance_metrics"]) == 1
                assert len(data["routing_decisions"]) == 1