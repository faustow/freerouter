"""Performance monitoring and feedback collection for FreeRouter."""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import asyncio
import hashlib

from .models import IntentType, FREE_MODELS
from .matrix_factorization_router import TrainingExample
from .random_forest_router import RFTrainingExample


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    query_hash: str
    query: str
    model_id: str
    intent: IntentType
    response_time: float  # seconds
    token_count: int
    user_rating: Optional[float] = None  # 0.0-5.0 scale
    correctness_score: Optional[float] = None  # 0.0-1.0 automated score
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class RoutingDecision:
    """Record of a routing decision and its outcome."""
    query_hash: str
    query: str
    router_type: str  # "semantic", "mf", "rf"
    predicted_model: str
    actual_model: str  # May differ due to rate limits
    confidence: float
    intent: IntentType
    was_fallback: bool
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class PerformanceMonitor:
    """Monitor and collect performance data for router improvement."""
    
    def __init__(self, data_dir: str = "data", max_history: int = 10000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_history = max_history
        
        # In-memory storage for recent data
        self.performance_metrics: deque = deque(maxlen=max_history)
        self.routing_decisions: deque = deque(maxlen=max_history)
        
        # Aggregated statistics
        self.stats = {
            "total_requests": 0,
            "model_performance": defaultdict(lambda: {
                "total_requests": 0,
                "avg_response_time": 0.0,
                "avg_rating": 0.0,
                "rating_count": 0,
                "success_rate": 0.0,
                "intents": defaultdict(int)
            }),
            "intent_performance": defaultdict(lambda: {
                "total_requests": 0,
                "model_distribution": defaultdict(int),
                "avg_rating": 0.0,
                "rating_count": 0
            }),
            "router_accuracy": {
                "semantic": {"correct": 0, "total": 0},
                "mf": {"correct": 0, "total": 0},
                "rf": {"correct": 0, "total": 0}
            }
        }
        
        # Load existing data
        self._load_historical_data()
        
        # Feedback callbacks
        self.feedback_callbacks: List[Callable[[PerformanceMetric], None]] = []
    
    def _hash_query(self, query: str) -> str:
        """Create a hash for the query to enable deduplication."""
        return hashlib.md5(query.encode()).hexdigest()[:16]
    
    def record_performance(
        self,
        query: str,
        model_id: str,
        intent: IntentType,
        response_time: float,
        token_count: int,
        user_rating: Optional[float] = None,
        correctness_score: Optional[float] = None
    ) -> str:
        """Record a performance measurement."""
        query_hash = self._hash_query(query)
        
        metric = PerformanceMetric(
            query_hash=query_hash,
            query=query,
            model_id=model_id,
            intent=intent,
            response_time=response_time,
            token_count=token_count,
            user_rating=user_rating,
            correctness_score=correctness_score
        )
        
        self.performance_metrics.append(metric)
        self._update_stats(metric)
        
        # Save to disk periodically
        if len(self.performance_metrics) % 100 == 0:
            self._save_to_disk()
        
        # Trigger callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(metric)
            except Exception as e:
                print(f"Error in feedback callback: {e}")
        
        return query_hash
    
    def record_routing_decision(
        self,
        query: str,
        router_type: str,
        predicted_model: str,
        actual_model: str,
        confidence: float,
        intent: IntentType,
        was_fallback: bool = False
    ) -> str:
        """Record a routing decision."""
        query_hash = self._hash_query(query)
        
        decision = RoutingDecision(
            query_hash=query_hash,
            query=query,
            router_type=router_type,
            predicted_model=predicted_model,
            actual_model=actual_model,
            confidence=confidence,
            intent=intent,
            was_fallback=was_fallback
        )
        
        self.routing_decisions.append(decision)
        
        # Update router accuracy
        is_correct = (predicted_model == actual_model) and not was_fallback
        self.stats["router_accuracy"][router_type]["total"] += 1
        if is_correct:
            self.stats["router_accuracy"][router_type]["correct"] += 1
        
        return query_hash
    
    def add_user_feedback(self, query_hash: str, rating: float):
        """Add user feedback to an existing performance record."""
        # Find the matching performance metric
        for metric in reversed(self.performance_metrics):
            if metric.query_hash == query_hash:
                old_rating = metric.user_rating
                metric.user_rating = rating
                
                # Update aggregated stats
                model_stats = self.stats["model_performance"][metric.model_id]
                intent_stats = self.stats["intent_performance"][metric.intent.value]
                
                if old_rating is None:
                    # First rating for this metric
                    model_stats["rating_count"] += 1
                    intent_stats["rating_count"] += 1
                    
                    model_stats["avg_rating"] = (
                        (model_stats["avg_rating"] * (model_stats["rating_count"] - 1) + rating) /
                        model_stats["rating_count"]
                    )
                    intent_stats["avg_rating"] = (
                        (intent_stats["avg_rating"] * (intent_stats["rating_count"] - 1) + rating) /
                        intent_stats["rating_count"]
                    )
                else:
                    # Update existing rating
                    model_stats["avg_rating"] = (
                        (model_stats["avg_rating"] * model_stats["rating_count"] - old_rating + rating) /
                        model_stats["rating_count"]
                    )
                    intent_stats["avg_rating"] = (
                        (intent_stats["avg_rating"] * intent_stats["rating_count"] - old_rating + rating) /
                        intent_stats["rating_count"]
                    )
                
                break
    
    def _update_stats(self, metric: PerformanceMetric):
        """Update aggregated statistics."""
        self.stats["total_requests"] += 1
        
        # Model performance
        model_stats = self.stats["model_performance"][metric.model_id]
        model_stats["total_requests"] += 1
        model_stats["intents"][metric.intent.value] += 1
        
        # Update average response time
        old_avg = model_stats["avg_response_time"]
        count = model_stats["total_requests"]
        model_stats["avg_response_time"] = (
            (old_avg * (count - 1) + metric.response_time) / count
        )
        
        # Update rating if available
        if metric.user_rating is not None:
            model_stats["rating_count"] += 1
            old_rating_avg = model_stats["avg_rating"]
            rating_count = model_stats["rating_count"]
            model_stats["avg_rating"] = (
                (old_rating_avg * (rating_count - 1) + metric.user_rating) / rating_count
            )
        
        # Intent performance
        intent_stats = self.stats["intent_performance"][metric.intent.value]
        intent_stats["total_requests"] += 1
        intent_stats["model_distribution"][metric.model_id] += 1
        
        if metric.user_rating is not None:
            intent_stats["rating_count"] += 1
            old_rating_avg = intent_stats["avg_rating"]
            rating_count = intent_stats["rating_count"]
            intent_stats["avg_rating"] = (
                (old_rating_avg * (rating_count - 1) + metric.user_rating) / rating_count
            )
    
    def get_training_data_for_mf(self, min_rating: float = 3.0) -> List[TrainingExample]:
        """Extract training data for Matrix Factorization router."""
        training_data = []
        
        for metric in self.performance_metrics:
            if metric.user_rating is not None or metric.correctness_score is not None:
                # Use user rating if available, otherwise correctness score
                correctness = metric.user_rating / 5.0 if metric.user_rating else metric.correctness_score
                
                # Skip very low quality examples
                if correctness is not None and correctness >= (min_rating / 5.0):
                    example = TrainingExample(
                        query=metric.query,
                        model_id=metric.model_id,
                        correctness=correctness,
                        response_time=metric.response_time,
                        user_rating=metric.user_rating
                    )
                    training_data.append(example)
        
        return training_data
    
    def get_training_data_for_rf(self, min_rating: float = 3.0) -> List[RFTrainingExample]:
        """Extract training data for Random Forest router."""
        training_data = []
        
        for metric in self.performance_metrics:
            # Determine success based on rating or correctness score
            success = True  # Default assumption
            
            if metric.user_rating is not None:
                success = metric.user_rating >= min_rating
            elif metric.correctness_score is not None:
                success = metric.correctness_score >= (min_rating / 5.0)
            
            example = RFTrainingExample(
                query=metric.query,
                model_id=metric.model_id,
                success=success
            )
            training_data.append(example)
        
        return training_data
    
    def get_model_performance_ranking(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Get models ranked by performance."""
        models = []
        
        for model_id, stats in self.stats["model_performance"].items():
            if stats["total_requests"] > 0:
                # Composite score: rating * 0.6 + response_time_score * 0.4
                rating_score = stats["avg_rating"] / 5.0 if stats["rating_count"] > 0 else 0.5
                
                # Response time score (inverse relationship, capped at 30s)
                avg_time = min(stats["avg_response_time"], 30.0)
                time_score = max(0, 1.0 - avg_time / 30.0)
                
                composite_score = rating_score * 0.6 + time_score * 0.4
                
                model_data = {
                    **stats,
                    "composite_score": composite_score,
                    "rating_score": rating_score,
                    "time_score": time_score
                }
                models.append((model_id, model_data))
        
        # Sort by composite score
        models.sort(key=lambda x: x[1]["composite_score"], reverse=True)
        return models
    
    def get_intent_insights(self) -> Dict[str, Dict[str, Any]]:
        """Get insights about intent-model performance."""
        insights = {}
        
        for intent, stats in self.stats["intent_performance"].items():
            if stats["total_requests"] > 0:
                # Find best performing model for this intent
                best_model = max(
                    stats["model_distribution"].items(),
                    key=lambda x: x[1]
                )[0]
                
                insights[intent] = {
                    "total_requests": stats["total_requests"],
                    "avg_rating": stats["avg_rating"],
                    "rating_count": stats["rating_count"],
                    "best_model": best_model,
                    "model_distribution": dict(stats["model_distribution"]),
                    "model_percentages": {
                        model: count / stats["total_requests"] * 100
                        for model, count in stats["model_distribution"].items()
                    }
                }
        
        return insights
    
    def get_router_accuracy(self) -> Dict[str, float]:
        """Get accuracy statistics for each router type."""
        accuracy = {}
        
        for router_type, stats in self.stats["router_accuracy"].items():
            if stats["total"] > 0:
                accuracy[router_type] = stats["correct"] / stats["total"]
            else:
                accuracy[router_type] = 0.0
        
        return accuracy
    
    def add_feedback_callback(self, callback: Callable[[PerformanceMetric], None]):
        """Add a callback to be triggered when new performance data is recorded."""
        self.feedback_callbacks.append(callback)
    
    def _save_to_disk(self):
        """Save current data to disk."""
        # Save performance metrics
        metrics_file = self.data_dir / "performance_metrics.jsonl"
        with open(metrics_file, "w") as f:
            for metric in self.performance_metrics:
                f.write(json.dumps(asdict(metric)) + "\n")
        
        # Save routing decisions
        decisions_file = self.data_dir / "routing_decisions.jsonl"
        with open(decisions_file, "w") as f:
            for decision in self.routing_decisions:
                decision_dict = asdict(decision)
                decision_dict["intent"] = decision_dict["intent"].value  # Convert enum
                f.write(json.dumps(decision_dict) + "\n")
        
        # Save aggregated stats
        stats_file = self.data_dir / "aggregated_stats.json"
        with open(stats_file, "w") as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            serializable_stats = json.loads(json.dumps(self.stats, default=dict))
            json.dump(serializable_stats, f, indent=2)
    
    def _load_historical_data(self):
        """Load historical data from disk."""
        try:
            # Load performance metrics
            metrics_file = self.data_dir / "performance_metrics.jsonl"
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        data["intent"] = IntentType(data["intent"])  # Convert back to enum
                        metric = PerformanceMetric(**data)
                        self.performance_metrics.append(metric)
                        self._update_stats(metric)
            
            # Load routing decisions
            decisions_file = self.data_dir / "routing_decisions.jsonl"
            if decisions_file.exists():
                with open(decisions_file, "r") as f:
                    for line in f:
                        data = json.loads(line.strip())
                        data["intent"] = IntentType(data["intent"])  # Convert back to enum
                        decision = RoutingDecision(**data)
                        self.routing_decisions.append(decision)
                        
                        # Update router accuracy
                        router_type = decision.router_type
                        is_correct = (decision.predicted_model == decision.actual_model) and not decision.was_fallback
                        self.stats["router_accuracy"][router_type]["total"] += 1
                        if is_correct:
                            self.stats["router_accuracy"][router_type]["correct"] += 1
            
        except Exception as e:
            print(f"Warning: Could not load historical data: {e}")
    
    def export_data(self, filepath: str, format: str = "json") -> str:
        """Export all collected data."""
        export_data = {
            "performance_metrics": [asdict(m) for m in self.performance_metrics],
            "routing_decisions": [asdict(d) for d in self.routing_decisions],
            "aggregated_stats": self.stats,
            "export_timestamp": time.time()
        }
        
        if format.lower() == "json":
            with open(filepath, "w") as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return f"Data exported to {filepath}"