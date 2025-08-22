"""
Model evaluation engine for automated benchmarking.

This module provides comprehensive evaluation capabilities for comparing
model performance across different task types and metrics.
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

import aiofiles

from .client import OpenRouterClient, ChatMessage
from .models import ModelManager, ModelProfile
from .analyzer import QueryType
from .config import Config, get_config

logger = logging.getLogger(__name__)


class EvaluationMetric(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    EFFICIENCY = "efficiency"


@dataclass
class EvaluationResult:
    """Result of evaluating a single response."""
    model_id: str
    query: str
    query_type: str
    response: str
    scores: Dict[str, float] = field(default_factory=dict)
    response_time: float = 0.0
    success: bool = True
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def overall_score(self) -> float:
        """Calculate overall weighted score."""
        if not self.scores:
            return 0.0
        
        # Default weights from config
        weights = {
            "accuracy": 0.3,
            "relevance": 0.25,
            "clarity": 0.2,
            "completeness": 0.15,
            "efficiency": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in self.scores.items():
            weight = weights.get(metric, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


@dataclass
class BenchmarkSuite:
    """A suite of benchmark tests."""
    name: str
    queries: List[Tuple[str, str]]  # (query, expected_type)
    description: str = ""
    
    @classmethod
    def from_config(cls, config: Config, suite_name: str = "default") -> "BenchmarkSuite":
        """Create benchmark suite from configuration."""
        queries = []
        
        for query_type, query_list in config.test_queries.items():
            for query in query_list:
                queries.append((query, query_type))
        
        return cls(
            name=suite_name,
            queries=queries,
            description="Default benchmark suite from configuration"
        )


@dataclass
class EvaluationReport:
    """Comprehensive evaluation report."""
    suite_name: str
    models_evaluated: List[str]
    total_queries: int
    evaluation_time: float
    results: List[EvaluationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_model_scores(self, model_id: str) -> Dict[str, float]:
        """Get average scores for a specific model."""
        model_results = [r for r in self.results if r.model_id == model_id]
        
        if not model_results:
            return {}
        
        # Calculate averages
        metrics = set()
        for result in model_results:
            metrics.update(result.scores.keys())
        
        scores = {}
        for metric in metrics:
            metric_scores = [r.scores.get(metric, 0) for r in model_results if metric in r.scores]
            if metric_scores:
                scores[metric] = sum(metric_scores) / len(metric_scores)
        
        # Add overall score
        overall_scores = [r.overall_score for r in model_results]
        scores["overall"] = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
        
        return scores
    
    def get_type_performance(self, query_type: str) -> Dict[str, float]:
        """Get performance breakdown by query type."""
        type_results = [r for r in self.results if r.query_type == query_type]
        
        if not type_results:
            return {}
        
        model_scores = {}
        for model_id in self.models_evaluated:
            model_type_results = [r for r in type_results if r.model_id == model_id]
            if model_type_results:
                overall_scores = [r.overall_score for r in model_type_results]
                model_scores[model_id] = sum(overall_scores) / len(overall_scores)
        
        return model_scores
    
    def get_ranking(self) -> List[Tuple[str, float]]:
        """Get models ranked by overall performance."""
        model_scores = []
        
        for model_id in self.models_evaluated:
            scores = self.get_model_scores(model_id)
            overall_score = scores.get("overall", 0.0)
            model_scores.append((model_id, overall_score))
        
        return sorted(model_scores, key=lambda x: x[1], reverse=True)


class ResponseEvaluator:
    """Evaluates individual model responses."""
    
    def __init__(self, config: Optional[Config] = None):
        """Initialize the response evaluator."""
        self.config = config or get_config()
        
        # Load scoring criteria from config
        self.criteria_weights = {}
        for criterion, details in self.config.scoring_criteria.items():
            self.criteria_weights[criterion] = details.weight
    
    def evaluate_response(
        self,
        query: str,
        response: str,
        query_type: str,
        response_time: float
    ) -> Dict[str, float]:
        """
        Evaluate a response across multiple metrics.
        
        Args:
            query: Original query
            response: Model response
            query_type: Type of query
            response_time: Time taken to generate response
            
        Returns:
            Dictionary of metric scores (1-5 scale)
        """
        scores = {}
        
        # Basic checks
        if not response or not response.strip():
            return {metric.value: 1.0 for metric in EvaluationMetric}
        
        # Accuracy scoring
        scores["accuracy"] = self._score_accuracy(query, response, query_type)
        
        # Relevance scoring
        scores["relevance"] = self._score_relevance(query, response)
        
        # Clarity scoring
        scores["clarity"] = self._score_clarity(response)
        
        # Completeness scoring
        scores["completeness"] = self._score_completeness(query, response, query_type)
        
        # Efficiency scoring (based on response time and length)
        scores["efficiency"] = self._score_efficiency(response_time, len(response))
        
        return scores
    
    def _score_accuracy(self, query: str, response: str, query_type: str) -> float:
        """Score response accuracy based on query type."""
        score = 3.0  # Base score
        response_lower = response.lower()
        query_lower = query.lower()
        
        if query_type == "math":
            # Check for mathematical elements
            if any(char in response for char in "0123456789=+-*/"):
                score += 1.0
            if any(word in response_lower for word in ["answer", "solution", "equals", "result"]):
                score += 0.5
            # Check for step-by-step solution
            if any(word in response_lower for word in ["step", "first", "then", "therefore"]):
                score += 0.5
        
        elif query_type == "coding":
            # Check for code elements
            if "```" in response or any(keyword in response_lower for keyword in ["def", "function", "class", "import"]):
                score += 1.0
            if any(lang in response_lower for lang in ["python", "javascript", "java", "html", "css"]):
                score += 0.5
            # Check for explanation
            if len(response) > 100 and any(word in response_lower for word in ["this", "will", "does", "function"]):
                score += 0.5
        
        elif query_type == "analysis":
            # Check for analytical elements
            if any(word in response_lower for word in ["advantage", "disadvantage", "pros", "cons", "benefit", "drawback"]):
                score += 1.0
            if any(phrase in response_lower for phrase in ["on the other hand", "however", "in contrast", "compared to"]):
                score += 0.5
            # Check for structured analysis
            if len(response.split('\n')) > 3:
                score += 0.5
        
        elif query_type == "creative_writing":
            # Check for creative elements
            if len(response) > 200:
                score += 1.0
            if any(word in response_lower for word in ["story", "character", "scene", "chapter"]):
                score += 0.5
            # Check for narrative elements
            if any(word in response for word in ['"', "'", "said", "thought"]):
                score += 0.5
        
        return min(score, 5.0)
    
    def _score_relevance(self, query: str, response: str) -> float:
        """Score how relevant the response is to the query."""
        score = 3.0
        
        # Check for keyword overlap
        query_words = set(re.findall(r'\w+', query.lower()))
        response_words = set(re.findall(r'\w+', response.lower()))
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        query_words -= stop_words
        response_words -= stop_words
        
        if query_words:
            overlap = len(query_words & response_words) / len(query_words)
            score += overlap * 2.0
        
        # Check if response addresses the query directly
        if "?" in query and any(word in response.lower() for word in ["yes", "no", "answer", "solution"]):
            score += 0.5
        
        return min(score, 5.0)
    
    def _score_clarity(self, response: str) -> float:
        """Score response clarity and readability."""
        score = 3.0
        
        # Length check
        length = len(response)
        if length < 20:
            score -= 1.0
        elif length > 50:
            score += 0.5
        
        # Sentence structure
        sentences = re.split(r'[.!?]+', response)
        valid_sentences = [s for s in sentences if len(s.strip()) > 5]
        
        if len(valid_sentences) >= 2:
            score += 0.5
        
        # Check for formatting
        if any(marker in response for marker in ['\n', '```', '*', '-', '1.', '2.']):
            score += 0.5
        
        # Check for clear language indicators
        clarity_indicators = ["first", "second", "next", "finally", "in summary", "to conclude"]
        if any(indicator in response.lower() for indicator in clarity_indicators):
            score += 0.5
        
        return min(score, 5.0)
    
    def _score_completeness(self, query: str, response: str, query_type: str) -> float:
        """Score response completeness."""
        score = 3.0
        
        # Base length scoring
        response_length = len(response)
        if response_length > 100:
            score += 0.5
        if response_length > 300:
            score += 0.5
        
        # Check for comprehensive coverage based on query type
        if query_type == "analysis" and response_length > 200:
            # Look for multiple perspectives
            if any(phrase in response.lower() for phrase in ["on one hand", "alternatively", "however", "in addition"]):
                score += 1.0
        
        elif query_type == "coding":
            # Look for explanation with code
            if "```" in response and response_length > 150:
                score += 1.0
        
        elif query_type == "math":
            # Look for step-by-step solution
            if any(word in response.lower() for word in ["step", "first", "then", "therefore"]) and response_length > 100:
                score += 1.0
        
        # Check if response seems cut off
        if response.endswith(("...", "etc.", "and so on")):
            score -= 0.5
        
        return min(score, 5.0)
    
    def _score_efficiency(self, response_time: float, response_length: int) -> float:
        """Score efficiency based on response time and output quality."""
        score = 3.0
        
        # Time-based scoring
        if response_time < 2.0:
            score += 1.0
        elif response_time < 5.0:
            score += 0.5
        elif response_time > 15.0:
            score -= 1.0
        elif response_time > 30.0:
            score -= 2.0
        
        # Length efficiency (not too short, not unnecessarily long)
        if 50 <= response_length <= 1000:
            score += 0.5
        elif response_length < 20:
            score -= 1.0
        elif response_length > 2000:
            score -= 0.5
        
        return max(1.0, min(score, 5.0))


class ModelEvaluator:
    """
    Comprehensive model evaluation system.
    
    Runs benchmark tests against multiple models and generates
    detailed performance reports.
    """
    
    def __init__(
        self,
        client: Optional[OpenRouterClient] = None,
        model_manager: Optional[ModelManager] = None,
        config: Optional[Config] = None
    ):
        """Initialize the model evaluator."""
        self.config = config or get_config()
        self.client = client
        self.model_manager = model_manager
        self.response_evaluator = ResponseEvaluator(self.config)
        
        # Results storage
        self.results_dir = Path("evaluation_results")
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info("Model evaluator initialized")
    
    async def evaluate_models(
        self,
        models: List[str],
        benchmark_suite: Optional[BenchmarkSuite] = None,
        max_concurrent: int = 3,
        save_results: bool = True
    ) -> EvaluationReport:
        """
        Evaluate multiple models against a benchmark suite.
        
        Args:
            models: List of model IDs to evaluate
            benchmark_suite: Benchmark suite to use
            max_concurrent: Maximum concurrent evaluations
            save_results: Whether to save results to file
            
        Returns:
            Comprehensive evaluation report
        """
        if benchmark_suite is None:
            benchmark_suite = BenchmarkSuite.from_config(self.config)
        
        if not self.client:
            self.client = OpenRouterClient(self.config)
        
        logger.info(f"Starting evaluation of {len(models)} models with {len(benchmark_suite.queries)} queries")
        
        start_time = time.time()
        results = []
        
        # Create semaphore for concurrent control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_single(model_id: str, query: str, query_type: str) -> EvaluationResult:
            """Evaluate a single model-query combination."""
            async with semaphore:
                return await self._evaluate_single_response(model_id, query, query_type)
        
        # Create all evaluation tasks
        tasks = []
        for model_id in models:
            for query, query_type in benchmark_suite.queries:
                task = evaluate_single(model_id, query, query_type)
                tasks.append(task)
        
        # Execute all evaluations
        logger.info(f"Executing {len(tasks)} evaluation tasks...")
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in task_results:
            if isinstance(result, EvaluationResult):
                results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Evaluation task failed: {result}")
        
        evaluation_time = time.time() - start_time
        
        # Create report
        report = EvaluationReport(
            suite_name=benchmark_suite.name,
            models_evaluated=models,
            total_queries=len(benchmark_suite.queries),
            evaluation_time=evaluation_time,
            results=results
        )
        
        # Generate summary
        report.summary = self._generate_summary(report)
        
        logger.info(f"Evaluation completed in {evaluation_time:.1f}s. Processed {len(results)} results.")
        
        # Save results if requested
        if save_results:
            await self._save_report(report)
        
        # Update model profiles with results
        if self.model_manager:
            await self._update_model_profiles(report)
        
        return report
    
    async def _evaluate_single_response(
        self,
        model_id: str,
        query: str,
        query_type: str
    ) -> EvaluationResult:
        """Evaluate a single model response."""
        try:
            start_time = time.time()
            
            # Get response from model
            messages = [ChatMessage(role="user", content=query)]
            response = await self.client.chat_completion(
                model=model_id,
                messages=messages,
                max_tokens=500,
                temperature=0.7
            )
            
            response_time = time.time() - start_time
            response_text = response.message_content
            
            # Evaluate the response
            scores = self.response_evaluator.evaluate_response(
                query, response_text, query_type, response_time
            )
            
            return EvaluationResult(
                model_id=model_id,
                query=query,
                query_type=query_type,
                response=response_text,
                scores=scores,
                response_time=response_time,
                success=True
            )
            
        except Exception as e:
            logger.warning(f"Evaluation failed for {model_id}: {e}")
            
            return EvaluationResult(
                model_id=model_id,
                query=query,
                query_type=query_type,
                response="",
                response_time=0.0,
                success=False,
                error=str(e)
            )
    
    def _generate_summary(self, report: EvaluationReport) -> Dict[str, Any]:
        """Generate summary statistics for the report."""
        summary = {
            "model_rankings": report.get_ranking(),
            "query_type_performance": {},
            "metric_averages": {},
            "success_rates": {},
            "response_times": {}
        }
        
        # Query type performance
        query_types = set(r.query_type for r in report.results)
        for query_type in query_types:
            summary["query_type_performance"][query_type] = report.get_type_performance(query_type)
        
        # Overall metric averages
        all_metrics = set()
        for result in report.results:
            all_metrics.update(result.scores.keys())
        
        for metric in all_metrics:
            metric_scores = [r.scores.get(metric, 0) for r in report.results if metric in r.scores]
            if metric_scores:
                summary["metric_averages"][metric] = sum(metric_scores) / len(metric_scores)
        
        # Success rates and response times by model
        for model_id in report.models_evaluated:
            model_results = [r for r in report.results if r.model_id == model_id]
            
            if model_results:
                successful = [r for r in model_results if r.success]
                summary["success_rates"][model_id] = len(successful) / len(model_results)
                
                if successful:
                    avg_time = sum(r.response_time for r in successful) / len(successful)
                    summary["response_times"][model_id] = avg_time
        
        return summary
    
    async def _save_report(self, report: EvaluationReport) -> None:
        """Save evaluation report to file."""
        timestamp = report.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_report_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            # Convert report to serializable format
            report_data = {
                "suite_name": report.suite_name,
                "models_evaluated": report.models_evaluated,
                "total_queries": report.total_queries,
                "evaluation_time": report.evaluation_time,
                "timestamp": report.timestamp.isoformat(),
                "summary": report.summary,
                "results": [
                    {
                        "model_id": r.model_id,
                        "query": r.query,
                        "query_type": r.query_type,
                        "response": r.response,
                        "scores": r.scores,
                        "response_time": r.response_time,
                        "success": r.success,
                        "error": r.error,
                        "timestamp": r.timestamp.isoformat(),
                        "overall_score": r.overall_score
                    }
                    for r in report.results
                ]
            }
            
            async with aiofiles.open(filepath, 'w') as f:
                await f.write(json.dumps(report_data, indent=2))
            
            logger.info(f"Evaluation report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation report: {e}")
    
    async def _update_model_profiles(self, report: EvaluationReport) -> None:
        """Update model profiles based on evaluation results."""
        if not self.model_manager:
            return
        
        for model_id in report.models_evaluated:
            model_results = [r for r in report.results if r.model_id == model_id]
            
            if not model_results:
                continue
            
            profile = self.model_manager.get_model_profile(model_id)
            if not profile:
                continue
            
            # Update capability scores based on query types
            query_type_scores = {}
            for result in model_results:
                if result.success and result.overall_score > 0:
                    query_type = result.query_type
                    if query_type not in query_type_scores:
                        query_type_scores[query_type] = []
                    query_type_scores[query_type].append(result.overall_score)
            
            # Update profile capabilities
            capability_map = {
                "coding": "coding",
                "creative_writing": "creative_writing",
                "analysis": "analysis",
                "math": "math",
                "reasoning": "reasoning",
                "conversation": "conversation"
            }
            
            for query_type, scores in query_type_scores.items():
                capability = capability_map.get(query_type)
                if capability and scores:
                    avg_score = sum(scores) / len(scores)
                    confidence = min(1.0, len(scores) / 10.0)  # Higher confidence with more samples
                    profile.update_capability(capability, avg_score, confidence)
            
            logger.debug(f"Updated profile for {model_id} based on evaluation results")


async def run_evaluation(
    models: Optional[List[str]] = None,
    config: Optional[Config] = None
) -> EvaluationReport:
    """
    Convenience function to run a complete evaluation.
    
    Args:
        models: List of models to evaluate. If None, evaluates all available models.
        config: Configuration to use
        
    Returns:
        Evaluation report
    """
    config = config or get_config()
    
    async with OpenRouterClient(config) as client:
        async with ModelManager(client, config) as model_manager:
            if models is None:
                models = model_manager.get_available_models()
            
            evaluator = ModelEvaluator(client, model_manager, config)
            return await evaluator.evaluate_models(models)