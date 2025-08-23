#!/usr/bin/env python3
"""
FreeRouter System Validation Script

This script runs comprehensive tests to validate the FreeRouter system
and identify any issues that need to be fixed.
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from freerouter.freerouter.config import Config, load_config
from freerouter.freerouter.client import OpenRouterClient
from freerouter.freerouter.router import FreeRouter
from freerouter.freerouter.models import ModelManager
from freerouter.freerouter.analyzer import QueryAnalyzer, QueryType
from freerouter.freerouter.evaluator import ResponseEvaluator, ModelEvaluator
from freerouter.freerouter.exceptions import *

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('validation.log')
    ]
)
logger = logging.getLogger(__name__)


class ValidationResult:
    """Stores validation test results."""
    
    def __init__(self):
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.errors = []
        self.warnings = []
        self.performance_metrics = {}
        self.start_time = time.time()
    
    def add_pass(self, test_name: str, details: str = ""):
        """Record a passing test."""
        self.tests_run += 1
        self.tests_passed += 1
        logger.info(f"✅ PASS: {test_name} {details}")
    
    def add_fail(self, test_name: str, error: str):
        """Record a failing test."""
        self.tests_run += 1
        self.tests_failed += 1
        self.errors.append(f"{test_name}: {error}")
        logger.error(f"❌ FAIL: {test_name} - {error}")
    
    def add_warning(self, test_name: str, warning: str):
        """Record a warning."""
        self.warnings.append(f"{test_name}: {warning}")
        logger.warning(f"⚠️  WARN: {test_name} - {warning}")
    
    def add_metric(self, name: str, value: Any):
        """Record a performance metric."""
        self.performance_metrics[name] = value
        logger.info(f"📊 METRIC: {name} = {value}")
    
    def summary(self) -> Dict[str, Any]:
        """Get validation summary."""
        duration = time.time() - self.start_time
        
        return {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "tests_run": self.tests_run,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "success_rate": round(self.tests_passed / self.tests_run * 100, 1) if self.tests_run > 0 else 0,
            "errors": self.errors,
            "warnings": self.warnings,
            "performance_metrics": self.performance_metrics,
            "overall_status": "PASS" if self.tests_failed == 0 else "FAIL"
        }


class SystemValidator:
    """Comprehensive system validator."""
    
    def __init__(self):
        self.result = ValidationResult()
        self.config = None
        self.api_key = os.getenv('OPENROUTER_API_KEY')
    
    async def validate_all(self) -> ValidationResult:
        """Run all validation tests."""
        logger.info("🚀 Starting FreeRouter system validation...")
        
        try:
            # Phase 1: Core functionality
            await self.validate_configuration()
            await self.validate_api_integration()
            await self.validate_query_processing()
            
            # Phase 2: Evaluation system
            await self.validate_evaluation_system()
            await self.validate_model_profiling()
            
            # Phase 3: Integration tests
            await self.validate_end_to_end()
            
            # Phase 4: Performance tests
            await self.validate_performance()
            
        except Exception as e:
            self.result.add_fail("System Validation", f"Unexpected error: {e}")
            logger.exception("Validation failed with exception")
        
        # Generate summary
        summary = self.result.summary()
        logger.info(f"\n📋 VALIDATION SUMMARY:")
        logger.info(f"   Tests run: {summary['tests_run']}")
        logger.info(f"   Passed: {summary['tests_passed']}")
        logger.info(f"   Failed: {summary['tests_failed']}")
        logger.info(f"   Success rate: {summary['success_rate']}%")
        logger.info(f"   Duration: {summary['duration_seconds']}s")
        logger.info(f"   Overall status: {summary['overall_status']}")
        
        if summary['errors']:
            logger.error("❌ ERRORS FOUND:")
            for error in summary['errors']:
                logger.error(f"   - {error}")
        
        if summary['warnings']:
            logger.warning("⚠️  WARNINGS:")
            for warning in summary['warnings']:
                logger.warning(f"   - {warning}")
        
        return self.result
    
    async def validate_configuration(self):
        """Validate configuration loading and setup."""
        try:
            # Test configuration loading
            self.config = load_config()
            self.result.add_pass("Configuration Loading", "Config loaded successfully")
            
            # Check API key
            if not self.api_key:
                self.result.add_warning("API Key", "No OPENROUTER_API_KEY environment variable set")
            else:
                self.result.add_pass("API Key", "API key found in environment")
            
            # Validate config structure
            required_sections = ['model_profiles', 'query_types', 'test_queries']
            for section in required_sections:
                if hasattr(self.config, section):
                    self.result.add_pass(f"Config Section: {section}", f"Found {len(getattr(self.config, section))} items")
                else:
                    self.result.add_fail(f"Config Section: {section}", "Missing required section")
            
        except Exception as e:
            self.result.add_fail("Configuration Validation", str(e))
    
    async def validate_api_integration(self):
        """Validate OpenRouter API integration."""
        if not self.api_key:
            self.result.add_warning("API Integration", "Skipping API tests - no API key")
            return
        
        try:
            # Test basic API connection
            config = Config(openrouter_api_key=self.api_key)
            
            async with OpenRouterClient(config) as client:
                start_time = time.time()
                models = await client.get_models()
                api_time = time.time() - start_time
                
                self.result.add_metric("api_response_time", round(api_time, 2))
                self.result.add_pass("API Connection", f"Retrieved {len(models)} models in {api_time:.2f}s")
                
                # Test free models filtering
                free_models = await client.get_free_models()
                self.result.add_pass("Free Models Filtering", f"Found {len(free_models)} free models")
                
                # Test model info retrieval
                if models:
                    test_model = models[0]
                    model_info = await client.get_model_info(test_model.id)
                    if model_info:
                        self.result.add_pass("Model Info Retrieval", f"Got info for {test_model.id}")
                    else:
                        self.result.add_fail("Model Info Retrieval", "No model info returned")
                
                # Test basic model functionality
                if free_models:
                    test_model = free_models[0]
                    is_working = await client.test_model(test_model.id, "Hello")
                    if is_working:
                        self.result.add_pass("Model Testing", f"Model {test_model.id} is working")
                    else:
                        self.result.add_warning("Model Testing", f"Model {test_model.id} failed test")
        
        except Exception as e:
            self.result.add_fail("API Integration", str(e))
    
    async def validate_query_processing(self):
        """Validate query analysis and processing."""
        try:
            analyzer = QueryAnalyzer(self.config)
            
            # Test query classification
            test_queries = [
                ("Write a Python function to sort a list", QueryType.CODING),
                ("Tell me a story about dragons", QueryType.CREATIVE_WRITING), 
                ("What is 2 + 2?", QueryType.MATH),
                ("Compare cats and dogs", QueryType.ANALYSIS),
                ("Hello, how are you?", QueryType.CONVERSATION),
                ("Why do we dream?", QueryType.REASONING),
            ]
            
            correct_classifications = 0
            for query, expected_type in test_queries:
                analysis = analyzer.analyze(query)
                
                # Allow some flexibility in classification
                acceptable_types = [expected_type]
                if expected_type == QueryType.CONVERSATION:
                    acceptable_types.append(QueryType.REASONING)
                elif expected_type == QueryType.REASONING:
                    acceptable_types.append(QueryType.CONVERSATION)
                
                if analysis.primary_type in acceptable_types:
                    correct_classifications += 1
                else:
                    self.result.add_warning(
                        "Query Classification",
                        f"'{query}' classified as {analysis.primary_type}, expected {expected_type}"
                    )
            
            classification_rate = correct_classifications / len(test_queries) * 100
            self.result.add_metric("query_classification_accuracy", round(classification_rate, 1))
            
            if classification_rate >= 70:  # Allow for some flexibility
                self.result.add_pass("Query Classification", f"{classification_rate:.1f}% accuracy")
            else:
                self.result.add_fail("Query Classification", f"Only {classification_rate:.1f}% accuracy")
            
            # Test feature extraction
            test_query = "Write a Python function using JavaScript and SQL to debug this error"
            analysis = analyzer.analyze(test_query)
            
            if analysis.features.has_code_indicators:
                self.result.add_pass("Feature Extraction", "Code indicators detected correctly")
            else:
                self.result.add_fail("Feature Extraction", "Failed to detect code indicators")
            
            if len(analysis.features.programming_languages) >= 2:
                self.result.add_pass("Language Detection", f"Detected languages: {analysis.features.programming_languages}")
            else:
                self.result.add_warning("Language Detection", "Should detect multiple programming languages")
        
        except Exception as e:
            self.result.add_fail("Query Processing", str(e))
    
    async def validate_evaluation_system(self):
        """Validate evaluation system objectivity and consistency."""
        try:
            evaluator = ResponseEvaluator(self.config)
            
            # Test evaluation consistency
            query = "What is 2 + 2?"
            response = "2 + 2 equals 4."
            query_type = "math"
            
            scores_runs = []
            for _ in range(5):
                scores = evaluator.evaluate_response(query, response, query_type, 1.5)
                scores_runs.append(scores)
            
            # Check consistency
            first_scores = scores_runs[0]
            max_variance = 0
            
            for run_scores in scores_runs[1:]:
                for metric, score in first_scores.items():
                    variance = abs(score - run_scores.get(metric, 0))
                    max_variance = max(max_variance, variance)
            
            if max_variance < 0.01:
                self.result.add_pass("Evaluation Consistency", f"Max variance: {max_variance:.3f}")
            else:
                self.result.add_fail("Evaluation Consistency", f"High variance: {max_variance:.3f}")
            
            # Test discrimination ability
            good_response = "2 + 2 equals 4. This is basic arithmetic."
            bad_response = "Purple elephants dance in the moonlight."
            
            good_scores = evaluator.evaluate_response(query, good_response, query_type, 1.0)
            bad_scores = evaluator.evaluate_response(query, bad_response, query_type, 1.0)
            
            good_overall = sum(good_scores.values()) / len(good_scores)
            bad_overall = sum(bad_scores.values()) / len(bad_scores)
            
            if good_overall > bad_overall:
                self.result.add_pass("Evaluation Discrimination", f"Good: {good_overall:.2f}, Bad: {bad_overall:.2f}")
            else:
                self.result.add_fail("Evaluation Discrimination", "Cannot distinguish good from bad responses")
            
            self.result.add_metric("evaluation_discrimination_ratio", round(good_overall / bad_overall, 2))
        
        except Exception as e:
            self.result.add_fail("Evaluation System", str(e))
    
    async def validate_model_profiling(self):
        """Validate model profiling and capability tracking."""
        try:
            if not self.api_key:
                self.result.add_warning("Model Profiling", "Skipping - no API key")
                return
            
            config = Config(openrouter_api_key=self.api_key)
            
            async with ModelManager(config=config) as manager:
                # Test model discovery
                start_time = time.time()
                discovered = await manager.discover_models()
                discovery_time = time.time() - start_time
                
                self.result.add_metric("model_discovery_time", round(discovery_time, 2))
                
                if len(discovered) > 0:
                    self.result.add_pass("Model Discovery", f"Discovered {len(discovered)} models in {discovery_time:.2f}s")
                else:
                    self.result.add_fail("Model Discovery", "No models discovered")
                
                # Test profile creation
                if discovered:
                    test_model = discovered[0]
                    profile = manager.get_model_profile(test_model)
                    
                    if profile:
                        self.result.add_pass("Profile Creation", f"Profile created for {test_model}")
                        
                        # Test capability scoring
                        if profile.capabilities:
                            self.result.add_pass("Capability Tracking", f"Tracking {len(profile.capabilities)} capabilities")
                        else:
                            self.result.add_warning("Capability Tracking", "No capabilities tracked")
                    else:
                        self.result.add_fail("Profile Creation", f"No profile created for {test_model}")
        
        except Exception as e:
            self.result.add_fail("Model Profiling", str(e))
    
    async def validate_end_to_end(self):
        """Validate complete end-to-end functionality."""
        if not self.api_key:
            self.result.add_warning("End-to-End", "Skipping - no API key")
            return
        
        try:
            config = Config(openrouter_api_key=self.api_key)
            
            async with FreeRouter(config) as router:
                # Test simple query routing
                test_queries = [
                    "What is Python?",
                    "Hello, how are you?",
                    "What is 5 + 3?",
                ]
                
                successful_routes = 0
                total_time = 0
                
                for query in test_queries:
                    try:
                        start_time = time.time()
                        result = await router.route_query(query, max_tokens=50)
                        route_time = time.time() - start_time
                        
                        total_time += route_time
                        
                        if result.response and result.response.message_content.strip():
                            successful_routes += 1
                        else:
                            self.result.add_warning("Query Routing", f"Empty response for: {query}")
                    
                    except Exception as e:
                        self.result.add_warning("Query Routing", f"Failed query '{query}': {e}")
                
                success_rate = successful_routes / len(test_queries) * 100
                avg_time = total_time / len(test_queries) if test_queries else 0
                
                self.result.add_metric("end_to_end_success_rate", round(success_rate, 1))
                self.result.add_metric("average_query_time", round(avg_time, 2))
                
                if success_rate >= 80:
                    self.result.add_pass("End-to-End Routing", f"{success_rate:.1f}% success, avg time: {avg_time:.2f}s")
                else:
                    self.result.add_fail("End-to-End Routing", f"Only {success_rate:.1f}% success rate")
                
                # Test model recommendations
                recommendations = await router.get_model_recommendations("Write code", top_k=3)
                if recommendations:
                    self.result.add_pass("Model Recommendations", f"Generated {len(recommendations)} recommendations")
                else:
                    self.result.add_fail("Model Recommendations", "No recommendations generated")
        
        except Exception as e:
            self.result.add_fail("End-to-End", str(e))
    
    async def validate_performance(self):
        """Validate system performance."""
        try:
            # Test analyzer performance
            analyzer = QueryAnalyzer(self.config)
            
            test_queries = ["Test query " + str(i) for i in range(100)]
            
            start_time = time.time()
            for query in test_queries:
                analyzer.analyze(query)
            analysis_time = time.time() - start_time
            
            queries_per_sec = len(test_queries) / analysis_time
            self.result.add_metric("analyzer_queries_per_second", round(queries_per_sec, 1))
            
            if queries_per_sec >= 50:  # Should be able to analyze 50+ queries/sec
                self.result.add_pass("Analyzer Performance", f"{queries_per_sec:.1f} queries/sec")
            else:
                self.result.add_warning("Analyzer Performance", f"Only {queries_per_sec:.1f} queries/sec")
            
            # Test memory usage (basic check)
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.result.add_metric("memory_usage_mb", round(memory_mb, 1))
            
            if memory_mb < 500:  # Should use less than 500MB
                self.result.add_pass("Memory Usage", f"{memory_mb:.1f} MB")
            else:
                self.result.add_warning("Memory Usage", f"High usage: {memory_mb:.1f} MB")
        
        except Exception as e:
            self.result.add_fail("Performance Validation", str(e))


async def main():
    """Run system validation."""
    print("🔍 FreeRouter System Validation")
    print("=" * 50)
    
    validator = SystemValidator()
    result = await validator.validate_all()
    
    # Save results to file
    results_file = Path("validation_results.json")
    with open(results_file, 'w') as f:
        json.dump(result.summary(), f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    if result.tests_failed > 0:
        print("❌ Validation FAILED - see errors above")
        sys.exit(1)
    else:
        print("✅ Validation PASSED")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())