#!/usr/bin/env python3
"""
Benchmark script for evaluating FreeRouter models.

This script provides convenient functions for running comprehensive
evaluations and generating detailed reports.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from freerouter.freerouter.evaluator import run_evaluation, BenchmarkSuite
from freerouter.freerouter.config import load_config
from freerouter.freerouter.models import ModelManager
from freerouter.freerouter.client import OpenRouterClient

logger = logging.getLogger(__name__)


async def benchmark_all_models(config_path: Optional[str] = None) -> None:
    """Run benchmark evaluation on all available models."""
    print("🚀 Starting comprehensive model benchmark...")
    
    try:
        # Load configuration
        config = load_config() if not config_path else load_config(models_config_path=config_path)
        
        # Run evaluation
        report = await run_evaluation(config=config)
        
        print(f"\n✅ Benchmark completed!")
        print(f"📊 Models evaluated: {len(report.models_evaluated)}")
        print(f"📋 Total queries: {report.total_queries}")
        print(f"⏱️  Total time: {report.evaluation_time:.1f}s")
        
        # Show top performers
        rankings = report.get_ranking()
        if rankings:
            print(f"\n🏆 Top 3 Models:")
            for i, (model_id, score) in enumerate(rankings[:3], 1):
                print(f"  {i}. {model_id}: {score:.2f}/5.0")
        
        # Show query type winners
        print(f"\n🎯 Best by Category:")
        for query_type, model_scores in report.summary.get("query_type_performance", {}).items():
            if model_scores:
                best_model, best_score = max(model_scores.items(), key=lambda x: x[1])
                print(f"  {query_type}: {best_model} ({best_score:.2f})")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logger.exception("Benchmark error")
        sys.exit(1)


async def quick_benchmark(models: Optional[List[str]] = None) -> None:
    """Run quick benchmark with a subset of queries."""
    print("⚡ Starting quick benchmark...")
    
    # Quick test queries
    quick_queries = [
        ("Write a Python function to calculate factorial", "coding"),
        ("What is the capital of France?", "conversation"),
        ("Solve: 2x + 5 = 15", "math"),
        ("Write a short story about a robot", "creative_writing"),
        ("Compare renewable vs fossil fuels", "analysis"),
        ("Why do we dream?", "reasoning")
    ]
    
    try:
        config = load_config()
        
        # Create quick benchmark suite
        benchmark_suite = BenchmarkSuite(
            name="quick_benchmark",
            queries=quick_queries,
            description="Quick benchmark for basic functionality testing"
        )
        
        async with OpenRouterClient(config) as client:
            async with ModelManager(client, config) as manager:
                from freerouter.freerouter.evaluator import ModelEvaluator
                
                # Get models to test
                if models:
                    test_models = models
                else:
                    test_models = manager.get_available_models()
                    if not test_models:
                        print("❌ No models available. Try running model discovery first.")
                        return
                
                print(f"🧪 Testing {len(test_models)} models with {len(quick_queries)} queries...")
                
                evaluator = ModelEvaluator(client, manager, config)
                report = await evaluator.evaluate_models(
                    test_models,
                    benchmark_suite,
                    max_concurrent=2,
                    save_results=True
                )
                
                print(f"\n✅ Quick benchmark completed!")
                print(f"⏱️  Time: {report.evaluation_time:.1f}s")
                
                # Show results
                rankings = report.get_ranking()
                if rankings:
                    print(f"\n🏆 Rankings:")
                    for i, (model_id, score) in enumerate(rankings, 1):
                        print(f"  {i}. {model_id}: {score:.2f}/5.0")
        
    except Exception as e:
        print(f"❌ Quick benchmark failed: {e}")
        logger.exception("Quick benchmark error")
        sys.exit(1)


async def test_model_connectivity() -> None:
    """Test connectivity to all configured models."""
    print("🔌 Testing model connectivity...")
    
    try:
        config = load_config()
        
        async with ModelManager(config=config) as manager:
            # Discover models first
            discovered = await manager.discover_models()
            print(f"📡 Discovered {len(discovered)} models")
            
            # Test each model
            results = []
            for model_id in discovered:
                print(f"  Testing {model_id}...", end="")
                success = await manager.test_model(model_id, quick_test=True)
                results.append((model_id, success))
                print(" ✅" if success else " ❌")
            
            # Summary
            working = sum(1 for _, success in results if success)
            print(f"\n📊 {working}/{len(results)} models are working")
            
            if working == 0:
                print("⚠️  No models are currently available. Check your API key and network connection.")
    
    except Exception as e:
        print(f"❌ Connectivity test failed: {e}")
        logger.exception("Connectivity test error")
        sys.exit(1)


def main():
    """Main script entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="FreeRouter Benchmark Tool")
    parser.add_argument("--quick", "-q", action="store_true", help="Run quick benchmark")
    parser.add_argument("--connectivity", "-c", action="store_true", help="Test model connectivity")
    parser.add_argument("--models", "-m", nargs="+", help="Specific models to test")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        if args.connectivity:
            asyncio.run(test_model_connectivity())
        elif args.quick:
            asyncio.run(quick_benchmark(args.models))
        else:
            asyncio.run(benchmark_all_models(args.config))
    except KeyboardInterrupt:
        print("\n⏹️  Benchmark cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()