"""
Command-line interface for FreeRouter.

Provides a user-friendly CLI for interacting with the FreeRouter system,
including query routing, model management, and evaluation commands.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.markdown import Markdown

from .router import FreeRouter
from .evaluator import ModelEvaluator, BenchmarkSuite, run_evaluation
from .config import Config, load_config
from .models import ModelManager
from .client import OpenRouterClient

console = Console()


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stderr)]
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file')
@click.pass_context
def cli(ctx, verbose: bool, config: Optional[str]) -> None:
    """FreeRouter - Intelligent Model Router for OpenRouter Free Models."""
    setup_logging(verbose)
    
    # Load configuration
    if config:
        ctx.obj = load_config(models_config_path=config)
    else:
        ctx.obj = load_config()
    
    # Check for API key
    if not ctx.obj.openrouter_api_key:
        console.print(Panel(
            "[red]Warning: No OpenRouter API key found!\n\n"
            "Set the OPENROUTER_API_KEY environment variable or\n"
            "add it to your configuration file.[/red]",
            title="⚠️  Configuration Warning"
        ))


@cli.command()
@click.argument('query', nargs=-1, required=True)
@click.option('--max-tokens', '-t', type=int, help='Maximum tokens to generate')
@click.option('--temperature', type=float, default=0.7, help='Sampling temperature (0.0-1.0)')
@click.option('--model', '-m', help='Preferred model to use')
@click.option('--explain', '-e', is_flag=True, help='Show routing decision explanation')
@click.option('--no-stream', is_flag=True, help='Disable response streaming')
@click.pass_obj
def ask(config: Config, query: tuple, max_tokens: Optional[int], temperature: float, 
        model: Optional[str], explain: bool, no_stream: bool) -> None:
    """Ask FreeRouter a question and get an intelligent response."""
    query_text = ' '.join(query)
    
    if not query_text.strip():
        console.print("[red]Error: Please provide a query.[/red]")
        return
    
    async def run_query():
        async with FreeRouter(config) as router:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
                transient=True
            ) as progress:
                task = progress.add_task("Analyzing query and selecting model...", total=None)
                
                try:
                    # Get routing recommendations if explain mode
                    if explain:
                        recommendations = await router.get_model_recommendations(query_text)
                        
                        if recommendations:
                            progress.update(task, description="Model selection complete")
                            progress.stop()
                            
                            # Show routing decision
                            table = Table(title="🤖 Model Selection")
                            table.add_column("Rank", style="cyan")
                            table.add_column("Model", style="green")
                            table.add_column("Confidence", style="yellow")
                            table.add_column("Reasoning", style="white")
                            
                            for i, (model_id, confidence, reasoning) in enumerate(recommendations, 1):
                                table.add_row(
                                    str(i),
                                    model_id,
                                    f"{confidence:.1%}",
                                    reasoning
                                )
                            
                            console.print(table)
                            console.print()
                    
                    progress.update(task, description="Generating response...")
                    
                    # Route the query
                    preferred_models = [model] if model else None
                    result = await router.route_query(
                        query_text,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        preferred_models=preferred_models
                    )
                    
                    progress.stop()
                    
                    # Display response
                    console.print(Panel(
                        result.response.message_content,
                        title=f"📝 Response (via {result.model_used})",
                        border_style="green"
                    ))
                    
                    if explain:
                        # Show detailed routing info
                        console.print(f"\n[dim]Execution time: {result.execution_time:.2f}s[/dim]")
                        if result.routing_decision.fallback_used:
                            console.print("[yellow]⚠️  Fallback model was used[/yellow]")
                
                except Exception as e:
                    progress.stop()
                    console.print(f"[red]Error: {e}[/red]")
                    sys.exit(1)
    
    asyncio.run(run_query())


@cli.command()
@click.option('--refresh', '-r', is_flag=True, help='Refresh model list from API')
@click.option('--test', '-t', is_flag=True, help='Test model availability')
@click.pass_obj
def models(config: Config, refresh: bool, test: bool) -> None:
    """List and manage available models."""
    
    async def run_models():
        async with ModelManager(config=config) as manager:
            if refresh:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Discovering models...", total=None)
                    discovered = await manager.discover_models()
                    progress.stop()
                    console.print(f"[green]Discovered {len(discovered)} models[/green]")
            
            if test:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("Testing model availability...", total=None)
                    available_models = manager.get_available_models()
                    for model_id in available_models:
                        await manager.test_model(model_id)
                    progress.stop()
            
            # Display models table
            profiles = manager.profiles
            
            if not profiles:
                console.print("[yellow]No models found. Try running with --refresh[/yellow]")
                return
            
            table = Table(title="🤖 Available Models")
            table.add_column("Model ID", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Context", style="yellow")
            table.add_column("Specialties", style="white")
            table.add_column("Success Rate", style="blue")
            table.add_column("Avg Time", style="magenta")
            
            for model_id, profile in sorted(profiles.items()):
                status_color = {
                    "available": "green",
                    "unavailable": "red", 
                    "error": "red",
                    "untested": "yellow"
                }.get(profile.status.value, "white")
                
                status = f"[{status_color}]{profile.status.value}[/{status_color}]"
                
                context = f"{profile.context_window:,}"
                specialties = ", ".join(profile.specialties[:2]) if profile.specialties else "-"
                
                if profile.performance.total_requests > 0:
                    success_rate = f"{profile.performance.success_rate:.1%}"
                    avg_time = f"{profile.performance.average_response_time:.1f}s"
                else:
                    success_rate = "-"
                    avg_time = "-"
                
                table.add_row(
                    model_id,
                    status,
                    context,
                    specialties,
                    success_rate,
                    avg_time
                )
            
            console.print(table)
    
    asyncio.run(run_models())


@cli.command()
@click.option('--models', '-m', multiple=True, help='Specific models to evaluate')
@click.option('--quick', '-q', is_flag=True, help='Run quick evaluation with fewer queries')
@click.option('--save/--no-save', default=True, help='Save results to file')
@click.pass_obj
def evaluate(config: Config, models: tuple, quick: bool, save: bool) -> None:
    """Run comprehensive model evaluation."""
    
    async def run_evaluation():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Setting up evaluation...", total=None)
            
            try:
                model_list = list(models) if models else None
                
                if quick:
                    # Create a smaller benchmark suite for quick testing
                    quick_queries = [
                        ("Write a Python function to reverse a string", "coding"),
                        ("Explain photosynthesis briefly", "conversation"),
                        ("What is 15% of 240?", "math"),
                        ("Compare cats and dogs as pets", "analysis")
                    ]
                    benchmark_suite = BenchmarkSuite(
                        name="quick_test",
                        queries=quick_queries,
                        description="Quick evaluation suite"
                    )
                else:
                    benchmark_suite = None
                
                progress.update(task, description="Running evaluation...")
                
                if model_list:
                    async with OpenRouterClient(config) as client:
                        async with ModelManager(client, config) as manager:
                            evaluator = ModelEvaluator(client, manager, config)
                            report = await evaluator.evaluate_models(
                                model_list,
                                benchmark_suite,
                                save_results=save
                            )
                else:
                    report = await run_evaluation(model_list, config)
                
                progress.stop()
                
                # Display results
                console.print(Panel(
                    f"Evaluation completed!\n\n"
                    f"📊 Models evaluated: {len(report.models_evaluated)}\n"
                    f"📋 Total queries: {report.total_queries}\n"
                    f"⏱️  Evaluation time: {report.evaluation_time:.1f}s\n"
                    f"✅ Results processed: {len(report.results)}",
                    title="🎯 Evaluation Complete",
                    border_style="green"
                ))
                
                # Show rankings
                rankings = report.get_ranking()
                if rankings:
                    table = Table(title="🏆 Model Rankings")
                    table.add_column("Rank", style="cyan")
                    table.add_column("Model", style="green")
                    table.add_column("Overall Score", style="yellow")
                    
                    for i, (model_id, score) in enumerate(rankings, 1):
                        table.add_row(str(i), model_id, f"{score:.2f}/5.0")
                    
                    console.print(table)
                
                # Show query type performance
                if report.summary.get("query_type_performance"):
                    console.print("\n[bold]Performance by Query Type:[/bold]")
                    for query_type, model_scores in report.summary["query_type_performance"].items():
                        if model_scores:
                            best_model = max(model_scores.items(), key=lambda x: x[1])
                            console.print(f"  {query_type}: [green]{best_model[0]}[/green] ({best_model[1]:.2f})")
                
            except Exception as e:
                progress.stop()
                console.print(f"[red]Evaluation failed: {e}[/red]")
                sys.exit(1)
    
    asyncio.run(run_evaluation())


@cli.command()
@click.pass_obj
def stats(config: Config) -> None:
    """Show router statistics and performance metrics."""
    
    async def show_stats():
        async with FreeRouter(config) as router:
            stats_data = router.get_routing_stats()
            
            console.print(Panel(
                f"📊 Total requests: {stats_data['total_requests']}\n"
                f"🤖 Available models: {len(stats_data['available_models'])}\n"
                f"📋 Known models: {stats_data['model_count']}",
                title="📈 Router Statistics",
                border_style="blue"
            ))
            
            if stats_data['model_usage']:
                table = Table(title="Model Usage")
                table.add_column("Model", style="cyan")
                table.add_column("Requests", style="green")
                table.add_column("Percentage", style="yellow")
                
                total = stats_data['total_requests']
                for model_id, count in sorted(stats_data['model_usage'].items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total * 100) if total > 0 else 0
                    table.add_row(model_id, str(count), f"{percentage:.1f}%")
                
                console.print(table)
            
            if stats_data['available_models']:
                console.print(f"\n[green]Available models:[/green] {', '.join(stats_data['available_models'])}")
    
    asyncio.run(show_stats())


@cli.command()
@click.argument('queries', nargs=-1, required=True)
@click.option('--verbose', '-v', is_flag=True, help='Show detailed analysis')
@click.pass_obj
def analyze(config: Config, queries: tuple, verbose: bool) -> None:
    """Analyze queries without sending them to models."""
    from .analyzer import QueryAnalyzer
    
    analyzer = QueryAnalyzer(config)
    
    for query_text in queries:
        analysis = analyzer.analyze(query_text)
        
        console.print(Panel(
            f"[bold]Query:[/bold] {query_text}\n\n"
            f"[bold]Primary Type:[/bold] {analysis.primary_type.value}\n"
            f"[bold]Confidence:[/bold] {analysis.confidence:.1%}\n"
            f"[bold]Complexity:[/bold] {analysis.complexity.value}",
            title="🔍 Query Analysis",
            border_style="blue"
        ))
        
        if verbose:
            if analysis.secondary_types:
                console.print(f"[bold]Secondary Types:[/bold] {', '.join(t.value for t in analysis.secondary_types)}")
            
            if analysis.matched_keywords:
                console.print(f"[bold]Matched Keywords:[/bold] {', '.join(analysis.matched_keywords)}")
            
            if analysis.suggested_models:
                console.print(f"[bold]Suggested Models:[/bold] {', '.join(analysis.suggested_models)}")
            
            # Show features
            features = analysis.features
            feature_info = [
                f"Length: {features.length} chars",
                f"Words: {features.word_count}",
                f"Sentences: {features.sentence_count}",
                f"Questions: {features.question_count}"
            ]
            
            if features.programming_languages:
                feature_info.append(f"Languages: {', '.join(features.programming_languages)}")
            
            console.print(f"[dim]{' | '.join(feature_info)}[/dim]")
        
        console.print()


@cli.command()
@click.option('--model', '-m', help='Test specific model')
@click.pass_obj
def test(config: Config, model: Optional[str]) -> None:
    """Test model connectivity and basic functionality."""
    
    async def run_test():
        async with ModelManager(config=config) as manager:
            if model:
                models_to_test = [model]
            else:
                models_to_test = manager.get_available_models()
                if not models_to_test:
                    console.print("[yellow]No models available to test. Try running 'freerouter models --refresh' first.[/yellow]")
                    return
            
            console.print(f"Testing {len(models_to_test)} model(s)...")
            
            results = []
            for model_id in models_to_test:
                with Progress(
                    SpinnerColumn(),
                    TextColumn(f"Testing {model_id}..."),
                    console=console,
                    transient=True
                ) as progress:
                    task = progress.add_task("", total=None)
                    success = await manager.test_model(model_id)
                    results.append((model_id, success))
            
            # Show results
            table = Table(title="🧪 Test Results")
            table.add_column("Model", style="cyan")
            table.add_column("Status", style="green")
            
            for model_id, success in results:
                status = "[green]✅ Working[/green]" if success else "[red]❌ Failed[/red]"
                table.add_row(model_id, status)
            
            console.print(table)
            
            working_count = sum(1 for _, success in results if success)
            console.print(f"\n[bold]{working_count}/{len(results)} models are working properly[/bold]")
    
    asyncio.run(run_test())


@cli.command()
@click.pass_obj
def version(config: Config) -> None:
    """Show version information."""
    from . import __version__
    
    console.print(Panel(
        f"[bold]FreeRouter[/bold] version [green]{__version__}[/green]\n\n"
        f"Intelligent Model Router for OpenRouter Free Models\n"
        f"Configuration loaded from: {config.__class__.__name__}",
        title="ℹ️  Version Info",
        border_style="blue"
    ))


def main() -> None:
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


if __name__ == '__main__':
    main()