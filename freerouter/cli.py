"""Command-line interface for FreeRouter."""

import asyncio
import json
import os
import sys
from typing import Optional

import click
from .controller import Controller
from .exceptions import FreeRouterError


@click.group()
@click.version_option(version="0.1.0", prog_name="freerouter")
def cli():
    """FreeRouter: Model routing for resource-constrained developers."""
    pass


@cli.command()
@click.option("--api-key", envvar="OPENROUTER_API_KEY", required=True,
              help="OpenRouter API key")
@click.option("--model", "-m", help="Preferred model (optional)")
@click.option("--precision", "-p", default=0.3, type=click.FloatRange(0.1, 0.9),
              help="Routing precision threshold (0.1-0.9)")
@click.option("--temperature", "-t", default=0.7, type=click.FloatRange(0.0, 2.0),
              help="Sampling temperature")
@click.option("--max-tokens", type=int, help="Maximum tokens in response")
@click.option("--stream", is_flag=True, help="Stream the response")
@click.option("--stats", is_flag=True, help="Show routing statistics after response")
@click.argument("prompt", required=False)
async def chat(
    api_key: str,
    model: Optional[str],
    precision: float,
    temperature: float,
    max_tokens: Optional[int],
    stream: bool,
    stats: bool,
    prompt: Optional[str]
):
    """Send a chat completion request."""
    if not prompt:
        prompt = click.prompt("Enter your prompt")
    
    controller = Controller(
        api_key=api_key,
        precision_threshold=precision
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        if stream:
            click.echo("Streaming response...\n", err=True)
            
            async for chunk in controller.chat_completions_create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            ):
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        click.echo(content, nl=False)
                
                # Show routing info from first chunk
                if "x_freerouter" in chunk:
                    click.echo(f"\n\n[Routed to: {chunk['x_freerouter']['routed_model']} "
                              f"({chunk['x_freerouter']['intent']})]\n", err=True)
        else:
            response = await controller.chat_completions_create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False
            )
            
            # Display routing info
            if "x_freerouter" in response:
                routing_info = response["x_freerouter"]
                click.echo(f"Routed to: {routing_info['routed_model']} "
                          f"({routing_info['intent']}, confidence: {routing_info['confidence']:.2f})\n",
                          err=True)
            
            # Display response
            if "choices" in response and response["choices"]:
                content = response["choices"][0]["message"]["content"]
                click.echo(content)
        
        # Show stats if requested
        if stats:
            stats_data = controller.get_stats()
            click.echo("\n" + "="*50, err=True)
            click.echo("FreeRouter Statistics:", err=True)
            click.echo(f"Total requests: {stats_data['total_requests']}", err=True)
            click.echo(f"Success rate: {stats_data['success_rate']:.2%}", err=True)
            click.echo(f"Fallback routes: {stats_data['fallback_routes']}", err=True)
            click.echo("Intent distribution:", err=True)
            for intent, count in stats_data['intent_distribution'].items():
                if count > 0:
                    click.echo(f"  {intent}: {count}", err=True)
    
    except FreeRouterError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    
    finally:
        await controller.close()


@cli.command()
@click.option("--api-key", envvar="OPENROUTER_API_KEY", required=True,
              help="OpenRouter API key")
def models(api_key: str):
    """List available models."""
    async def _list_models():
        controller = Controller(api_key=api_key)
        try:
            models_list = controller.list_models()
            
            click.echo("Available Models:")
            click.echo("="*80)
            
            for model in models_list:
                specialization = ", ".join(model["x_freerouter"]["specialization"])
                click.echo(f"ID: {model['id']}")
                click.echo(f"Provider: {model['owned_by']}")
                click.echo(f"Specialization: {specialization}")
                click.echo(f"Rate Limits: {model['x_freerouter']['rate_limit_rpm']} RPM, "
                          f"{model['x_freerouter']['rate_limit_rpd']} RPD")
                click.echo(f"Context: {model['x_freerouter']['context_length']} tokens")
                click.echo("-" * 40)
                
        finally:
            await controller.close()
    
    asyncio.run(_list_models())


@cli.command()
@click.option("--api-key", envvar="OPENROUTER_API_KEY", required=True,
              help="OpenRouter API key")
@click.option("--precision", "-p", default=0.3, type=click.FloatRange(0.1, 0.9),
              help="Routing precision threshold")
def status(api_key: str, precision: float):
    """Show FreeRouter status and available models."""
    async def _show_status():
        controller = Controller(
            api_key=api_key,
            precision_threshold=precision
        )
        try:
            stats = controller.get_stats()
            available = stats["available_models"]
            
            click.echo("FreeRouter Status")
            click.echo("="*50)
            click.echo(f"Available models: {len(available)}")
            click.echo(f"Precision threshold: {precision}")
            
            if available:
                click.echo("\nCurrently available models:")
                for model_id in available:
                    click.echo(f"  ✓ {model_id}")
            else:
                click.echo("\n⚠️  No models currently available (rate limited)")
            
            if stats["total_requests"] > 0:
                click.echo(f"\nSession Statistics:")
                click.echo(f"Total requests: {stats['total_requests']}")
                click.echo(f"Success rate: {stats['success_rate']:.2%}")
                
        finally:
            await controller.close()
    
    asyncio.run(_show_status())


@cli.command()
@click.argument("query")
def route(query: str):
    """Test routing decision for a query without making API call."""
    from .semantic_router import SemanticRouter
    from .models import FREE_MODELS
    
    router = SemanticRouter()
    intent, confidence = router.classify_intent(query)
    specialized_models = router.get_intent_models(intent)
    
    click.echo(f"Query: {query}")
    click.echo(f"Detected Intent: {intent.value} (confidence: {confidence:.2f})")
    click.echo(f"Specialized Models:")
    
    for model_id in specialized_models:
        if model_id in FREE_MODELS:
            model_info = FREE_MODELS[model_id]
            click.echo(f"  ✓ {model_id} ({model_info.name})")
    
    explanation = router.explain_routing(query, specialized_models[0] if specialized_models else "none", intent, confidence)
    click.echo(f"\nRouting Explanation: {explanation}")


def main():
    """Main entry point that handles async commands."""
    # Check for async commands
    if len(sys.argv) > 1 and sys.argv[1] in ['chat', 'models', 'status']:
        # Use click's async support
        cli(_anyio_backend="asyncio")
    else:
        cli()


if __name__ == "__main__":
    main()