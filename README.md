# FreeRouter 🚀

**Intelligent Model Router for OpenRouter Free Models**

FreeRouter is a production-ready Python application that intelligently routes user queries to the most appropriate free model available on OpenRouter. The system is completely transparent to users - they ask questions, and FreeRouter automatically selects the best model without requiring any technical knowledge.

## ✨ Features

### 🧠 Intelligent Query Analysis
- **Automatic Query Classification**: Detects query types (coding, creative writing, analysis, math, reasoning, conversation)
- **Complexity Assessment**: Evaluates query complexity to optimize model selection
- **Multi-type Support**: Handles queries with multiple characteristics

### 🎯 Smart Model Routing
- **Capability-Based Selection**: Routes queries based on model strengths and specialties
- **Performance Optimization**: Considers response time, success rate, and reliability
- **Fallback Mechanisms**: Graceful handling when preferred models are unavailable
- **Context-Aware**: Takes into account query complexity and context window requirements

### 📊 Comprehensive Evaluation
- **Automated Benchmarking**: Systematic evaluation framework for comparing models
- **Multi-Metric Assessment**: Accuracy, relevance, clarity, completeness, and efficiency scoring
- **Continuous Learning**: Performance tracking and routing improvement over time
- **Detailed Reporting**: Rich evaluation reports with rankings and insights

### 🔧 Production-Ready Architecture
- **Async-First Design**: Handle concurrent requests efficiently
- **Rate Limiting**: Respect OpenRouter API limits with intelligent backoff
- **Robust Error Handling**: Comprehensive exception handling and recovery
- **Configurable**: YAML-based configuration for models, evaluation criteria, and routing rules

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/freerouter.git
cd freerouter

# Install with poetry (recommended)
poetry install

# Or install with pip
pip install -e .
```

### Setup

1. **Get your OpenRouter API key** from [OpenRouter](https://openrouter.ai)

2. **Set your API key**:
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   
   # Or create a .env file
   cp .env.example .env
   # Edit .env and add your API key
   ```

3. **Test the installation**:
   ```bash
   freerouter ask "Hello, how are you?"
   ```

### Basic Usage

```bash
# Ask a question (automatic model selection)
freerouter ask "Write a Python function to calculate factorial"

# Get detailed routing explanation
freerouter ask "Explain machine learning" --explain

# Use a specific model preference
freerouter ask "Write a poem about nature" --model anthropic/claude-3-haiku

# Check available models
freerouter models

# Run model evaluation
freerouter evaluate --quick
```

## 📖 Usage Examples

### Command Line Interface

```bash
# Simple query
freerouter ask "What is the capital of France?"

# Complex coding query with explanation
freerouter ask "Implement a binary search tree in Python with insert, delete, and search methods" --explain

# Creative writing with temperature control
freerouter ask "Write a science fiction story about AI" --temperature 0.9

# Math problem
freerouter ask "Solve the quadratic equation x² - 5x + 6 = 0"

# Analysis query
freerouter ask "Compare the pros and cons of renewable energy sources"
```

### Python API

```python
import asyncio
from freerouter import FreeRouter

async def main():
    async with FreeRouter() as router:
        # Simple query routing
        result = await router.route_query("Write a function to reverse a string")
        print(f"Model used: {result.model_used}")
        print(f"Response: {result.response.message_content}")
        
        # Get model recommendations
        recommendations = await router.get_model_recommendations(
            "Create a machine learning model"
        )
        for model, confidence, reasoning in recommendations:
            print(f"{model}: {confidence:.1%} - {reasoning}")

asyncio.run(main())
```

### Model Management

```python
from freerouter import ModelManager, OpenRouterClient

async def manage_models():
    async with ModelManager() as manager:
        # Discover available models
        models = await manager.discover_models()
        print(f"Found {len(models)} models")
        
        # Test model availability
        for model_id in models:
            working = await manager.test_model(model_id)
            print(f"{model_id}: {'✅' if working else '❌'}")
        
        # Get best models for coding
        from freerouter.analyzer import QueryType
        best_coding_models = manager.get_best_models(QueryType.CODING)
        print(f"Best coding models: {best_coding_models}")
```

### Evaluation and Benchmarking

```python
from freerouter.evaluator import run_evaluation

async def benchmark_models():
    # Run comprehensive evaluation
    report = await run_evaluation()
    
    # Show results
    print(f"Evaluated {len(report.models_evaluated)} models")
    print(f"Total queries: {report.total_queries}")
    
    # Model rankings
    rankings = report.get_ranking()
    for i, (model, score) in enumerate(rankings[:5], 1):
        print(f"{i}. {model}: {score:.2f}/5.0")
```

## ⚙️ Configuration

FreeRouter uses YAML configuration files for customization:

### Model Configuration (`configs/models.yaml`)

```yaml
# Default settings for all models
default_config:
  timeout: 30
  max_retries: 3
  rate_limit:
    requests_per_minute: 20
    requests_per_hour: 200

# Model capability profiles (1-5 scale)
model_profiles:
  "openai/gpt-3.5-turbo":
    capabilities:
      coding: 4
      reasoning: 4
      creative_writing: 3
      analysis: 4
      conversation: 4
      math: 4
    specialties: ["general purpose", "coding"]
    context_window: 4096

# Query type mappings
query_types:
  coding:
    keywords: ["code", "function", "algorithm", "debug"]
    patterns: ["write.*function", "implement.*algorithm"]
    preferred_models: ["openai/gpt-3.5-turbo"]
```

### Evaluation Configuration (`configs/evaluation.yaml`)

```yaml
# Evaluation settings
evaluation_settings:
  evaluation_interval: 24  # hours
  min_queries_per_category: 10
  response_timeout: 30
  concurrent_evaluations: 3

# Test queries for benchmarking
test_queries:
  coding:
    - "Write a Python function to calculate factorial"
    - "Implement a binary search algorithm"
    # ... more queries

# Scoring criteria with weights
scoring_criteria:
  accuracy:
    weight: 0.3
    description: "How correct and factual is the response?"
  relevance:
    weight: 0.25
    description: "How well does the response address the query?"
```

## 🏗️ Architecture

FreeRouter is built with a modular, async-first architecture:

```
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│   Query Analyzer   │    │   Model Manager     │    │   Evaluation Engine │
│                     │    │                     │    │                     │
│ • Query classification │  │ • Model discovery   │    │ • Automated testing │
│ • Type detection    │    │ • Capability tracking │  │ • Performance metrics│
│ • Complexity assessment│  │ • Health monitoring │    │ • Comparative analysis│
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
           │                           │                           │
           └───────────────┬───────────────────────────────────────┘
                          │
                ┌─────────────────────┐
                │    Core Router      │
                │                     │
                │ • Intelligent routing│
                │ • Fallback handling │
                │ • Performance optimization│
                └─────────────────────┘
                          │
                ┌─────────────────────┐
                │  OpenRouter Client  │
                │                     │
                │ • API integration   │
                │ • Rate limiting     │
                │ • Error handling    │
                └─────────────────────┘
```

### Key Components

- **Query Analyzer**: Classifies queries by type, complexity, and characteristics
- **Model Manager**: Discovers, profiles, and monitors model performance
- **Core Router**: Implements intelligent routing logic with fallback mechanisms
- **Evaluation Engine**: Provides comprehensive model benchmarking and assessment
- **OpenRouter Client**: Handles API communication with rate limiting and error recovery

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=freerouter

# Run specific test files
poetry run pytest tests/test_router.py

# Run integration tests
poetry run pytest tests/test_integration.py
```

### Manual Testing

```bash
# Test model connectivity
freerouter test

# Test specific model
freerouter test --model anthropic/claude-3-haiku

# Quick benchmark
python scripts/benchmark.py --quick

# Full benchmark
python scripts/benchmark.py
```

## 📊 Monitoring and Analytics

### View Statistics

```bash
# Show routing statistics
freerouter stats

# Analyze queries without sending to models
freerouter analyze "Write a web scraper" --verbose

# List available models with performance metrics
freerouter models --test
```

### Performance Tracking

FreeRouter automatically tracks:
- **Model Success Rates**: How often models respond successfully
- **Response Times**: Average response time per model
- **Usage Patterns**: Which models are used most frequently
- **Error Rates**: Types and frequency of errors
- **Capability Scores**: Performance across different query types

## 🔧 Advanced Usage

### Custom Routing Strategies

```python
from freerouter.router import RoutingStrategy, RoutingDecision

class CustomStrategy(RoutingStrategy):
    def select_model(self, query_analysis, available_models, model_manager):
        # Implement custom routing logic
        return RoutingDecision(
            selected_model="preferred_model",
            confidence=0.9,
            reasoning="Custom routing logic"
        )

# Use custom strategy
router = FreeRouter()
router.routing_strategy = CustomStrategy()
```

### Batch Processing

```python
async def process_queries_batch(queries):
    async with FreeRouter() as router:
        results = []
        for query in queries:
            result = await router.route_query(query)
            results.append(result)
        return results
```

### Model Performance Analysis

```python
async def analyze_model_performance():
    async with ModelManager() as manager:
        for model_id, profile in manager.profiles.items():
            print(f"\nModel: {model_id}")
            print(f"Success Rate: {profile.performance.success_rate:.1%}")
            print(f"Avg Response Time: {profile.performance.average_response_time:.2f}s")
            
            for capability, score_obj in profile.capabilities.items():
                print(f"{capability}: {score_obj.score:.1f}/5.0")
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Check if API key is set
   echo $OPENROUTER_API_KEY
   
   # Test API connectivity
   freerouter test
   ```

2. **No Models Available**
   ```bash
   # Refresh model list
   freerouter models --refresh
   
   # Check model status
   freerouter models --test
   ```

3. **Rate Limiting**
   - Adjust rate limits in configuration
   - Use longer intervals between requests
   - Consider API key tier limits

4. **Poor Routing Decisions**
   ```bash
   # Run evaluation to check model performance
   freerouter evaluate --quick
   
   # Check routing explanation
   freerouter ask "your query" --explain
   ```

### Debug Mode

```bash
# Enable verbose logging
freerouter --verbose ask "your query"

# Check configuration
python -c "from freerouter.config import get_config; print(get_config().model_dump_json(indent=2))"
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/yourusername/freerouter.git
cd freerouter

# Install development dependencies
poetry install --with dev

# Run pre-commit hooks
poetry run pre-commit install

# Run tests
poetry run pytest
```

### Code Quality

We maintain high code quality standards:
- **Type Hints**: Full type annotation throughout
- **Testing**: >85% test coverage required
- **Linting**: Black, isort, flake8, mypy
- **Documentation**: Comprehensive docstrings and examples

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai) for providing access to free AI models
- The open-source community for inspiration and best practices
- Contributors and testers who help improve FreeRouter

## 📞 Support

- **Documentation**: Full documentation at [docs.freerouter.dev](https://docs.freerouter.dev)
- **Issues**: Report bugs on [GitHub Issues](https://github.com/yourusername/freerouter/issues)
- **Discussions**: Join discussions on [GitHub Discussions](https://github.com/yourusername/freerouter/discussions)
- **Email**: contact@freerouter.dev

---

**Made with ❤️ by the FreeRouter team**

*Democratizing access to AI models by removing the complexity of model selection*