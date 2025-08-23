# FreeRouter

**Model routing for resource-constrained developers using free models from OpenRouter**

FreeRouter democratizes intelligent model routing by leveraging OpenRouter's 50+ free models, combining lightweight algorithms with practical deployment options tailored for developers with limited resources.

## 🚀 Quick Start

### Installation

```bash
pip install freerouter
```

### Basic Usage

```python
from freerouter import Controller

# Initialize with your OpenRouter API key
controller = Controller(
    api_key="your-openrouter-api-key",
    precision_threshold=0.3  # Higher = faster, lower = more accurate
)

# OpenAI-compatible interface
response = await controller.chat_completions_create(
    messages=[{"role": "user", "content": "Write a Python function to calculate fibonacci"}]
)

print(response["choices"][0]["message"]["content"])
```

### CLI Usage

```bash
# Set your API key
export OPENROUTER_API_KEY="your-key-here"

# Send a chat request
freerouter chat "Write a function to sort an array in Python"

# List available models
freerouter models

# Check status
freerouter status

# Test routing (without API call)
freerouter route "Debug this JavaScript code"
```

## 🎯 Features

### Smart Routing
- **Semantic Intent Classification**: Automatically detects coding, mathematics, writing, analysis, and general queries
- **Rate Limit Handling**: Intelligent management of OpenRouter's free tier limits (20 RPM, 50-1000 RPD)
- **Fallback Logic**: Graceful degradation when primary models are rate-limited
- **Threshold Control**: Configurable precision-speed tradeoffs

### Free Model Pool
FreeRouter includes optimized routing for OpenRouter's best free models:

- **DeepSeek R1** (1T+ params) - Coding, Analysis, Mathematics
- **Kimi K2** (1T params, 200K context) - Writing, Analysis  
- **Gemma 2 9B IT** - General, Writing
- **Llama 3 8B Instruct** - General, Coding
- **Phi-3 Medium 128K** - Coding, Mathematics

### Developer Experience
- **OpenAI Compatible**: Drop-in replacement requiring only base URL changes
- **Real-time Statistics**: Track routing decisions, success rates, and intent distribution
- **Streaming Support**: Full streaming response support with routing information
- **CLI Interface**: Complete command-line tool for testing and development

## 📋 Requirements

- Python 3.9+
- OpenRouter API key (free tier supported)

## 🏗️ Architecture

FreeRouter uses a **semantic routing** approach for Phase 1:

1. **Intent Classification**: Pattern matching and keyword analysis to classify queries
2. **Model Selection**: Route to specialized models based on detected intent
3. **Rate Limit Management**: Track usage across all models with intelligent fallbacks
4. **Quality Monitoring**: Collect routing statistics and success metrics

Future phases will add Matrix Factorization and Random Forest routing algorithms.

## 🔧 Configuration

### Environment Variables

```bash
OPENROUTER_API_KEY=your-openrouter-api-key
```

### Advanced Configuration

```python
controller = Controller(
    api_key="your-key",
    routing_model="semantic",  # Future: "mf", "rf"
    precision_threshold=0.3,   # 0.1 (accurate) to 0.9 (fast)
    enable_fallbacks=True,
    max_fallbacks=3,
    rate_limit_buffer=0.1      # Keep 10% buffer
)
```

## 📊 Monitoring

### Get Routing Statistics

```python
stats = controller.get_stats()
print(f"Success rate: {stats['success_rate']:.2%}")
print(f"Intent distribution: {stats['intent_distribution']}")
```

### CLI Statistics

```bash
freerouter chat --stats "Your query here"
```

## 🧪 Development

### Setup

```bash
git clone https://github.com/freerouter/freerouter
cd freerouter
poetry install
```

### Testing

```bash
poetry run pytest
poetry run pytest --cov=freerouter
```

### Linting

```bash
poetry run black freerouter/ tests/
poetry run isort freerouter/ tests/
poetry run flake8 freerouter/ tests/
poetry run mypy freerouter/
```

## 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Roadmap

**Phase 1: Foundation** ✅
- [x] Semantic routing with intent classification
- [x] OpenRouter integration with rate limiting
- [x] OpenAI-compatible API
- [x] CLI interface

**Phase 2: Intelligence** (Coming Soon)
- [ ] Matrix Factorization routing
- [ ] Random Forest fallback routing
- [ ] Performance monitoring and feedback collection
- [ ] HuggingFace model distribution

**Phase 3: Scale** (Future)
- [ ] Community evaluation framework
- [ ] API service deployment
- [ ] Adaptive learning from usage patterns
- [ ] Advanced analytics and optimization

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [OpenRouter](https://openrouter.ai) for providing free model access
- [RouteLLM](https://github.com/lm-sys/RouteLLM) for routing algorithm research
- The open-source community for model development and research

---

**Made for developers who need intelligent routing without breaking the bank** 💪