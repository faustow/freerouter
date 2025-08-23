"""HuggingFace integration for model distribution and inference endpoints."""

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import asyncio
import aiofiles

from huggingface_hub import HfApi, Repository, upload_file, snapshot_download
from huggingface_hub.inference_api import InferenceApi
import torch

from .matrix_factorization_router import MatrixFactorizationRouter
from .random_forest_router import RandomForestRouter
from .performance_monitor import PerformanceMonitor
from .exceptions import FreeRouterError


class HuggingFaceIntegration:
    """Integration with HuggingFace Hub for model distribution and inference."""
    
    def __init__(self, hf_token: Optional[str] = None, organization: str = "freerouter"):
        self.hf_token = hf_token
        self.organization = organization
        self.api = HfApi(token=hf_token) if hf_token else None
        
        # Model registry
        self.available_models = {
            "mf-free-v1": {
                "type": "matrix_factorization", 
                "repo": f"{organization}/mf-router-free-v1",
                "description": "Matrix factorization router for OpenRouter free models",
                "specialization": ["coding", "mathematics", "analysis"]
            },
            "rf-free-v1": {
                "type": "random_forest",
                "repo": f"{organization}/rf-router-free-v1", 
                "description": "Random forest router for OpenRouter free models",
                "specialization": ["general", "writing", "coding"]
            },
            "hybrid-free-v1": {
                "type": "hybrid",
                "repo": f"{organization}/hybrid-router-free-v1",
                "description": "Hybrid MF+RF router with semantic fallback",
                "specialization": ["all"]
            }
        }
    
    async def upload_mf_model(
        self, 
        mf_router: MatrixFactorizationRouter,
        model_name: str = "mf-free-v1",
        commit_message: str = "Upload Matrix Factorization router"
    ) -> str:
        """Upload a trained Matrix Factorization model to HuggingFace."""
        if self.api is None:
            raise FreeRouterError("HuggingFace token required for model upload")
        
        repo_id = self.available_models[model_name]["repo"]
        
        # Save model to temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "mf_model.pt"
            mf_router.save_model(str(model_path))
            
            # Create model card
            model_card = self._generate_mf_model_card(mf_router, model_name)
            card_path = Path(temp_dir) / "README.md"
            
            async with aiofiles.open(card_path, 'w') as f:
                await f.write(model_card)
            
            # Create config file
            config = {
                "model_type": "matrix_factorization",
                "freerouter_version": "0.2.0",
                "embedding_model": mf_router.embedding_model_name,
                "query_dim": mf_router.query_dim,
                "num_models": len(mf_router.model_to_idx),
                "training_examples": len(mf_router.training_examples),
                "stats": mf_router.get_stats()
            }
            
            config_path = Path(temp_dir) / "config.json"
            async with aiofiles.open(config_path, 'w') as f:
                await f.write(json.dumps(config, indent=2))
            
            # Upload files
            try:
                # Create repository if it doesn't exist
                self.api.create_repo(repo_id, exist_ok=True, repo_type="model")
                
                # Upload model file
                upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo="mf_model.pt",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=commit_message
                )
                
                # Upload config
                upload_file(
                    path_or_fileobj=str(config_path),
                    path_in_repo="config.json", 
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message="Add model config"
                )
                
                # Upload model card
                upload_file(
                    path_or_fileobj=str(card_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message="Add model card"
                )
                
                return f"https://huggingface.co/{repo_id}"
                
            except Exception as e:
                raise FreeRouterError(f"Failed to upload MF model: {str(e)}")
    
    async def upload_rf_model(
        self,
        rf_router: RandomForestRouter,
        model_name: str = "rf-free-v1",
        commit_message: str = "Upload Random Forest router"
    ) -> str:
        """Upload a trained Random Forest model to HuggingFace."""
        if self.api is None:
            raise FreeRouterError("HuggingFace token required for model upload")
        
        repo_id = self.available_models[model_name]["repo"]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "rf_model.pkl"
            rf_router.save_model(str(model_path))
            
            # Create model card
            model_card = self._generate_rf_model_card(rf_router, model_name)
            card_path = Path(temp_dir) / "README.md"
            
            async with aiofiles.open(card_path, 'w') as f:
                await f.write(model_card)
            
            # Create config
            config = {
                "model_type": "random_forest",
                "freerouter_version": "0.2.0",
                "embedding_model": rf_router.embedding_model_name,
                "feature_dim": rf_router.feature_dim,
                "rf_params": rf_router.rf_params,
                "training_examples": len(rf_router.training_examples),
                "stats": rf_router.get_stats()
            }
            
            config_path = Path(temp_dir) / "config.json"
            async with aiofiles.open(config_path, 'w') as f:
                await f.write(json.dumps(config, indent=2))
            
            try:
                self.api.create_repo(repo_id, exist_ok=True, repo_type="model")
                
                upload_file(
                    path_or_fileobj=str(model_path),
                    path_in_repo="rf_model.pkl",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message=commit_message
                )
                
                upload_file(
                    path_or_fileobj=str(config_path),
                    path_in_repo="config.json",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message="Add model config"
                )
                
                upload_file(
                    path_or_fileobj=str(card_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message="Add model card"
                )
                
                return f"https://huggingface.co/{repo_id}"
                
            except Exception as e:
                raise FreeRouterError(f"Failed to upload RF model: {str(e)}")
    
    async def download_mf_model(self, model_name: str = "mf-free-v1") -> MatrixFactorizationRouter:
        """Download and load a Matrix Factorization model from HuggingFace."""
        if model_name not in self.available_models:
            raise FreeRouterError(f"Unknown model: {model_name}")
        
        repo_id = self.available_models[model_name]["repo"]
        
        try:
            # Download model files
            model_dir = snapshot_download(
                repo_id=repo_id,
                token=self.hf_token,
                allow_patterns=["*.pt", "*.json"]
            )
            
            # Load model
            mf_router = MatrixFactorizationRouter()
            model_path = Path(model_dir) / "mf_model.pt"
            mf_router.load_model(str(model_path))
            
            return mf_router
            
        except Exception as e:
            raise FreeRouterError(f"Failed to download MF model {model_name}: {str(e)}")
    
    async def download_rf_model(self, model_name: str = "rf-free-v1") -> RandomForestRouter:
        """Download and load a Random Forest model from HuggingFace."""
        if model_name not in self.available_models:
            raise FreeRouterError(f"Unknown model: {model_name}")
        
        repo_id = self.available_models[model_name]["repo"]
        
        try:
            model_dir = snapshot_download(
                repo_id=repo_id,
                token=self.hf_token,
                allow_patterns=["*.pkl", "*.json"]
            )
            
            rf_router = RandomForestRouter()
            model_path = Path(model_dir) / "rf_model.pkl"
            rf_router.load_model(str(model_path))
            
            return rf_router
            
        except Exception as e:
            raise FreeRouterError(f"Failed to download RF model {model_name}: {str(e)}")
    
    def _generate_mf_model_card(self, mf_router: MatrixFactorizationRouter, model_name: str) -> str:
        """Generate a model card for Matrix Factorization router."""
        stats = mf_router.get_stats()
        
        return f"""---
library_name: freerouter
tags:
- routing
- llm
- openrouter
- matrix-factorization
license: mit
model-index:
- name: {model_name}
  results:
  - task:
      type: model-routing
      name: Model Routing
    dataset:
      type: freerouter-feedback
      name: FreeRouter User Feedback
    metrics:
    - type: accuracy
      value: {stats.get('avg_confidence', 0.0):.3f}
---

# {self.available_models[model_name]['description']}

This is a Matrix Factorization router trained for FreeRouter, designed to intelligently route queries to the best available free models from OpenRouter.

## Model Details

- **Model Type**: Matrix Factorization Router
- **Framework**: PyTorch
- **Embedding Model**: {mf_router.embedding_model_name}
- **Query Dimension**: {mf_router.query_dim}
- **Training Examples**: {len(mf_router.training_examples)}
- **Specialization**: {', '.join(self.available_models[model_name]['specialization'])}

## Performance

- **Predictions Made**: {stats.get('predictions_made', 0)}
- **Average Confidence**: {stats.get('avg_confidence', 0.0):.3f}
- **Model Usage Distribution**: {stats.get('model_usage', {{}})}

## Usage

```python
from freerouter.huggingface_integration import HuggingFaceIntegration
from freerouter import Controller

# Download and load the model
hf_integration = HuggingFaceIntegration()
mf_router = await hf_integration.download_mf_model("{model_name}")

# Use with FreeRouter Controller
controller = Controller(
    api_key="your-openrouter-key",
    routing_model="mf",
    mf_router=mf_router
)
```

## Training Data

This model was trained on user feedback and performance data collected from FreeRouter usage, focusing on:
- Query-model correctness pairs
- Response time optimization
- User satisfaction ratings
- Intent-based specialization

## Limitations

- Optimized specifically for OpenRouter's free model tier
- Performance depends on the quality and diversity of training feedback
- May not generalize well to models outside the training set

## Citation

```bibtex
@software{{freerouter2024,
  title={{FreeRouter: Model routing for resource-constrained developers}},
  author={{FreeRouter Team}},
  year={{2024}},
  url={{https://github.com/freerouter/freerouter}}
}}
```
"""
    
    def _generate_rf_model_card(self, rf_router: RandomForestRouter, model_name: str) -> str:
        """Generate a model card for Random Forest router."""
        stats = rf_router.get_stats()
        
        return f"""---
library_name: freerouter
tags:
- routing
- llm
- openrouter
- random-forest
- scikit-learn
license: mit
model-index:
- name: {model_name}
  results:
  - task:
      type: model-routing
      name: Model Routing
    dataset:
      type: freerouter-feedback
      name: FreeRouter User Feedback
    metrics:
    - type: accuracy
      value: {stats.get('accuracy', 0.0):.3f}
---

# {self.available_models[model_name]['description']}

This is a Random Forest router trained for FreeRouter, providing interpretable routing decisions with confidence scores for OpenRouter's free model ecosystem.

## Model Details

- **Model Type**: Random Forest Router
- **Framework**: scikit-learn
- **Embedding Model**: {rf_router.embedding_model_name}
- **Feature Dimension**: {rf_router.feature_dim}
- **RF Parameters**: {rf_router.rf_params}
- **Training Examples**: {len(rf_router.training_examples)}
- **Specialization**: {', '.join(self.available_models[model_name]['specialization'])}

## Performance

- **Predictions Made**: {stats.get('predictions_made', 0)}
- **Accuracy**: {stats.get('accuracy', 0.0):.3f}
- **Correct Predictions**: {stats.get('correct_predictions', 0)}
- **Confidence Distribution**: {stats.get('confidence_distribution', {{}})}

## Features

This model uses a combination of semantic embeddings and manual features:
- **Semantic Features**: {rf_router.feature_dim} dimensions from sentence transformers
- **Manual Features**: Query length, word count, content type detection, question patterns
- **Intent Detection**: Coding, mathematics, writing, analysis patterns
- **Complexity Metrics**: Parentheses, quotes, long words, structural patterns

## Usage

```python
from freerouter.huggingface_integration import HuggingFaceIntegration
from freerouter import Controller

# Download and load the model
hf_integration = HuggingFaceIntegration()
rf_router = await hf_integration.download_rf_model("{model_name}")

# Use with FreeRouter Controller
controller = Controller(
    api_key="your-openrouter-key",
    routing_model="rf",
    rf_router=rf_router
)

# Get routing preferences for a query
preferences = rf_router.get_model_preferences("Write a Python function")
print(preferences)
```

## Training Data

- **Success-based Learning**: Trained on binary success/failure feedback
- **Threshold-based Routing**: Supports configurable confidence thresholds
- **Interpretable Decisions**: Feature importances available for debugging
- **Continuous Learning**: Supports online updates with new feedback

## Limitations

- Binary classification approach (success/failure) vs. continuous scoring
- Requires sufficient training data for each model in the pool
- Performance varies with query similarity to training distribution
- Manual features may need tuning for new model types

## Citation

```bibtex
@software{{freerouter2024,
  title={{FreeRouter: Model routing for resource-constrained developers}},
  author={{FreeRouter Team}}, 
  year={{2024}},
  url={{https://github.com/freerouter/freerouter}}
}}
```
"""
    
    async def create_inference_endpoint(
        self,
        model_name: str,
        endpoint_name: Optional[str] = None,
        accelerator: str = "cpu",
        instance_size: str = "small"
    ) -> str:
        """Create a HuggingFace Inference Endpoint for the router."""
        if self.api is None:
            raise FreeRouterError("HuggingFace token required for inference endpoints")
        
        repo_id = self.available_models[model_name]["repo"]
        endpoint_name = endpoint_name or f"freerouter-{model_name}"
        
        try:
            # This would use the HuggingFace Inference Endpoints API
            # Currently a placeholder for the actual implementation
            endpoint_config = {
                "repository": repo_id,
                "framework": "custom",
                "task": "other",
                "accelerator": accelerator,
                "instance_size": instance_size,
                "min_replica": 0,
                "max_replica": 1
            }
            
            # In a real implementation, this would create the endpoint
            # For now, return a placeholder URL
            return f"https://api-inference.huggingface.co/models/{repo_id}"
            
        except Exception as e:
            raise FreeRouterError(f"Failed to create inference endpoint: {str(e)}")
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available pre-trained router models."""
        return self.available_models.copy()
    
    async def upload_performance_data(
        self,
        performance_monitor: PerformanceMonitor,
        dataset_name: str = "freerouter-feedback-v1"
    ) -> str:
        """Upload anonymized performance data for community training."""
        if self.api is None:
            raise FreeRouterError("HuggingFace token required for data upload")
        
        repo_id = f"{self.organization}/{dataset_name}"
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Export anonymized data
            data_path = Path(temp_dir) / "feedback_data.json"
            performance_monitor.export_data(str(data_path))
            
            # Create dataset card
            dataset_card = self._generate_dataset_card(performance_monitor, dataset_name)
            card_path = Path(temp_dir) / "README.md"
            
            async with aiofiles.open(card_path, 'w') as f:
                await f.write(dataset_card)
            
            try:
                self.api.create_repo(repo_id, exist_ok=True, repo_type="dataset")
                
                upload_file(
                    path_or_fileobj=str(data_path),
                    path_in_repo="feedback_data.json",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message="Upload community feedback data"
                )
                
                upload_file(
                    path_or_fileobj=str(card_path),
                    path_in_repo="README.md",
                    repo_id=repo_id,
                    token=self.hf_token,
                    commit_message="Add dataset card"
                )
                
                return f"https://huggingface.co/datasets/{repo_id}"
                
            except Exception as e:
                raise FreeRouterError(f"Failed to upload performance data: {str(e)}")
    
    def _generate_dataset_card(self, monitor: PerformanceMonitor, dataset_name: str) -> str:
        """Generate a dataset card for performance data."""
        stats = {
            "total_requests": monitor.stats["total_requests"],
            "model_count": len(monitor.stats["model_performance"]),
            "intent_count": len(monitor.stats["intent_performance"])
        }
        
        return f"""---
library_name: freerouter
tags:
- routing
- llm
- feedback
- openrouter
license: mit
size_categories:
- 1K<n<10K
task_categories:
- other
---

# {dataset_name.title().replace('-', ' ')}

Community-contributed feedback data for training FreeRouter model routing algorithms.

## Dataset Details

- **Total Requests**: {stats['total_requests']}
- **Models Covered**: {stats['model_count']}
- **Intent Categories**: {stats['intent_count']}
- **Collection Period**: User feedback from FreeRouter deployments
- **Privacy**: All personally identifiable information removed

## Structure

```json
{{
  "performance_metrics": [
    {{
      "query_hash": "abc123...",
      "query": "Write a Python function...",
      "model_id": "deepseek/deepseek-r1",
      "intent": "coding",
      "response_time": 2.5,
      "token_count": 150,
      "user_rating": 4.2,
      "correctness_score": 0.85,
      "timestamp": 1699999999
    }}
  ],
  "routing_decisions": [...],
  "aggregated_stats": {{...}}
}}
```

## Usage

```python
from datasets import load_dataset
from freerouter.performance_monitor import PerformanceMonitor

# Load the dataset
dataset = load_dataset("freerouter/{dataset_name}")

# Use for training
monitor = PerformanceMonitor()
# ... process dataset and train models
```

## Citation

```bibtex
@dataset{{freerouter_feedback2024,
  title={{FreeRouter Community Feedback Dataset}},
  author={{FreeRouter Community}},
  year={{2024}},
  url={{https://huggingface.co/datasets/{self.organization}/{dataset_name}}}
}}
```
"""