"""Matrix Factorization router implementation for FreeRouter Phase 2."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer

from .models import RouteDecision, IntentType, FREE_MODELS
from .exceptions import RoutingError


@dataclass
class TrainingExample:
    """Training example for matrix factorization."""
    query: str
    model_id: str
    correctness: float  # 0.0 to 1.0, where 1.0 means perfect answer
    response_time: float  # seconds
    user_rating: Optional[float] = None  # Optional user feedback


class RoutingDataset(Dataset):
    """Dataset for training the matrix factorization router."""
    
    def __init__(self, examples: List[TrainingExample], query_embeddings: np.ndarray, 
                 model_to_idx: Dict[str, int]):
        self.examples = examples
        self.query_embeddings = query_embeddings
        self.model_to_idx = model_to_idx
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[idx]
        query_embedding = torch.FloatTensor(self.query_embeddings[idx])
        model_idx = torch.LongTensor([self.model_to_idx[example.model_id]])
        
        # Combine correctness and response time for target score
        # Higher correctness = better, lower response time = better
        time_penalty = min(example.response_time / 10.0, 1.0)  # Normalize to 0-1
        target = example.correctness * (1.0 - time_penalty * 0.3)  # 30% time penalty
        target_score = torch.FloatTensor([target])
        
        return query_embedding, model_idx, target_score


class MatrixFactorizationModel(nn.Module):
    """Matrix factorization model for routing decisions.
    
    Architecture: S(M_i, Q_j) = M_i^T W Q_j
    where M_i are model embeddings and Q_j are query embeddings.
    """
    
    def __init__(self, query_dim: int, num_models: int, embedding_dim: int = 128):
        super().__init__()
        
        self.query_dim = query_dim
        self.num_models = num_models
        self.embedding_dim = embedding_dim
        
        # Query encoder: maps query embeddings to routing space
        self.query_encoder = nn.Sequential(
            nn.Linear(query_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Model embeddings: learned representations for each model
        self.model_embeddings = nn.Embedding(num_models, embedding_dim)
        
        # Interaction matrix W for computing scores
        self.interaction_matrix = nn.Linear(embedding_dim, embedding_dim, bias=False)
        
        # Initialize embeddings
        nn.init.normal_(self.model_embeddings.weight, std=0.1)
        nn.init.xavier_uniform_(self.interaction_matrix.weight)
    
    def forward(self, query_embeddings: torch.Tensor, model_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute routing scores."""
        # Encode queries to routing space
        query_encoded = self.query_encoder(query_embeddings)  # (batch, embedding_dim)
        
        # Get model embeddings
        model_embeds = self.model_embeddings(model_indices.squeeze(-1))  # (batch, embedding_dim)
        
        # Apply interaction matrix
        query_transformed = self.interaction_matrix(query_encoded)  # (batch, embedding_dim)
        
        # Compute scores: dot product between transformed query and model embeddings
        scores = torch.sum(query_transformed * model_embeds, dim=1, keepdim=True)  # (batch, 1)
        
        return torch.sigmoid(scores)  # Normalize to 0-1
    
    def predict_all_models(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Predict scores for all models given a query."""
        batch_size = query_embedding.size(0)
        
        # Encode query
        query_encoded = self.query_encoder(query_embedding)  # (batch, embedding_dim)
        query_transformed = self.interaction_matrix(query_encoded)  # (batch, embedding_dim)
        
        # Get all model embeddings
        all_model_indices = torch.arange(self.num_models).unsqueeze(0).expand(batch_size, -1)  # (batch, num_models)
        all_model_embeds = self.model_embeddings(all_model_indices)  # (batch, num_models, embedding_dim)
        
        # Compute scores for all models
        query_expanded = query_transformed.unsqueeze(1).expand(-1, self.num_models, -1)  # (batch, num_models, embedding_dim)
        scores = torch.sum(query_expanded * all_model_embeds, dim=2)  # (batch, num_models)
        
        return torch.sigmoid(scores)


class MatrixFactorizationRouter:
    """Matrix factorization router for intelligent model routing."""
    
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.mf_model = None
        self.model_to_idx = {}
        self.idx_to_model = {}
        self.query_dim = None
        
        # Training data collection
        self.training_examples: List[TrainingExample] = []
        
        # Performance tracking
        self.routing_stats = {
            "predictions_made": 0,
            "avg_confidence": 0.0,
            "model_usage": {model_id: 0 for model_id in FREE_MODELS.keys()}
        }
    
    def _initialize_embeddings(self):
        """Initialize the sentence transformer model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            # Get embedding dimension from a test sentence
            test_embedding = self.embedding_model.encode(["test"])
            self.query_dim = test_embedding.shape[1]
    
    def _prepare_model_mappings(self):
        """Prepare model ID to index mappings."""
        model_ids = list(FREE_MODELS.keys())
        self.model_to_idx = {model_id: idx for idx, model_id in enumerate(model_ids)}
        self.idx_to_model = {idx: model_id for model_id, idx in self.model_to_idx.items()}
    
    def add_training_example(self, query: str, model_id: str, correctness: float, 
                           response_time: float, user_rating: Optional[float] = None):
        """Add a training example for the router."""
        example = TrainingExample(
            query=query,
            model_id=model_id,
            correctness=correctness,
            response_time=response_time,
            user_rating=user_rating
        )
        self.training_examples.append(example)
    
    def train(self, epochs: int = 50, batch_size: int = 32, learning_rate: float = 0.001,
             validation_split: float = 0.2) -> Dict[str, float]:
        """Train the matrix factorization model."""
        if len(self.training_examples) < 10:
            raise RoutingError("Need at least 10 training examples to train MF router")
        
        self._initialize_embeddings()
        self._prepare_model_mappings()
        
        # Generate query embeddings
        queries = [ex.query for ex in self.training_examples]
        query_embeddings = self.embedding_model.encode(queries)
        
        # Split data
        n_train = int(len(self.training_examples) * (1 - validation_split))
        train_examples = self.training_examples[:n_train]
        train_embeddings = query_embeddings[:n_train]
        val_examples = self.training_examples[n_train:]
        val_embeddings = query_embeddings[n_train:]
        
        # Create datasets
        train_dataset = RoutingDataset(train_examples, train_embeddings, self.model_to_idx)
        val_dataset = RoutingDataset(val_examples, val_embeddings, self.model_to_idx)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize model
        self.mf_model = MatrixFactorizationModel(
            query_dim=self.query_dim,
            num_models=len(FREE_MODELS),
            embedding_dim=128
        )
        
        # Training setup
        optimizer = optim.Adam(self.mf_model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training
            self.mf_model.train()
            train_loss = 0.0
            
            for query_emb, model_idx, target in train_loader:
                optimizer.zero_grad()
                predictions = self.mf_model(query_emb, model_idx)
                loss = criterion(predictions, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            self.mf_model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for query_emb, model_idx, target in val_loader:
                    predictions = self.mf_model(query_emb, target)
                    loss = criterion(predictions, target)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        return {
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "epochs_trained": epochs
        }
    
    def route_query(self, query: str, available_models: List[str], 
                   threshold: float = 0.5) -> Tuple[str, float, str]:
        """Route a query to the best available model."""
        if self.mf_model is None:
            raise RoutingError("Model not trained. Call train() first or load a pre-trained model.")
        
        self._initialize_embeddings()
        
        # Get query embedding
        query_embedding = self.embedding_model.encode([query])
        query_tensor = torch.FloatTensor(query_embedding)
        
        # Get predictions for all models
        self.mf_model.eval()
        with torch.no_grad():
            all_scores = self.mf_model.predict_all_models(query_tensor)
            scores = all_scores[0].numpy()  # Get first (and only) batch item
        
        # Filter to available models and sort by score
        available_scores = []
        for model_id in available_models:
            if model_id in self.model_to_idx:
                idx = self.model_to_idx[model_id]
                score = scores[idx]
                available_scores.append((model_id, score))
        
        if not available_scores:
            raise RoutingError("No available models found in trained model set")
        
        # Sort by score (higher is better)
        available_scores.sort(key=lambda x: x[1], reverse=True)
        
        best_model, best_score = available_scores[0]
        
        # Update statistics
        self.routing_stats["predictions_made"] += 1
        self.routing_stats["avg_confidence"] = (
            (self.routing_stats["avg_confidence"] * (self.routing_stats["predictions_made"] - 1) + best_score) /
            self.routing_stats["predictions_made"]
        )
        self.routing_stats["model_usage"][best_model] += 1
        
        reasoning = f"Matrix factorization predicted {best_model} with score {best_score:.3f}"
        
        return best_model, float(best_score), reasoning
    
    def save_model(self, filepath: str):
        """Save the trained model and metadata."""
        if self.mf_model is None:
            raise RoutingError("No model to save")
        
        save_data = {
            "model_state_dict": self.mf_model.state_dict(),
            "model_config": {
                "query_dim": self.query_dim,
                "num_models": len(FREE_MODELS),
                "embedding_dim": 128
            },
            "model_mappings": {
                "model_to_idx": self.model_to_idx,
                "idx_to_model": self.idx_to_model
            },
            "embedding_model_name": self.embedding_model_name,
            "training_examples": len(self.training_examples),
            "routing_stats": self.routing_stats
        }
        
        torch.save(save_data, filepath)
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        save_data = torch.load(filepath, map_location="cpu")
        
        # Restore configuration
        self.embedding_model_name = save_data["embedding_model_name"]
        self.query_dim = save_data["model_config"]["query_dim"]
        self.model_to_idx = save_data["model_mappings"]["model_to_idx"]
        self.idx_to_model = save_data["model_mappings"]["idx_to_model"]
        self.routing_stats = save_data["routing_stats"]
        
        # Initialize model
        self.mf_model = MatrixFactorizationModel(**save_data["model_config"])
        self.mf_model.load_state_dict(save_data["model_state_dict"])
        self.mf_model.eval()
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            **self.routing_stats,
            "training_examples": len(self.training_examples),
            "model_trained": self.mf_model is not None
        }