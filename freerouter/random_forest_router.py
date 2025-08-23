"""Random Forest router implementation for FreeRouter Phase 2."""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer

from .models import FREE_MODELS
from .exceptions import RoutingError


@dataclass
class RFTrainingExample:
    """Training example for Random Forest router."""
    query: str
    model_id: str
    features: Optional[np.ndarray] = None  # Will be computed from query
    success: bool = True  # Whether this was a successful routing


class RandomForestRouter:
    """Random Forest router using query embeddings and success feedback.
    
    Based on RoRF (Routing on Random Forest) approach with threshold-based routing
    where P(A|x) = P(l=0|x) + P(l=1|x) enables fine-tuned cost-quality tradeoffs.
    """
    
    def __init__(
        self, 
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_depth: int = 20,
        n_estimators: int = 100,
        random_state: int = 42
    ):
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        
        # Random Forest parameters based on research
        self.rf_params = {
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "random_state": random_state,
            "n_jobs": -1  # Use all available cores
        }
        
        self.rf_model = None
        self.model_to_idx = {}
        self.idx_to_model = {}
        self.feature_dim = None
        
        # Training data
        self.training_examples: List[RFTrainingExample] = []
        
        # Performance tracking
        self.routing_stats = {
            "predictions_made": 0,
            "correct_predictions": 0,
            "accuracy": 0.0,
            "model_usage": {model_id: 0 for model_id in FREE_MODELS.keys()},
            "confidence_distribution": {"low": 0, "medium": 0, "high": 0}
        }
    
    def _initialize_embeddings(self):
        """Initialize the sentence transformer model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            # Get embedding dimension
            test_embedding = self.embedding_model.encode(["test"])
            self.feature_dim = test_embedding.shape[1]
    
    def _prepare_model_mappings(self):
        """Prepare model ID to index mappings."""
        model_ids = list(FREE_MODELS.keys())
        self.model_to_idx = {model_id: idx for idx, model_id in enumerate(model_ids)}
        self.idx_to_model = {idx: model_id for model_id, idx in self.model_to_idx.items()}
    
    def _extract_features(self, query: str) -> np.ndarray:
        """Extract features from a query using embeddings + manual features."""
        self._initialize_embeddings()
        
        # Get semantic embedding
        embedding = self.embedding_model.encode([query])[0]
        
        # Add manual features based on query characteristics
        manual_features = []
        
        # Length features
        manual_features.append(len(query))
        manual_features.append(len(query.split()))
        
        # Content type features
        manual_features.append(1.0 if any(kw in query.lower() for kw in ['def', 'function', 'class', 'import']) else 0.0)  # Code
        manual_features.append(1.0 if any(kw in query.lower() for kw in ['calculate', 'solve', 'equation', '=']) else 0.0)  # Math
        manual_features.append(1.0 if any(kw in query.lower() for kw in ['write', 'essay', 'story', 'creative']) else 0.0)  # Writing
        manual_features.append(1.0 if any(kw in query.lower() for kw in ['analyze', 'compare', 'evaluate']) else 0.0)  # Analysis
        
        # Question type features
        manual_features.append(1.0 if query.strip().endswith('?') else 0.0)  # Is question
        manual_features.append(1.0 if query.lower().startswith(('how', 'what', 'why', 'when', 'where')) else 0.0)  # Question word
        
        # Complexity features
        manual_features.append(query.count('('))  # Parentheses count
        manual_features.append(query.count('"') + query.count("'"))  # Quote count
        manual_features.append(len([w for w in query.split() if len(w) > 10]))  # Long words
        
        # Combine embedding with manual features
        combined_features = np.concatenate([embedding, np.array(manual_features)])
        
        return combined_features
    
    def add_training_example(self, query: str, model_id: str, success: bool = True):
        """Add a training example."""
        if model_id not in FREE_MODELS:
            raise RoutingError(f"Unknown model ID: {model_id}")
        
        example = RFTrainingExample(
            query=query,
            model_id=model_id,
            success=success
        )
        self.training_examples.append(example)
    
    def train(self, test_size: float = 0.2) -> Dict[str, Any]:
        """Train the Random Forest model."""
        if len(self.training_examples) < 5:
            raise RoutingError("Need at least 5 training examples to train RF router")
        
        self._prepare_model_mappings()
        
        # Extract features and labels
        X = []
        y = []
        
        for example in self.training_examples:
            features = self._extract_features(example.query)
            X.append(features)
            y.append(self.model_to_idx[example.model_id])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        if len(X) >= 10:  # Only split if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.rf_params['random_state'], stratify=y
            )
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(**self.rf_params)
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.rf_model.predict(X_train)
        test_pred = self.rf_model.predict(X_test)
        
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        
        # Get feature importances
        feature_names = (
            [f"embed_{i}" for i in range(self.feature_dim)] + 
            ["query_len", "word_count", "has_code", "has_math", "has_writing", 
             "has_analysis", "is_question", "starts_question", "parentheses", 
             "quotes", "long_words"]
        )
        
        importances = list(zip(feature_names, self.rf_model.feature_importances_))
        importances.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "n_training_examples": len(self.training_examples),
            "n_features": X.shape[1],
            "top_features": importances[:10],  # Top 10 most important features
            "model_distribution": {
                self.idx_to_model[idx]: int(np.sum(y == idx)) 
                for idx in range(len(FREE_MODELS))
            }
        }
    
    def route_query(
        self, 
        query: str, 
        available_models: List[str], 
        threshold: float = 0.3
    ) -> Tuple[str, float, str]:
        """Route a query using the trained Random Forest model."""
        if self.rf_model is None:
            raise RoutingError("Model not trained. Call train() first.")
        
        # Extract features
        features = self._extract_features(query).reshape(1, -1)
        
        # Get predictions and probabilities
        predicted_idx = self.rf_model.predict(features)[0]
        probabilities = self.rf_model.predict_proba(features)[0]
        
        # Get predicted model
        predicted_model = self.idx_to_model[predicted_idx]
        confidence = probabilities[predicted_idx]
        
        # Check if predicted model is available
        if predicted_model in available_models:
            selected_model = predicted_model
        else:
            # Find best available model by probability
            available_probs = []
            for model_id in available_models:
                if model_id in self.model_to_idx:
                    idx = self.model_to_idx[model_id]
                    prob = probabilities[idx]
                    available_probs.append((model_id, prob))
            
            if not available_probs:
                raise RoutingError("No available models found in trained model set")
            
            # Sort by probability
            available_probs.sort(key=lambda x: x[1], reverse=True)
            selected_model, confidence = available_probs[0]
        
        # Threshold-based routing: if confidence is too low, use fallback logic
        if confidence < threshold:
            # Use simple heuristics as fallback
            if any(kw in query.lower() for kw in ['code', 'function', 'python', 'javascript']):
                coding_models = [m for m in available_models if 'deepseek' in m.lower() or 'phi' in m.lower()]
                if coding_models:
                    selected_model = coding_models[0]
                    reasoning = f"RF low confidence ({confidence:.3f}), used coding heuristic"
                else:
                    reasoning = f"RF prediction: {selected_model} (low confidence: {confidence:.3f})"
            elif any(kw in query.lower() for kw in ['math', 'calculate', 'equation']):
                math_models = [m for m in available_models if 'phi' in m.lower() or 'deepseek' in m.lower()]
                if math_models:
                    selected_model = math_models[0]
                    reasoning = f"RF low confidence ({confidence:.3f}), used math heuristic"
                else:
                    reasoning = f"RF prediction: {selected_model} (low confidence: {confidence:.3f})"
            else:
                reasoning = f"RF prediction: {selected_model} (low confidence: {confidence:.3f})"
        else:
            reasoning = f"RF prediction: {selected_model} (confidence: {confidence:.3f})"
        
        # Update statistics
        self.routing_stats["predictions_made"] += 1
        self.routing_stats["model_usage"][selected_model] += 1
        
        # Confidence distribution
        if confidence < 0.4:
            self.routing_stats["confidence_distribution"]["low"] += 1
        elif confidence < 0.7:
            self.routing_stats["confidence_distribution"]["medium"] += 1
        else:
            self.routing_stats["confidence_distribution"]["high"] += 1
        
        return selected_model, float(confidence), reasoning
    
    def update_accuracy(self, query: str, predicted_model: str, was_correct: bool):
        """Update accuracy statistics based on feedback."""
        if was_correct:
            self.routing_stats["correct_predictions"] += 1
        
        total_predictions = max(self.routing_stats["predictions_made"], 1)
        self.routing_stats["accuracy"] = (
            self.routing_stats["correct_predictions"] / total_predictions
        )
        
        # Add as training example for future retraining
        self.add_training_example(query, predicted_model, was_correct)
    
    def get_model_preferences(self, query: str) -> Dict[str, float]:
        """Get probability distribution over all models for a query."""
        if self.rf_model is None:
            raise RoutingError("Model not trained.")
        
        features = self._extract_features(query).reshape(1, -1)
        probabilities = self.rf_model.predict_proba(features)[0]
        
        preferences = {}
        for idx, prob in enumerate(probabilities):
            model_id = self.idx_to_model[idx]
            preferences[model_id] = float(prob)
        
        return preferences
    
    def save_model(self, filepath: str):
        """Save the trained Random Forest model."""
        if self.rf_model is None:
            raise RoutingError("No model to save")
        
        save_data = {
            "rf_model": self.rf_model,
            "model_mappings": {
                "model_to_idx": self.model_to_idx,
                "idx_to_model": self.idx_to_model
            },
            "embedding_model_name": self.embedding_model_name,
            "feature_dim": self.feature_dim,
            "rf_params": self.rf_params,
            "training_examples": len(self.training_examples),
            "routing_stats": self.routing_stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load_model(self, filepath: str):
        """Load a pre-trained Random Forest model."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        self.rf_model = save_data["rf_model"]
        self.model_to_idx = save_data["model_mappings"]["model_to_idx"]
        self.idx_to_model = save_data["model_mappings"]["idx_to_model"]
        self.embedding_model_name = save_data["embedding_model_name"]
        self.feature_dim = save_data["feature_dim"]
        self.rf_params = save_data["rf_params"]
        self.routing_stats = save_data["routing_stats"]
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            **self.routing_stats,
            "training_examples": len(self.training_examples),
            "model_trained": self.rf_model is not None,
            "feature_dim": self.feature_dim
        }