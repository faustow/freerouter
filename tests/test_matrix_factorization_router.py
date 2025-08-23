"""Tests for Matrix Factorization router."""

import pytest
import tempfile
import torch
import numpy as np
from unittest.mock import Mock, patch

from freerouter.matrix_factorization_router import (
    MatrixFactorizationRouter, 
    TrainingExample, 
    MatrixFactorizationModel
)
from freerouter.exceptions import RoutingError


class TestMatrixFactorizationRouter:
    @pytest.fixture
    def router(self):
        return MatrixFactorizationRouter()
    
    @pytest.fixture
    def training_data(self):
        return [
            TrainingExample("Write a Python function", "deepseek/deepseek-r1", 0.9, 2.5),
            TrainingExample("Calculate 2+2", "microsoft/phi-3-medium-128k-instruct:free", 0.8, 1.2),
            TrainingExample("Write a story", "google/gemma-2-9b-it:free", 0.85, 3.0),
            TrainingExample("Debug this code", "deepseek/deepseek-r1", 0.95, 2.8),
            TrainingExample("Solve equation", "microsoft/phi-3-medium-128k-instruct:free", 0.9, 1.5),
            TrainingExample("Creative writing", "meta-llama/llama-3-8b-instruct:free", 0.7, 4.0),
            TrainingExample("Analyze data", "deepseek/deepseek-r1", 0.88, 2.2),
            TrainingExample("Math problem", "microsoft/phi-3-medium-128k-instruct:free", 0.92, 1.8),
            TrainingExample("Write essay", "google/gemma-2-9b-it:free", 0.8, 3.5),
            TrainingExample("Code review", "deepseek/deepseek-r1", 0.87, 2.1),
        ]
    
    def test_add_training_example(self, router):
        """Test adding training examples."""
        router.add_training_example("Test query", "deepseek/deepseek-r1", 0.9, 2.0)
        
        assert len(router.training_examples) == 1
        example = router.training_examples[0]
        assert example.query == "Test query"
        assert example.model_id == "deepseek/deepseek-r1"
        assert example.correctness == 0.9
        assert example.response_time == 2.0
    
    def test_training_with_insufficient_data(self, router):
        """Test training with too few examples."""
        router.add_training_example("Test", "deepseek/deepseek-r1", 0.9, 2.0)
        
        with pytest.raises(RoutingError, match="Need at least 10 training examples"):
            router.train(epochs=1)
    
    def test_training_with_sufficient_data(self, router, training_data):
        """Test successful training."""
        for example in training_data:
            router.training_examples.append(example)
        
        # Mock sentence transformer to avoid downloading
        with patch('freerouter.matrix_factorization_router.SentenceTransformer') as mock_st:
            mock_embeddings = np.random.randn(len(training_data), 384)  # MiniLM dimension
            mock_st.return_value.encode.return_value = mock_embeddings
            router.query_dim = 384
            
            results = router.train(epochs=5, batch_size=4)
            
            assert "final_train_loss" in results
            assert "final_val_loss" in results
            assert "epochs_trained" in results
            assert results["epochs_trained"] == 5
            assert router.mf_model is not None
    
    def test_routing_without_training(self, router):
        """Test routing before training should raise error."""
        with pytest.raises(RoutingError, match="Model not trained"):
            router.route_query("Test query", ["deepseek/deepseek-r1"])
    
    def test_routing_after_training(self, router, training_data):
        """Test routing after training."""
        for example in training_data:
            router.training_examples.append(example)
        
        with patch('freerouter.matrix_factorization_router.SentenceTransformer') as mock_st:
            # Mock training embeddings
            train_embeddings = np.random.randn(len(training_data), 384)
            query_embedding = np.random.randn(1, 384)
            
            mock_st.return_value.encode.side_effect = [train_embeddings, query_embedding]
            router.query_dim = 384
            
            # Train the model
            router.train(epochs=2, batch_size=4)
            
            # Test routing
            available_models = ["deepseek/deepseek-r1", "google/gemma-2-9b-it:free"]
            model_id, confidence, reasoning = router.route_query(
                "Write some Python code", 
                available_models
            )
            
            assert model_id in available_models
            assert 0.0 <= confidence <= 1.0
            assert "Matrix factorization predicted" in reasoning
            
            # Check stats were updated
            stats = router.get_stats()
            assert stats["predictions_made"] == 1
            assert stats["model_usage"][model_id] == 1
    
    def test_save_and_load_model(self, router, training_data):
        """Test saving and loading trained models."""
        for example in training_data:
            router.training_examples.append(example)
        
        with patch('freerouter.matrix_factorization_router.SentenceTransformer') as mock_st:
            train_embeddings = np.random.randn(len(training_data), 384)
            mock_st.return_value.encode.return_value = train_embeddings
            router.query_dim = 384
            
            router.train(epochs=2, batch_size=4)
            
            # Save model
            with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
                router.save_model(f.name)
                
                # Create new router and load
                new_router = MatrixFactorizationRouter()
                new_router.load_model(f.name)
                
                assert new_router.mf_model is not None
                assert new_router.query_dim == 384
                assert new_router.model_to_idx == router.model_to_idx
    
    def test_statistics_tracking(self, router, training_data):
        """Test that statistics are properly tracked."""
        for example in training_data:
            router.training_examples.append(example)
        
        with patch('freerouter.matrix_factorization_router.SentenceTransformer') as mock_st:
            train_embeddings = np.random.randn(len(training_data), 384)
            query_embeddings = np.random.randn(3, 384)
            
            mock_st.return_value.encode.side_effect = [train_embeddings, *query_embeddings]
            router.query_dim = 384
            
            router.train(epochs=2)
            
            available_models = ["deepseek/deepseek-r1", "google/gemma-2-9b-it:free"]
            
            # Make multiple routing decisions
            for i in range(3):
                router.route_query(f"Query {i}", available_models)
            
            stats = router.get_stats()
            assert stats["predictions_made"] == 3
            assert stats["avg_confidence"] > 0.0
            assert sum(stats["model_usage"].values()) == 3


class TestMatrixFactorizationModel:
    def test_model_initialization(self):
        """Test model initialization."""
        model = MatrixFactorizationModel(
            query_dim=384,
            num_models=5,
            embedding_dim=128
        )
        
        assert model.query_dim == 384
        assert model.num_models == 5
        assert model.embedding_dim == 128
        assert model.query_encoder is not None
        assert model.model_embeddings is not None
        assert model.interaction_matrix is not None
    
    def test_forward_pass(self):
        """Test forward pass of the model."""
        model = MatrixFactorizationModel(384, 5, 128)
        
        # Create dummy inputs
        query_embeddings = torch.randn(2, 384)  # batch_size=2
        model_indices = torch.LongTensor([[0], [2]])  # Two different models
        
        scores = model(query_embeddings, model_indices)
        
        assert scores.shape == (2, 1)
        assert torch.all((scores >= 0) & (scores <= 1))  # Sigmoid output
    
    def test_predict_all_models(self):
        """Test prediction for all models."""
        model = MatrixFactorizationModel(384, 5, 128)
        
        query_embedding = torch.randn(1, 384)
        all_scores = model.predict_all_models(query_embedding)
        
        assert all_scores.shape == (1, 5)
        assert torch.all((all_scores >= 0) & (all_scores <= 1))
    
    def test_model_embeddings_learned(self):
        """Test that model embeddings are properly learned."""
        model = MatrixFactorizationModel(384, 3, 64)
        
        # Get initial embeddings
        initial_embeds = model.model_embeddings.weight.data.clone()
        
        # Simulate training step
        optimizer = torch.optim.Adam(model.parameters())
        query_embeddings = torch.randn(4, 384)
        model_indices = torch.LongTensor([[0], [1], [2], [0]])
        targets = torch.FloatTensor([[0.9], [0.7], [0.8], [0.85]])
        
        predictions = model(query_embeddings, model_indices)
        loss = torch.nn.MSELoss()(predictions, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Embeddings should have changed
        new_embeds = model.model_embeddings.weight.data
        assert not torch.equal(initial_embeds, new_embeds)