"""Tests for semantic routing functionality."""

import pytest
from freerouter.semantic_router import SemanticRouter
from freerouter.models import IntentType


class TestSemanticRouter:
    def setup_method(self):
        self.router = SemanticRouter()
    
    def test_coding_intent_detection(self):
        """Test detection of coding-related queries."""
        queries = [
            "Write a Python function to calculate fibonacci numbers",
            "Debug this JavaScript code that's throwing an error",
            "How do I import numpy in Python?",
            "```python\ndef hello():\n    print('world')\n```",
            "What's the time complexity of this algorithm?",
            "Create a REST API using FastAPI"
        ]
        
        for query in queries:
            intent, confidence = self.router.classify_intent(query, threshold=0.5)
            assert intent == IntentType.CODING, f"Failed for query: {query}"
            assert confidence > 0.5
    
    def test_mathematics_intent_detection(self):
        """Test detection of mathematics-related queries."""
        queries = [
            "Calculate the derivative of x^2 + 3x + 1",
            "Solve this quadratic equation: 2x^2 - 4x + 2 = 0",
            "What is 15 * 23?",
            "Find the integral of sin(x) from 0 to π",
            "Prove that the square root of 2 is irrational",
            "Calculate 5! (factorial of 5)"
        ]
        
        for query in queries:
            intent, confidence = self.router.classify_intent(query, threshold=0.5)
            assert intent == IntentType.MATHEMATICS, f"Failed for query: {query}"
            assert confidence > 0.5
    
    def test_writing_intent_detection(self):
        """Test detection of writing-related queries."""
        queries = [
            "Write a blog post about machine learning",
            "Create a creative story about a dragon",
            "Help me write an essay on climate change",
            "Proofread this paragraph for grammar errors",
            "Draft a professional email to my boss",
            "Write dialogue for a character who is angry"
        ]
        
        for query in queries:
            intent, confidence = self.router.classify_intent(query, threshold=0.5)
            assert intent == IntentType.WRITING, f"Failed for query: {query}"
            assert confidence > 0.5
    
    def test_analysis_intent_detection(self):
        """Test detection of analysis-related queries."""
        queries = [
            "Analyze the pros and cons of renewable energy",
            "Compare Python and JavaScript for web development",
            "What are the trends in the stock market this year?",
            "Evaluate the effectiveness of this marketing strategy",
            "Research the causes of inflation in 2023",
            "Examine the correlation between exercise and mental health"
        ]
        
        for query in queries:
            intent, confidence = self.router.classify_intent(query, threshold=0.5)
            assert intent == IntentType.ANALYSIS, f"Failed for query: {query}"
            assert confidence > 0.5
    
    def test_general_intent_fallback(self):
        """Test that unclear queries fall back to general intent."""
        queries = [
            "Hello",
            "What's the weather like?",
            "How are you?",
            "Tell me a joke",
            "Random question"
        ]
        
        for query in queries:
            intent, confidence = self.router.classify_intent(query, threshold=0.7)
            # These should either be GENERAL or have low confidence
            assert intent == IntentType.GENERAL or confidence < 0.7
    
    def test_confidence_thresholding(self):
        """Test that low confidence queries default to general."""
        ambiguous_query = "This is a very ambiguous query with no clear intent"
        intent, confidence = self.router.classify_intent(ambiguous_query, threshold=0.8)
        
        # Should default to GENERAL due to low confidence
        if confidence < 0.8:
            assert intent == IntentType.GENERAL
    
    def test_get_intent_models(self):
        """Test getting models for specific intents."""
        coding_models = self.router.get_intent_models(IntentType.CODING)
        assert len(coding_models) > 0
        
        # Should include DeepSeek R1 for coding
        assert any("deepseek" in model.lower() for model in coding_models)
        
        math_models = self.router.get_intent_models(IntentType.MATHEMATICS)
        assert len(math_models) > 0
        
        # Should include Phi-3 for mathematics
        assert any("phi-3" in model.lower() for model in math_models)
    
    def test_route_query(self):
        """Test end-to-end query routing."""
        available_models = [
            "deepseek/deepseek-r1",
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3-8b-instruct:free"
        ]
        
        # Test coding query
        model_id, intent, confidence = self.router.route_query(
            "Write a function to sort an array",
            available_models
        )
        
        assert model_id in available_models
        assert intent == IntentType.CODING
        assert confidence > 0.5
        
        # Should prefer DeepSeek for coding
        assert "deepseek" in model_id.lower()
    
    def test_explain_routing(self):
        """Test routing explanation generation."""
        explanation = self.router.explain_routing(
            "Write some code",
            "deepseek/deepseek-r1", 
            IntentType.CODING,
            0.85
        )
        
        assert "deepseek/deepseek-r1" in explanation
        assert "coding" in explanation.lower()
        assert "0.85" in explanation
    
    def test_empty_query_handling(self):
        """Test handling of empty or None queries."""
        intent, confidence = self.router.classify_intent("", threshold=0.7)
        assert intent == IntentType.GENERAL
        
        intent, confidence = self.router.classify_intent("   ", threshold=0.7)
        assert intent == IntentType.GENERAL