"""
Tests for the QueryAnalyzer module.
"""

import pytest
from freerouter.freerouter.analyzer import QueryAnalyzer, QueryType, QueryComplexity


class TestQueryAnalyzer:
    """Test cases for QueryAnalyzer."""
    
    def test_init(self, mock_config):
        """Test QueryAnalyzer initialization."""
        analyzer = QueryAnalyzer(mock_config)
        assert analyzer.config == mock_config
        assert len(analyzer._compiled_patterns) >= 0
    
    def test_analyze_coding_query(self, mock_query_analyzer):
        """Test analysis of coding queries."""
        queries = [
            "Write a Python function to calculate factorial",
            "How do I implement a binary search algorithm?",
            "Debug this JavaScript code snippet",
            "Create a REST API endpoint in Flask"
        ]
        
        for query in queries:
            analysis = mock_query_analyzer.analyze(query)
            assert analysis.primary_type == QueryType.CODING
            assert analysis.confidence > 0
            assert "python" in analysis.features.programming_languages or \
                   "javascript" in analysis.features.programming_languages or \
                   analysis.features.has_code_indicators
    
    def test_analyze_creative_writing_query(self, mock_query_analyzer):
        """Test analysis of creative writing queries."""
        queries = [
            "Write a short story about time travel",
            "Create a poem about the ocean", 
            "Describe a magical forest scene",
            "Write dialogue between two characters"
        ]
        
        for query in queries:
            analysis = mock_query_analyzer.analyze(query)
            assert analysis.primary_type == QueryType.CREATIVE_WRITING
            assert analysis.confidence > 0
            assert analysis.features.has_creative_indicators
    
    def test_analyze_math_query(self, mock_query_analyzer):
        """Test analysis of math queries."""
        queries = [
            "Solve: 2x + 5 = 15",
            "Calculate the area of a circle with radius 7",
            "What is 15% of 240?",
            "Find the derivative of x^2 + 3x + 2"
        ]
        
        for query in queries:
            analysis = mock_query_analyzer.analyze(query)
            assert analysis.primary_type == QueryType.MATH
            assert analysis.confidence > 0
            assert analysis.features.has_math_indicators
    
    def test_analyze_conversation_query(self, mock_query_analyzer):
        """Test analysis of conversational queries."""
        queries = [
            "Hello, how are you?",
            "What's the weather like today?",
            "Can you help me?",
            "Tell me about yourself"
        ]
        
        for query in queries:
            analysis = mock_query_analyzer.analyze(query)
            # Conversation queries might be classified as conversation or reasoning
            assert analysis.primary_type in [QueryType.CONVERSATION, QueryType.REASONING]
            assert analysis.confidence > 0
    
    def test_analyze_empty_query(self, mock_query_analyzer):
        """Test analysis of empty or invalid queries."""
        empty_queries = ["", "   ", None]
        
        for query in empty_queries:
            analysis = mock_query_analyzer.analyze(query or "")
            assert analysis.primary_type == QueryType.UNKNOWN
            assert analysis.confidence == 0.0
    
    def test_query_complexity_detection(self, mock_query_analyzer):
        """Test query complexity detection."""
        # Simple query
        simple_analysis = mock_query_analyzer.analyze("Hi")
        assert simple_analysis.complexity == QueryComplexity.SIMPLE
        
        # Moderate query  
        moderate_analysis = mock_query_analyzer.analyze("Write a Python function to calculate factorial")
        assert moderate_analysis.complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
        
        # Complex query
        complex_query = (
            "Write a comprehensive Python web application using Flask that includes "
            "user authentication, database integration with SQLAlchemy, API endpoints "
            "for CRUD operations, error handling, logging, and unit tests. The application "
            "should follow best practices for security and performance optimization."
        )
        complex_analysis = mock_query_analyzer.analyze(complex_query)
        assert complex_analysis.complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
    
    def test_feature_extraction(self, mock_query_analyzer):
        """Test feature extraction from queries."""
        query = "Write a Python function using JavaScript and SQL to debug this error"
        analysis = mock_query_analyzer.analyze(query)
        
        features = analysis.features
        assert features.word_count > 0
        assert features.length > 0
        assert features.has_code_indicators
        assert "python" in features.programming_languages
        assert "javascript" in features.programming_languages
    
    def test_batch_analysis(self, mock_query_analyzer, sample_queries):
        """Test batch analysis of multiple queries."""
        all_queries = []
        for query_list in sample_queries.values():
            all_queries.extend(query_list)
        
        analyses = mock_query_analyzer.analyze_batch(all_queries)
        assert len(analyses) == len(all_queries)
        
        for analysis in analyses:
            assert hasattr(analysis, 'primary_type')
            assert hasattr(analysis, 'confidence')
            assert analysis.confidence >= 0
    
    def test_type_distribution(self, mock_query_analyzer, sample_queries):
        """Test query type distribution calculation."""
        all_queries = []
        for query_list in sample_queries.values():
            all_queries.extend(query_list[:2])  # Take first 2 from each category
        
        distribution = mock_query_analyzer.get_type_distribution(all_queries)
        
        assert isinstance(distribution, dict)
        assert all(isinstance(count, int) for count in distribution.values())
        assert sum(distribution.values()) == len(all_queries)
    
    def test_secondary_types_detection(self, mock_query_analyzer):
        """Test detection of secondary query types."""
        # Query that could be both coding and analysis
        query = "Compare different sorting algorithms and implement the most efficient one in Python"
        analysis = mock_query_analyzer.analyze(query)
        
        # Should detect coding as primary or secondary
        assert (analysis.primary_type == QueryType.CODING or 
                QueryType.CODING in analysis.secondary_types or
                analysis.primary_type == QueryType.ANALYSIS or
                QueryType.ANALYSIS in analysis.secondary_types)
    
    def test_suggested_models(self, mock_query_analyzer):
        """Test that suggested models are returned for known query types."""
        query = "Write a Python function"
        analysis = mock_query_analyzer.analyze(query)
        
        # Should have suggested models based on configuration
        assert isinstance(analysis.suggested_models, list)
        # The exact models depend on configuration, so just check structure
    
    def test_matched_keywords_and_patterns(self, mock_query_analyzer):
        """Test that keywords and patterns are matched correctly."""
        query = "Write a function in Python"
        analysis = mock_query_analyzer.analyze(query)
        
        # Should have matched some keywords
        assert isinstance(analysis.matched_keywords, list)
        assert isinstance(analysis.matched_patterns, list)
        
        # For coding query, should match coding-related keywords
        if analysis.primary_type == QueryType.CODING:
            assert any(keyword in ["code", "function", "python"] 
                      for keyword in analysis.matched_keywords)