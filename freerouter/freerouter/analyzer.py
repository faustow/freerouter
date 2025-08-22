"""
Query analyzer for classifying user queries.

This module analyzes incoming queries to determine their type, complexity,
and characteristics to help with intelligent model routing.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum

from .config import Config, get_config

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Enumeration of query types."""
    CODING = "coding"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS = "analysis"
    MATH = "math"
    REASONING = "reasoning"
    CONVERSATION = "conversation"
    UNKNOWN = "unknown"


class QueryComplexity(Enum):
    """Enumeration of query complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@dataclass
class QueryFeatures:
    """Features extracted from a query."""
    length: int
    word_count: int
    sentence_count: int
    question_count: int
    has_code_indicators: bool
    has_math_indicators: bool
    has_creative_indicators: bool
    has_analysis_indicators: bool
    has_reasoning_indicators: bool
    programming_languages: Set[str] = field(default_factory=set)
    technical_terms: Set[str] = field(default_factory=set)


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    query: str
    primary_type: QueryType
    secondary_types: List[QueryType] = field(default_factory=list)
    complexity: QueryComplexity = QueryComplexity.MODERATE
    confidence: float = 0.0
    features: QueryFeatures = field(default_factory=lambda: QueryFeatures(0, 0, 0, 0, False, False, False, False, False))
    matched_keywords: List[str] = field(default_factory=list)
    matched_patterns: List[str] = field(default_factory=list)
    suggested_models: List[str] = field(default_factory=list)


class QueryAnalyzer:
    """
    Analyzes user queries to determine their type and characteristics.
    
    Uses keyword matching, pattern recognition, and heuristics to classify
    queries into different types for optimal model routing.
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the query analyzer.
        
        Args:
            config: Configuration object. If None, uses global config.
        """
        self.config = config or get_config()
        
        # Compile regex patterns for better performance
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self._compile_patterns()
        
        # Programming language indicators
        self.programming_languages = {
            'python', 'javascript', 'java', 'c++', 'cpp', 'c#', 'csharp',
            'go', 'rust', 'swift', 'kotlin', 'php', 'ruby', 'scala',
            'typescript', 'html', 'css', 'sql', 'bash', 'shell', 'r',
            'matlab', 'julia', 'perl', 'lua', 'dart', 'objective-c'
        }
        
        # Technical terms that indicate coding queries
        self.code_indicators = {
            'function', 'method', 'class', 'variable', 'array', 'list',
            'dictionary', 'object', 'api', 'database', 'framework',
            'library', 'import', 'module', 'package', 'syntax', 'error',
            'debug', 'compile', 'runtime', 'algorithm', 'data structure'
        }
        
        # Math-related terms
        self.math_indicators = {
            'equation', 'formula', 'calculate', 'solve', 'derivative',
            'integral', 'matrix', 'vector', 'probability', 'statistics',
            'algebra', 'geometry', 'calculus', 'trigonometry', 'logarithm'
        }
        
        # Creative writing indicators
        self.creative_indicators = {
            'story', 'poem', 'character', 'plot', 'narrative', 'fiction',
            'creative', 'imaginative', 'fantasy', 'adventure', 'mystery',
            'romance', 'dialogue', 'scene', 'chapter', 'novel'
        }
        
        # Analysis indicators
        self.analysis_indicators = {
            'analyze', 'compare', 'evaluate', 'assess', 'review', 'critique',
            'examine', 'study', 'research', 'investigate', 'pros and cons',
            'advantages', 'disadvantages', 'benefits', 'drawbacks'
        }
        
        # Reasoning indicators
        self.reasoning_indicators = {
            'why', 'because', 'therefore', 'logic', 'reasoning', 'argument',
            'premise', 'conclusion', 'paradox', 'contradiction', 'philosophy',
            'ethics', 'morality', 'dilemma', 'thought experiment'
        }
        
        logger.info("Query analyzer initialized")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns from configuration."""
        for query_type, type_config in self.config.query_types.items():
            patterns = []
            for pattern_str in type_config.patterns:
                try:
                    pattern = re.compile(pattern_str, re.IGNORECASE)
                    patterns.append(pattern)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern '{pattern_str}' for {query_type}: {e}")
            
            self._compiled_patterns[query_type] = patterns
        
        logger.debug(f"Compiled {sum(len(p) for p in self._compiled_patterns.values())} patterns")
    
    def _extract_features(self, query: str) -> QueryFeatures:
        """Extract features from the query text."""
        query_lower = query.lower()
        
        # Basic text metrics
        length = len(query)
        words = query.split()
        word_count = len(words)
        sentences = re.split(r'[.!?]+', query)
        sentence_count = len([s for s in sentences if s.strip()])
        question_count = query.count('?')
        
        # Check for various indicators
        has_code_indicators = any(indicator in query_lower for indicator in self.code_indicators)
        has_math_indicators = any(indicator in query_lower for indicator in self.math_indicators)
        has_creative_indicators = any(indicator in query_lower for indicator in self.creative_indicators)
        has_analysis_indicators = any(indicator in query_lower for indicator in self.analysis_indicators)
        has_reasoning_indicators = any(indicator in query_lower for indicator in self.reasoning_indicators)
        
        # Detect programming languages
        programming_languages = set()
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.programming_languages:
                programming_languages.add(word_lower)
        
        # Detect technical terms
        technical_terms = set()
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.code_indicators:
                technical_terms.add(word_lower)
        
        return QueryFeatures(
            length=length,
            word_count=word_count,
            sentence_count=sentence_count,
            question_count=question_count,
            has_code_indicators=has_code_indicators,
            has_math_indicators=has_math_indicators,
            has_creative_indicators=has_creative_indicators,
            has_analysis_indicators=has_analysis_indicators,
            has_reasoning_indicators=has_reasoning_indicators,
            programming_languages=programming_languages,
            technical_terms=technical_terms
        )
    
    def _calculate_complexity(self, features: QueryFeatures, query: str) -> QueryComplexity:
        """Determine query complexity based on features."""
        complexity_score = 0
        
        # Length-based scoring
        if features.length > 500:
            complexity_score += 2
        elif features.length > 200:
            complexity_score += 1
        
        # Word count scoring
        if features.word_count > 100:
            complexity_score += 2
        elif features.word_count > 50:
            complexity_score += 1
        
        # Multiple sentences increase complexity
        if features.sentence_count > 3:
            complexity_score += 1
        
        # Multiple questions increase complexity
        if features.question_count > 2:
            complexity_score += 1
        
        # Technical terms and programming languages add complexity
        if len(features.technical_terms) > 3:
            complexity_score += 1
        if len(features.programming_languages) > 1:
            complexity_score += 1
        
        # Code blocks or complex patterns
        if '```' in query or 'def ' in query or 'class ' in query:
            complexity_score += 2
        
        # Multi-step requests
        multi_step_indicators = ['first', 'then', 'next', 'finally', 'step 1', 'step 2']
        if any(indicator in query.lower() for indicator in multi_step_indicators):
            complexity_score += 1
        
        if complexity_score >= 4:
            return QueryComplexity.COMPLEX
        elif complexity_score >= 2:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.SIMPLE
    
    def _match_keywords_and_patterns(self, query: str) -> Dict[str, Tuple[List[str], List[str]]]:
        """Match keywords and patterns for each query type."""
        query_lower = query.lower()
        matches: Dict[str, Tuple[List[str], List[str]]] = {}
        
        for query_type, type_config in self.config.query_types.items():
            matched_keywords = []
            matched_patterns = []
            
            # Check keywords
            for keyword in type_config.keywords:
                if keyword.lower() in query_lower:
                    matched_keywords.append(keyword)
            
            # Check patterns
            if query_type in self._compiled_patterns:
                for pattern in self._compiled_patterns[query_type]:
                    if pattern.search(query):
                        matched_patterns.append(pattern.pattern)
            
            matches[query_type] = (matched_keywords, matched_patterns)
        
        return matches
    
    def _calculate_type_scores(
        self,
        features: QueryFeatures,
        matches: Dict[str, Tuple[List[str], List[str]]]
    ) -> Dict[QueryType, float]:
        """Calculate confidence scores for each query type."""
        scores = {query_type: 0.0 for query_type in QueryType}
        
        # Base scoring from keyword and pattern matches
        for query_type_str, (keywords, patterns) in matches.items():
            try:
                query_type = QueryType(query_type_str)
                score = len(keywords) * 1.0 + len(patterns) * 2.0
                scores[query_type] = score
            except ValueError:
                continue
        
        # Feature-based scoring adjustments
        if features.has_code_indicators or features.programming_languages:
            scores[QueryType.CODING] += 3.0
        
        if features.has_math_indicators:
            scores[QueryType.MATH] += 3.0
        
        if features.has_creative_indicators:
            scores[QueryType.CREATIVE_WRITING] += 2.0
        
        if features.has_analysis_indicators:
            scores[QueryType.ANALYSIS] += 2.0
        
        if features.has_reasoning_indicators:
            scores[QueryType.REASONING] += 2.0
        
        # Question patterns
        if features.question_count > 0:
            scores[QueryType.CONVERSATION] += 1.0
        
        # Length-based adjustments
        if features.length < 50:
            scores[QueryType.CONVERSATION] += 1.0
        elif features.length > 300:
            scores[QueryType.ANALYSIS] += 1.0
        
        # Normalize scores
        max_score = max(scores.values()) if scores.values() else 1.0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        return scores
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Analyze a query and return its classification.
        
        Args:
            query: The user query to analyze
            
        Returns:
            QueryAnalysis object with classification results
        """
        if not query or not query.strip():
            return QueryAnalysis(
                query=query,
                primary_type=QueryType.UNKNOWN,
                confidence=0.0
            )
        
        # Extract features
        features = self._extract_features(query)
        
        # Calculate complexity
        complexity = self._calculate_complexity(features, query)
        
        # Match keywords and patterns
        matches = self._match_keywords_and_patterns(query)
        
        # Calculate type scores
        type_scores = self._calculate_type_scores(features, matches)
        
        # Determine primary and secondary types
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        
        primary_type = sorted_types[0][0] if sorted_types[0][1] > 0 else QueryType.UNKNOWN
        primary_confidence = sorted_types[0][1]
        
        # Secondary types (those with score > 30% of primary)
        secondary_types = []
        threshold = primary_confidence * 0.3
        for query_type, score in sorted_types[1:]:
            if score > threshold and score > 0.1:
                secondary_types.append(query_type)
        
        # Get matched keywords and patterns for primary type
        primary_type_str = primary_type.value if primary_type != QueryType.UNKNOWN else 'conversation'
        matched_keywords, matched_patterns = matches.get(primary_type_str, ([], []))
        
        # Get suggested models
        suggested_models = self.config.get_preferred_models(primary_type_str)
        
        analysis = QueryAnalysis(
            query=query,
            primary_type=primary_type,
            secondary_types=secondary_types,
            complexity=complexity,
            confidence=primary_confidence,
            features=features,
            matched_keywords=matched_keywords,
            matched_patterns=matched_patterns,
            suggested_models=suggested_models
        )
        
        logger.debug(f"Query analyzed: type={primary_type.value}, confidence={primary_confidence:.2f}, complexity={complexity.value}")
        
        return analysis
    
    def analyze_batch(self, queries: List[str]) -> List[QueryAnalysis]:
        """Analyze multiple queries in batch."""
        return [self.analyze(query) for query in queries]
    
    def get_type_distribution(self, queries: List[str]) -> Dict[QueryType, int]:
        """Get the distribution of query types in a batch."""
        analyses = self.analyze_batch(queries)
        distribution = {query_type: 0 for query_type in QueryType}
        
        for analysis in analyses:
            distribution[analysis.primary_type] += 1
        
        return distribution