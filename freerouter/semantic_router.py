"""Semantic routing for intent classification without training."""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .models import IntentType


@dataclass
class IntentSignature:
    """Signature patterns for intent detection."""
    keywords: List[str]
    patterns: List[str]  # Regex patterns
    weight: float = 1.0


class SemanticRouter:
    """Training-free semantic router using pattern matching."""
    
    def __init__(self):
        self.intent_signatures = {
            IntentType.CODING: IntentSignature(
                keywords=[
                    "function", "class", "method", "variable", "import", "def",
                    "return", "if", "else", "for", "while", "try", "except",
                    "debug", "error", "bug", "compile", "syntax", "code",
                    "python", "javascript", "java", "c++", "rust", "go",
                    "algorithm", "data structure", "api", "library", "framework",
                    "git", "repository", "commit", "pull request", "merge"
                ],
                patterns=[
                    r"```[\w]*\s*\n.*?\n```",  # Code blocks
                    r"`[^`]+`",  # Inline code
                    r"\b(def|class|function|var|let|const|int|string|bool)\b",
                    r"\b\w+\(\)",  # Function calls
                    r"[a-zA-Z_]\w*\.[a-zA-Z_]\w*",  # Method calls
                ],
                weight=1.2
            ),
            IntentType.MATHEMATICS: IntentSignature(
                keywords=[
                    "calculate", "solve", "equation", "formula", "math", "algebra",
                    "calculus", "derivative", "integral", "matrix", "vector",
                    "probability", "statistics", "geometry", "trigonometry",
                    "theorem", "proof", "number", "prime", "factorial",
                    "graph", "plot", "function", "variable", "constant"
                ],
                patterns=[
                    r"\b\d+\s*[+\-*/^]\s*\d+",  # Basic math operations
                    r"[a-zA-Z]\s*=\s*\d+",  # Variable assignments
                    r"\b(sin|cos|tan|log|ln|sqrt|exp)\(",  # Math functions
                    r"\b\d+!\b",  # Factorials
                    r"[∑∏∫∂√π∞]",  # Math symbols
                    r"\$.*?\$",  # LaTeX math
                ],
                weight=1.3
            ),
            IntentType.WRITING: IntentSignature(
                keywords=[
                    "write", "essay", "article", "story", "blog", "content",
                    "creative", "narrative", "character", "plot", "dialogue",
                    "grammar", "style", "tone", "edit", "proofread", "revise",
                    "paragraph", "sentence", "word", "author", "publish",
                    "draft", "outline", "summary", "conclusion", "introduction"
                ],
                patterns=[
                    r"\b(write|create|compose|draft)\s+\w+",
                    r"(story|essay|article|blog|content)\s+(about|on)",
                    r"\b(tone|style|voice)\s+(should|is|be)",
                    r"(first|second|third)\s+person",
                ],
                weight=1.1
            ),
            IntentType.ANALYSIS: IntentSignature(
                keywords=[
                    "analyze", "compare", "evaluate", "assess", "review",
                    "examine", "investigate", "research", "study", "explore",
                    "data", "trend", "pattern", "insight", "conclusion",
                    "hypothesis", "theory", "evidence", "result", "finding",
                    "correlation", "causation", "significance", "interpretation"
                ],
                patterns=[
                    r"\b(analyze|compare|evaluate)\s+\w+",
                    r"what\s+(is|are)\s+the\s+(trend|pattern|correlation)",
                    r"(pros\s+and\s+cons|advantages\s+and\s+disadvantages)",
                    r"\b(data|statistics|metrics|results)\s+(show|indicate|suggest)",
                ],
                weight=1.0
            ),
            IntentType.GENERAL: IntentSignature(
                keywords=[
                    "what", "how", "why", "when", "where", "who", "which",
                    "explain", "describe", "tell", "help", "question", "answer",
                    "information", "knowledge", "learn", "understand", "know"
                ],
                patterns=[
                    r"^(what|how|why|when|where|who)\s+",
                    r"\b(can\s+you|could\s+you|please)\b",
                    r"\?$",  # Questions
                ],
                weight=0.8
            )
        }
    
    def classify_intent(self, query: str, threshold: float = 0.7) -> Tuple[IntentType, float]:
        """Classify the intent of a query."""
        query_lower = query.lower()
        scores = {}
        
        for intent_type, signature in self.intent_signatures.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in signature.keywords 
                                if keyword in query_lower)
            keyword_score = (keyword_matches / len(signature.keywords)) * signature.weight
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in signature.patterns 
                                if re.search(pattern, query, re.IGNORECASE | re.DOTALL))
            pattern_score = (pattern_matches / max(len(signature.patterns), 1)) * signature.weight
            
            # Combined score with pattern emphasis
            score = (keyword_score * 0.6) + (pattern_score * 0.4)
            scores[intent_type] = score
        
        # Find the highest scoring intent
        best_intent = max(scores.items(), key=lambda x: x[1])
        
        # If confidence is too low, default to GENERAL
        if best_intent[1] < threshold:
            return IntentType.GENERAL, scores[IntentType.GENERAL]
        
        return best_intent[0], best_intent[1]
    
    def get_intent_models(self, intent: IntentType) -> List[str]:
        """Get models specialized for a given intent."""
        from .models import FREE_MODELS
        
        specialized_models = []
        for model_id, model_info in FREE_MODELS.items():
            if intent in model_info.intent_specialization:
                specialized_models.append(model_id)
        
        # If no specialized models, return all models
        if not specialized_models:
            specialized_models = list(FREE_MODELS.keys())
        
        return specialized_models
    
    def route_query(self, query: str, available_models: List[str], 
                   threshold: float = 0.7) -> Tuple[str, IntentType, float]:
        """Route a query to the best available model."""
        intent, confidence = self.classify_intent(query, threshold)
        intent_models = self.get_intent_models(intent)
        
        # Find the best available model for this intent
        for model_id in intent_models:
            if model_id in available_models:
                return model_id, intent, confidence
        
        # Fallback to first available model
        if available_models:
            return available_models[0], intent, confidence
        
        # No models available
        raise ValueError("No models available for routing")
    
    def explain_routing(self, query: str, model_id: str, intent: IntentType, 
                       confidence: float) -> str:
        """Generate explanation for routing decision."""
        return (f"Routed to {model_id} based on {intent.value} intent "
                f"(confidence: {confidence:.2f}). Query matched patterns "
                f"typical of {intent.value} tasks.")