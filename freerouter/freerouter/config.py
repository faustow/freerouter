"""
Configuration management for FreeRouter.

This module handles loading and managing configuration from YAML files,
environment variables, and default settings.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = Field(default=20, ge=1)
    requests_per_hour: int = Field(default=200, ge=1)


class ModelCapabilities(BaseModel):
    """Model capability ratings (1-5 scale)."""
    coding: int = Field(default=3, ge=1, le=5)
    reasoning: int = Field(default=3, ge=1, le=5)
    creative_writing: int = Field(default=3, ge=1, le=5)
    analysis: int = Field(default=3, ge=1, le=5)
    conversation: int = Field(default=3, ge=1, le=5)
    math: int = Field(default=3, ge=1, le=5)


class ModelProfile(BaseModel):
    """Configuration for a specific model."""
    capabilities: ModelCapabilities
    specialties: List[str] = Field(default_factory=list)
    context_window: int = Field(default=4096, ge=1)
    cost_tier: str = Field(default="free")
    timeout: Optional[int] = Field(default=None, ge=1)
    max_retries: Optional[int] = Field(default=None, ge=0)
    rate_limit: Optional[RateLimitConfig] = None


class QueryTypeConfig(BaseModel):
    """Configuration for query type detection."""
    keywords: List[str] = Field(default_factory=list)
    patterns: List[str] = Field(default_factory=list)
    preferred_models: List[str] = Field(default_factory=list)


class DefaultConfig(BaseModel):
    """Default configuration settings."""
    timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=0)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)


class EvaluationSettings(BaseModel):
    """Evaluation configuration."""
    evaluation_interval: int = Field(default=24, ge=1)
    min_queries_per_category: int = Field(default=10, ge=1)
    response_timeout: int = Field(default=30, ge=1)
    concurrent_evaluations: int = Field(default=3, ge=1)


class ScoringCriteria(BaseModel):
    """Scoring criteria for model evaluation."""
    weight: float = Field(ge=0.0, le=1.0)
    description: str


class Config(BaseModel):
    """Main configuration class for FreeRouter."""
    
    # API Configuration
    openrouter_api_key: str = Field(default="")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    
    # Default settings
    default_config: DefaultConfig = Field(default_factory=DefaultConfig)
    
    # Model profiles
    model_profiles: Dict[str, ModelProfile] = Field(default_factory=dict)
    
    # Query type configurations
    query_types: Dict[str, QueryTypeConfig] = Field(default_factory=dict)
    
    # Evaluation settings
    evaluation_settings: EvaluationSettings = Field(default_factory=EvaluationSettings)
    
    # Test queries for evaluation
    test_queries: Dict[str, List[str]] = Field(default_factory=dict)
    
    # Scoring criteria
    scoring_criteria: Dict[str, ScoringCriteria] = Field(default_factory=dict)
    
    # Evaluation thresholds
    thresholds: Dict[str, float] = Field(default_factory=dict)
    
    # Logging configuration
    log_level: str = Field(default="INFO")
    log_file: Optional[str] = Field(default=None)
    
    @validator('openrouter_api_key', pre=True, always=True)
    def set_api_key(cls, v: str) -> str:
        """Set API key from environment variable if not provided."""
        if not v:
            v = os.getenv('OPENROUTER_API_KEY', '')
        if not v:
            logger.warning("No OpenRouter API key found. Set OPENROUTER_API_KEY environment variable.")
        return v
    
    @validator('log_level')
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    def get_model_config(self, model_name: str) -> ModelProfile:
        """Get configuration for a specific model."""
        if model_name in self.model_profiles:
            return self.model_profiles[model_name]
        
        # Return default configuration if model not found
        logger.warning(f"No configuration found for model '{model_name}', using defaults")
        return ModelProfile(capabilities=ModelCapabilities())
    
    def get_preferred_models(self, query_type: str) -> List[str]:
        """Get preferred models for a query type."""
        if query_type in self.query_types:
            return self.query_types[query_type].preferred_models
        return []
    
    def get_model_capability(self, model_name: str, capability: str) -> int:
        """Get capability rating for a model."""
        model_config = self.get_model_config(model_name)
        return getattr(model_config.capabilities, capability, 3)


def load_config(
    models_config_path: Optional[Union[str, Path]] = None,
    evaluation_config_path: Optional[Union[str, Path]] = None,
    **kwargs: Any
) -> Config:
    """
    Load configuration from YAML files and environment variables.
    
    Args:
        models_config_path: Path to models configuration YAML file
        evaluation_config_path: Path to evaluation configuration YAML file
        **kwargs: Additional configuration overrides
        
    Returns:
        Loaded configuration object
    """
    # Default config paths
    if models_config_path is None:
        models_config_path = Path(__file__).parent.parent.parent / "configs" / "models.yaml"
    if evaluation_config_path is None:
        evaluation_config_path = Path(__file__).parent.parent.parent / "configs" / "evaluation.yaml"
    
    config_data: Dict[str, Any] = {}
    
    # Load models configuration
    try:
        if Path(models_config_path).exists():
            with open(models_config_path, 'r', encoding='utf-8') as f:
                models_data = yaml.safe_load(f)
                if models_data:
                    config_data.update(models_data)
            logger.info(f"Loaded models configuration from {models_config_path}")
        else:
            logger.warning(f"Models configuration file not found: {models_config_path}")
    except Exception as e:
        logger.error(f"Error loading models configuration: {e}")
    
    # Load evaluation configuration
    try:
        if Path(evaluation_config_path).exists():
            with open(evaluation_config_path, 'r', encoding='utf-8') as f:
                eval_data = yaml.safe_load(f)
                if eval_data:
                    config_data.update(eval_data)
            logger.info(f"Loaded evaluation configuration from {evaluation_config_path}")
        else:
            logger.warning(f"Evaluation configuration file not found: {evaluation_config_path}")
    except Exception as e:
        logger.error(f"Error loading evaluation configuration: {e}")
    
    # Apply any overrides
    config_data.update(kwargs)
    
    # Create and return configuration
    try:
        config = Config(**config_data)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error creating configuration: {e}")
        # Return default configuration as fallback
        return Config()


def setup_logging(config: Config) -> None:
    """Set up logging based on configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging_config = {
        'level': getattr(logging, config.log_level),
        'format': log_format,
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if config.log_file:
        logging_config['filename'] = config.log_file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)
    
    # Set up logger for this module
    logger.setLevel(getattr(logging, config.log_level))
    
    logger.info(f"Logging configured with level: {config.log_level}")
    if config.log_file:
        logger.info(f"Logging to file: {config.log_file}")


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
        setup_logging(_config)
    return _config


def set_config(config: Config) -> None:
    """Set the global configuration instance."""
    global _config
    _config = config
    setup_logging(config)