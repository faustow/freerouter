"""Custom exceptions for FreeRouter."""


class FreeRouterError(Exception):
    """Base exception for FreeRouter."""
    pass


class RateLimitError(FreeRouterError):
    """Raised when model rate limits are exceeded."""
    pass


class ModelUnavailableError(FreeRouterError):
    """Raised when no suitable model is available."""
    pass


class RoutingError(FreeRouterError):
    """Raised when routing decision fails."""
    pass


class ConfigurationError(FreeRouterError):
    """Raised when configuration is invalid."""
    pass