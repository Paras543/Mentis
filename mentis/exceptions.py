"""
Custom exception hierarchy for Mentis.

"""

from __future__ import annotations


class MentisError(Exception):
    """Base exception for all Mentis-raised errors."""


class DatasetError(MentisError):
    """Raised when the input dataset is invalid, malformed, or unusable."""


class ModelError(MentisError):
    """Raised when model training, comparison, or inference fails."""


class DeploymentError(MentisError):
    """Raised when a deployment-readiness check fails critically."""


class ValidationError(MentisError):
    """Raised when input validation (schema, types, config) fails."""


class ConfigurationError(MentisError):
    """Raised when a YAML/user configuration is invalid or incomplete."""


class ExplainabilityError(MentisError):
    """Raised when explainability generation (SHAP, PDP, etc.) fails."""


class ReportGenerationError(MentisError):
    """Raised when report building (HTML/PDF/Markdown) fails."""
