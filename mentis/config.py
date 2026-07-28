"""
Configuration handling for Mentis.

Supports loading project settings from a YAML file (as described in
the project spec) and validating them via Pydantic, so misconfigured
projects fail fast with clear error messages rather than surfacing
cryptic errors deep inside a scan or comparison run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from mentis.exceptions import ConfigurationError


class ProjectConfig(BaseModel):
    """
    Top-level project settings.

    Attributes:
        task: ML task type, "classification" or "regression".
        target: Name of the target/label column.
    """

    task: Literal["classification", "regression"] = "classification"
    target: str | None = None


class ScannerConfig(BaseModel):
    """
    Settings controlling the Dataset Scanner.

    Attributes:
        leakage: Whether to run the data-leakage check.
        missing_threshold: Fraction of missing values above which a
            column is flagged as critical.
        correlation_threshold: Absolute correlation above which two
            features are flagged as redundant.
    """

    leakage: bool = True
    missing_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    correlation_threshold: float = Field(default=0.90, ge=0.0, le=1.0)


class ComparisonConfig(BaseModel):
    """
    Settings controlling automated model comparison.

    Attributes:
        cv: Number of cross-validation folds.
        models: Optional explicit list of model names to include.
            If None, a sensible default set is used.
    """

    cv: int = Field(default=5, ge=2)
    models: list[str] | None = None


class ReportConfig(BaseModel):
    """
    Settings controlling report generation.

    Attributes:
        format: Output format for generated reports.
        output_dir: Directory where reports are written.
    """

    format: Literal["html", "markdown", "pdf"] = "html"
    output_dir: str = "mentis_reports"


class MentisConfig(BaseModel):
    """
    Root configuration object, mirroring the structure of a Mentis
    YAML config file.

    Attributes:
        project: Project-level settings (task, target).
        scanner: Dataset Scanner settings.
        comparison: Model comparison settings.
        report: Report generation settings.
    """

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MentisConfig":
        """
        Load a `MentisConfig` from a YAML file.

        Args:
            path: Path to a YAML config file.

        Returns:
            A validated `MentisConfig` instance.

        Raises:
            ConfigurationError: If the file is missing, unreadable, not
                valid YAML, or fails schema validation.

        Examples:
            >>> config = MentisConfig.from_yaml("mentis.yaml")  # doctest: +SKIP
        """
        config_path = Path(path)
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")

        try:
            raw: dict[str, Any] = yaml.safe_load(config_path.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Invalid YAML in '{config_path}': {exc}") from exc

        try:
            return cls(**raw)
        except PydanticValidationError as exc:
            raise ConfigurationError(f"Invalid Mentis configuration: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict representation of this configuration."""
        return self.model_dump()
    

    