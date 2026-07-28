"""
Pipeline Auditor for Mentis: checks an ML project's structure and
production readiness, producing a Production Readiness Score.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from mentis.constants import SEVERITY_CRITICAL, SEVERITY_INFO, SEVERITY_WARNING
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

# (check_name, path/glob, severity, weight, suggestion)
_CHECKS: list[tuple[str, str, str, int, str]] = [
    ("README", "README.md", SEVERITY_WARNING, 5, "Add a README.md describing the project."),
    ("requirements", "requirements.txt", SEVERITY_WARNING, 5, "Add requirements.txt or pyproject.toml dependencies."),
    ("gitignore", ".gitignore", SEVERITY_INFO, 3, "Add a .gitignore to avoid committing artifacts/secrets."),
    ("dockerfile", "Dockerfile", SEVERITY_WARNING, 8, "Add a Dockerfile for reproducible deployment."),
    ("tests", "tests", SEVERITY_CRITICAL, 15, "Add a tests/ directory with automated tests."),
    ("ci_cd", ".github/workflows", SEVERITY_CRITICAL, 15, "Add CI/CD via GitHub Actions."),
    ("logging_config", "logging.conf", SEVERITY_INFO, 3, "Add explicit logging configuration."),
    ("config_files", "config.yaml", SEVERITY_INFO, 5, "Add a config file for environment-specific settings."),
    ("env_vars", ".env.example", SEVERITY_WARNING, 5, "Add a .env.example documenting required env vars."),
    ("model_artifact", "models", SEVERITY_INFO, 5, "Add a models/ directory for versioned model artifacts."),
    ("versioning", "CHANGELOG.md", SEVERITY_INFO, 3, "Add a CHANGELOG.md to track versions."),
    ("pre_commit", ".pre-commit-config.yaml", SEVERITY_INFO, 3, "Add pre-commit hooks for code quality."),
    ("makefile", "Makefile", SEVERITY_INFO, 3, "Add a Makefile for common dev commands."),
]

_MAX_SCORE = sum(w for *_r, w, _s in _CHECKS)


@dataclass
class AuditFinding:
    """
    Result of a single audit check.

    Attributes:
        name: Check identifier (e.g. "tests").
        passed: Whether the expected file/directory was found.
        severity: "info", "warning", or "critical" if missing.
        suggestion: Actionable advice if the check failed.
    """

    name: str
    passed: bool
    severity: str
    suggestion: str


@dataclass
class AuditResult:
    """
    Full result of a Pipeline Auditor run.

    Attributes:
        project_path: Root path that was audited.
        findings: One `AuditFinding` per check performed.
        score: Production Readiness Score, 0-100.
    """

    project_path: str
    findings: list[AuditFinding] = field(default_factory=list)
    score: float = 0.0

    def failed(self, severity: str | None = None) -> list[AuditFinding]:
        """Return failed findings, optionally filtered by severity."""
        return [
            f for f in self.findings
            if not f.passed and (severity is None or f.severity == severity)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_path": self.project_path,
            "score": self.score,
            "findings": [f.__dict__ for f in self.findings],
        }

    def __repr__(self) -> str:
        n_failed = len(self.failed())
        return f"<AuditResult score={self.score:.1f}/100 failed={n_failed}/{len(self.findings)}>"


class PipelineAuditor:
    """
    Audits an ML project's directory structure against production
    readiness best practices.

    Examples:
        >>> auditor = PipelineAuditor()
        >>> result = auditor.audit(".")  # doctest: +SKIP
        >>> result.score  # doctest: +SKIP
    """

    def audit(self, project_path: str = ".") -> AuditResult:
        """
        Run all pipeline audit checks against a project directory.

        Args:
            project_path: Root directory of the project to audit.

        Returns:
            An `AuditResult` with per-check findings and an overall
            Production Readiness Score (0-100).

        Examples:
            >>> auditor = PipelineAuditor()
            >>> result = auditor.audit(".")  # doctest: +SKIP
        """
        findings: list[AuditFinding] = []
        earned = 0

        for name, rel_path, severity, weight, suggestion in _CHECKS:
            full_path = os.path.join(project_path, rel_path)

            # Special case: accept pyproject.toml as an alternative to requirements.txt
            if name == "requirements":
                alt_path = os.path.join(project_path, "pyproject.toml")
                passed = os.path.exists(full_path) or os.path.exists(alt_path)
            else:
                passed = os.path.exists(full_path)

            if passed:
                earned += weight
            findings.append(
                AuditFinding(name=name, passed=passed, severity=severity, suggestion=suggestion)
            )

        score = round((earned / _MAX_SCORE) * 100, 1) if _MAX_SCORE else 0.0
        return AuditResult(project_path=project_path, findings=findings, score=score)
    

    