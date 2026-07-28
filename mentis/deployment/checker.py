"""
Deployment Checker for Mentis: verifies a project's deployment
readiness (containerization, orchestration, health endpoints,
observability, security) and produces a Deployment Score.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from mentis.constants import SEVERITY_CRITICAL, SEVERITY_INFO, SEVERITY_WARNING
from mentis.utils.logger import get_logger

logger = get_logger(__name__)

_FILE_CHECKS: list[tuple[str, str, str, int, str]] = [
    ("dockerfile", "Dockerfile", SEVERITY_CRITICAL, 15, "Add a Dockerfile to containerize the service."),
    ("docker_compose", "docker-compose.yml", SEVERITY_WARNING, 8, "Add docker-compose.yml for local orchestration."),
    ("k8s_manifests", "k8s", SEVERITY_WARNING, 10, "Add Kubernetes manifests (k8s/) for cluster deployment."),
    ("env_vars", ".env.example", SEVERITY_WARNING, 5, "Add .env.example documenting required environment variables."),
    ("secrets_template", "secrets.example.yaml", SEVERITY_INFO, 3, "Document secrets management (without committing real secrets)."),
]

_APP_ENTRYPOINTS: list[str] = ["main.py", "app.py", "server.py", "api.py"]
_HEALTH_PATTERNS: dict[str, str] = {
    "health_endpoint": r"/health",
    "liveness_endpoint": r"/live|/liveness|/healthz",
    "readiness_endpoint": r"/ready|/readiness",
}
_FRAMEWORK_PATTERNS: dict[str, str] = {
    "fastapi": r"from\s+fastapi\s+import|FastAPI\(",
    "flask": r"from\s+flask\s+import|Flask\(__name__\)",
}

_MAX_SCORE = sum(w for *_r, w, _s in _FILE_CHECKS) + 30  # +30 for framework/health/logging checks


@dataclass
class DeploymentFinding:
    """
    Result of a single deployment readiness check.

    Attributes:
        name: Check identifier (e.g. "dockerfile", "health_endpoint").
        passed: Whether the check succeeded.
        severity: "info", "warning", or "critical" if missing.
        suggestion: Actionable advice if the check failed.
    """

    name: str
    passed: bool
    severity: str
    suggestion: str


@dataclass
class DeploymentResult:
    """
    Full result of a Deployment Checker run.

    Attributes:
        project_path: Root path that was checked.
        findings: One `DeploymentFinding` per check performed.
        score: Deployment Score, 0-100.
        detected_framework: "fastapi", "flask", or None if undetected.
    """

    project_path: str
    findings: list[DeploymentFinding] = field(default_factory=list)
    score: float = 0.0
    detected_framework: str | None = None

    def failed(self, severity: str | None = None) -> list[DeploymentFinding]:
        """Return failed findings, optionally filtered by severity."""
        return [
            f for f in self.findings
            if not f.passed and (severity is None or f.severity == severity)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_path": self.project_path,
            "score": self.score,
            "detected_framework": self.detected_framework,
            "findings": [f.__dict__ for f in self.findings],
        }

    def __repr__(self) -> str:
        n_failed = len(self.failed())
        return f"<DeploymentResult score={self.score:.1f}/100 failed={n_failed}/{len(self.findings)}>"


class DeploymentChecker:
    """
    Checks a project's deployment readiness: containerization,
    orchestration manifests, health endpoints, and framework detection.

    Examples:
        >>> checker = DeploymentChecker()
        >>> result = checker.check(".")  # doctest: +SKIP
        >>> result.score  # doctest: +SKIP
    """

    def check(self, project_path: str = ".") -> DeploymentResult:
        """
        Run all deployment readiness checks against a project directory.

        Args:
            project_path: Root directory of the project to check.

        Returns:
            A `DeploymentResult` with per-check findings, detected web
            framework (if any), and an overall Deployment Score
            (0-100).

        Examples:
            >>> checker = DeploymentChecker()
            >>> result = checker.check(".")  # doctest: +SKIP
        """
        findings: list[DeploymentFinding] = []
        earned = 0

        for name, rel_path, severity, weight, suggestion in _FILE_CHECKS:
            full_path = os.path.join(project_path, rel_path)
            passed = os.path.exists(full_path)
            if passed:
                earned += weight
            findings.append(
                DeploymentFinding(name=name, passed=passed, severity=severity, suggestion=suggestion)
            )

        source_text = self._read_entrypoint_source(project_path)

        framework = self._detect_framework(source_text)
        framework_passed = framework is not None
        if framework_passed:
            earned += 10
        findings.append(
            DeploymentFinding(
                name="web_framework",
                passed=framework_passed,
                severity=SEVERITY_WARNING,
                suggestion="Use FastAPI or Flask to serve the model via an API.",
            )
        )

        for check_name, pattern in _HEALTH_PATTERNS.items():
            passed = bool(re.search(pattern, source_text, re.IGNORECASE))
            if passed:
                earned += 5
            findings.append(
                DeploymentFinding(
                    name=check_name,
                    passed=passed,
                    severity=SEVERITY_WARNING,
                    suggestion=f"Expose a {check_name.replace('_', ' ')} for orchestrator probes.",
                )
            )

        logging_passed = bool(re.search(r"import logging|get_logger", source_text))
        if logging_passed:
            earned += 5
        findings.append(
            DeploymentFinding(
                name="logging",
                passed=logging_passed,
                severity=SEVERITY_INFO,
                suggestion="Add structured logging to the service entrypoint.",
            )
        )

        score = round((earned / _MAX_SCORE) * 100, 1) if _MAX_SCORE else 0.0
        return DeploymentResult(
            project_path=project_path,
            findings=findings,
            score=score,
            detected_framework=framework,
        )

    @staticmethod
    def _read_entrypoint_source(project_path: str) -> str:
        combined = ""
        for entry in _APP_ENTRYPOINTS:
            full_path = os.path.join(project_path, entry)
            if os.path.exists(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                        combined += f.read() + "\n"
                except OSError as exc:
                    logger.warning(f"Could not read {full_path}: {exc}")
        return combined

    @staticmethod
    def _detect_framework(source_text: str) -> str | None:
        for name, pattern in _FRAMEWORK_PATTERNS.items():
            if re.search(pattern, source_text):
                return name
        return None
    

    