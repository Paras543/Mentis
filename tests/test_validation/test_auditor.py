"""
Tests for mentis/validation/auditor.py — PipelineAuditor.
"""

from __future__ import annotations

import os

import pytest

from mentis.validation.auditor import AuditFinding, AuditResult, PipelineAuditor


class TestPipelineAuditor:
    def test_returns_audit_result(self, tmp_path):
        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        assert isinstance(result, AuditResult)

    def test_empty_dir_score_zero(self, tmp_path):
        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        assert result.score == 0.0

    def test_readme_detected(self, tmp_path):
        readme = tmp_path / "README.md"
        readme.write_text("# Project")

        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        assert result.score > 0
        finding = next(f for f in result.findings if f.name == "README")
        assert finding.passed is True

    def test_pyproject_toml_accepted_for_requirements(self, tmp_path):
        """Fix verification: pyproject.toml should satisfy the requirements check."""
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'test'")

        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        finding = next(f for f in result.findings if f.name == "requirements")
        assert finding.passed is True

    def test_requirements_txt_accepted_for_requirements(self, tmp_path):
        reqs = tmp_path / "requirements.txt"
        reqs.write_text("pandas>=2.0")

        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        finding = next(f for f in result.findings if f.name == "requirements")
        assert finding.passed is True

    def test_all_files_present_score_100(self, tmp_path):
        from mentis.validation.auditor import _CHECKS
        for name, rel_path, *rest in _CHECKS:
            p = tmp_path / rel_path
            if rel_path in (".github/workflows", "tests", "models", "k8s"):
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("dummy")

        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        assert result.score == 100.0

    def test_failed_filter_all_and_by_severity(self, tmp_path):
        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        failed_all = result.failed()
        failed_critical = result.failed(severity="critical")
        assert len(failed_all) > len(failed_critical)
        assert all(f.severity == "critical" for f in failed_critical)

    def test_to_dict(self, tmp_path):
        auditor = PipelineAuditor()
        result = auditor.audit(str(tmp_path))
        d = result.to_dict()
        assert "project_path" in d
        assert "score" in d
        assert "findings" in d
