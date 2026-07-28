"""
Tests for mentis/reporting/report_builder.py — report context assembly and output.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from mentis import Guardian
from mentis.exceptions import ReportGenerationError
from mentis.reporting.report_builder import ReportBuilder


@pytest.fixture
def guardian_with_scan(clean_df, clf_guardian):
    clf_guardian.scan(clean_df, target="target")
    return clf_guardian


@pytest.fixture
def guardian_with_audit(tmp_path, clf_guardian):
    clf_guardian.audit_pipeline(str(tmp_path))
    return clf_guardian


@pytest.fixture
def guardian_with_deployment(tmp_path, clf_guardian):
    clf_guardian.deploy_check(str(tmp_path))
    return clf_guardian


class TestReportBuilderContext:
    def test_context_has_required_keys(self, guardian_with_scan):
        builder = ReportBuilder()
        ctx = builder.build_context(guardian_with_scan)
        for key in ("generated_at", "scan", "comparison", "explain", "audit",
                    "deployment", "bias", "drift", "monitoring", "has_any_results"):
            assert key in ctx

    def test_has_any_results_true_after_scan(self, guardian_with_scan):
        ctx = ReportBuilder().build_context(guardian_with_scan)
        assert ctx["has_any_results"] is True

    def test_has_any_results_false_for_fresh_guardian(self):
        ctx = ReportBuilder().build_context(Guardian())
        assert ctx["has_any_results"] is False

    def test_scan_context_populated(self, guardian_with_scan):
        ctx = ReportBuilder().build_context(guardian_with_scan)
        assert ctx["scan"] is not None

    def test_none_sections_when_not_run(self, guardian_with_scan):
        ctx = ReportBuilder().build_context(guardian_with_scan)
        assert ctx["comparison"] is None
        assert ctx["explain"] is None
        assert ctx["bias"] is None


class TestReportBuilderUnsupportedFormat:
    def test_unsupported_format_raises(self, guardian_with_scan, tmp_path):
        builder = ReportBuilder()
        with pytest.raises(ReportGenerationError, match="Unsupported report format"):
            builder.build(guardian_with_scan, output_dir=str(tmp_path), fmt="docx")

    def test_no_results_raises(self, tmp_path):
        builder = ReportBuilder()
        guardian = Guardian()
        with pytest.raises(ReportGenerationError, match="No results available"):
            builder.build(guardian, output_dir=str(tmp_path), fmt="html")


class TestReportBuilderHtml:
    def test_html_report_created(self, guardian_with_scan, tmp_path):
        path = ReportBuilder().build(
            guardian_with_scan, output_dir=str(tmp_path), fmt="html"
        )
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_html_report_nonempty(self, guardian_with_scan, tmp_path):
        path = ReportBuilder().build(
            guardian_with_scan, output_dir=str(tmp_path), fmt="html"
        )
        content = open(path).read()
        assert len(content) > 100

    def test_html_contains_mentis_heading(self, guardian_with_scan, tmp_path):
        path = ReportBuilder().build(
            guardian_with_scan, output_dir=str(tmp_path), fmt="html"
        )
        content = open(path).read()
        assert "Mentis" in content

    def test_html_has_inline_styles(self, guardian_with_scan, tmp_path):
        """After the CSS fix, styles must be inlined (no broken external link)."""
        path = ReportBuilder().build(
            guardian_with_scan, output_dir=str(tmp_path), fmt="html"
        )
        content = open(path).read()
        assert "<style>" in content
        # External stylesheet reference should NOT be present (it would 404)
        assert 'href="styles.css"' not in content


class TestReportBuilderMarkdown:
    def test_markdown_report_created(self, guardian_with_scan, tmp_path):
        path = ReportBuilder().build(
            guardian_with_scan, output_dir=str(tmp_path), fmt="markdown"
        )
        assert os.path.exists(path)
        assert path.endswith(".md")

    def test_markdown_contains_section_headers(self, guardian_with_scan, tmp_path):
        path = ReportBuilder().build(
            guardian_with_scan, output_dir=str(tmp_path), fmt="markdown"
        )
        content = open(path).read()
        assert "# Mentis Report" in content
        assert "## Dataset Scan" in content

    def test_markdown_write_failure_raises(self, guardian_with_scan):
        builder = ReportBuilder()
        ctx = builder.build_context(guardian_with_scan)
        with pytest.raises(ReportGenerationError, match="Failed to write Markdown"):
            builder._build_markdown(ctx, "/nonexistent_dir/report.md")


class TestReportBuilderAuditDeployment:
    def test_audit_in_report(self, guardian_with_audit, tmp_path):
        path = ReportBuilder().build(
            guardian_with_audit, output_dir=str(tmp_path), fmt="html"
        )
        content = open(path).read()
        assert "Pipeline Audit" in content or "Audit" in content

    def test_deployment_in_report(self, guardian_with_deployment, tmp_path):
        path = ReportBuilder().build(
            guardian_with_deployment, output_dir=str(tmp_path), fmt="html"
        )
        content = open(path).read()
        assert "Deployment" in content
