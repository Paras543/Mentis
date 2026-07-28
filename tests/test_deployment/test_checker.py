"""
Tests for mentis/deployment/checker.py — DeploymentChecker.
"""

from __future__ import annotations

import os

import pytest

from mentis.deployment.checker import DeploymentChecker, DeploymentResult


class TestDeploymentChecker:
    def test_returns_deployment_result(self, tmp_path):
        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        assert isinstance(result, DeploymentResult)

    def test_empty_directory_score_zero(self, tmp_path):
        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        assert result.score == 0.0

    def test_detects_dockerfile(self, tmp_path):
        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM python:3.12")

        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        assert result.score > 0
        finding = next(f for f in result.findings if f.name == "dockerfile")
        assert finding.passed is True

    def test_detects_fastapi_framework(self, tmp_path):
        app_file = tmp_path / "main.py"
        app_file.write_text("from fastapi import FastAPI\napp = FastAPI()\n@app.get('/health')\ndef health(): return {'status': 'ok'}")

        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        assert result.detected_framework == "fastapi"
        assert result.score > 0
        health_finding = next(f for f in result.findings if f.name == "health_endpoint")
        assert health_finding.passed is True

    def test_detects_flask_framework(self, tmp_path):
        app_file = tmp_path / "app.py"
        app_file.write_text("from flask import Flask\napp = Flask(__name__)\nimport logging")

        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        assert result.detected_framework == "flask"
        logging_finding = next(f for f in result.findings if f.name == "logging")
        assert logging_finding.passed is True

    def test_failed_findings_filter(self, tmp_path):
        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        failed = result.failed()
        assert len(failed) > 0
        assert all(f.passed is False for f in failed)

    def test_to_dict_keys(self, tmp_path):
        checker = DeploymentChecker()
        result = checker.check(str(tmp_path))
        d = result.to_dict()
        assert "project_path" in d
        assert "score" in d
        assert "detected_framework" in d
        assert "findings" in d
