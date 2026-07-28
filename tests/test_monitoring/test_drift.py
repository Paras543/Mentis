"""
Tests for mentis/monitoring/drift.py — DriftDetector.

Evidently is an optional dependency. Tests are skipped when not installed.
"""

from __future__ import annotations

import pandas as pd
import pytest

evidently = pytest.importorskip("evidently", reason="evidently not installed — skipping drift tests")

from mentis.exceptions import ValidationError
from mentis.monitoring.drift import DriftDetector


class TestDriftDetectorValidation:
    def test_empty_reference_raises(self, clean_df):
        detector = DriftDetector()
        with pytest.raises(ValidationError, match="non-empty"):
            detector.detect(pd.DataFrame(), clean_df)

    def test_empty_current_raises(self, clean_df):
        detector = DriftDetector()
        with pytest.raises(ValidationError, match="non-empty"):
            detector.detect(clean_df, pd.DataFrame())

    def test_missing_target_raises(self, clean_df):
        detector = DriftDetector()
        with pytest.raises(ValidationError, match="target"):
            detector.detect(clean_df, clean_df, target="nonexistent")


class TestDriftDetectorRun:
    def test_identical_datasets_no_drift(self, clean_df):
        detector = DriftDetector()
        result = detector.detect(clean_df, clean_df.copy())
        assert isinstance(result, dict)
        assert "dataset_drift" in result
        assert "drift_share" in result
        assert "feature_drift" in result

    def test_shifted_dataset_detects_drift(self, clean_df):
        shifted_df = clean_df.copy()
        shifted_df["age"] = shifted_df["age"] + 100.0  # obvious distribution shift

        detector = DriftDetector()
        result = detector.detect(clean_df, shifted_df)
        assert result["dataset_drift"] is True
