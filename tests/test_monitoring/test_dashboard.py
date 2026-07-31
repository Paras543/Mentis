"""
Tests for mentis/monitoring/dashboard.py — ModelMonitor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mentis.exceptions import ValidationError
from mentis.monitoring.dashboard import ModelMonitor, MonitoringSnapshot


class TestModelMonitor:
    def test_returns_monitoring_snapshot(self, clean_df):
        preds = np.random.default_rng(42).uniform(0, 1, len(clean_df))
        monitor = ModelMonitor()
        snapshot = monitor.snapshot(clean_df, preds)
        assert isinstance(snapshot, MonitoringSnapshot)

    def test_empty_dataframe_raises(self):
        monitor = ModelMonitor()
        with pytest.raises(ValidationError, match="must not be empty"):
            monitor.snapshot(pd.DataFrame(), np.array([]))

    def test_prediction_distribution_keys(self, clean_df):
        preds = np.random.default_rng(42).uniform(0, 1, len(clean_df))
        snapshot = ModelMonitor().snapshot(clean_df, preds)
        dist = snapshot.prediction_distribution
        for key in ("mean", "std", "min", "max", "p25", "p50", "p75"):
            assert key in dist

    def test_degradation_detected_above_tolerance(self, clean_df):
        preds = np.array([0, 1] * 50)
        baseline = {"f1": 0.90}
        current = {"f1": 0.70}  # -22.2% drop > 5% tolerance
        snapshot = ModelMonitor().snapshot(
            clean_df,
            preds,
            current_metrics=current,
            baseline_metrics=baseline,
            degradation_tolerance=0.05,
        )
        assert snapshot.degraded is True
        assert "f1" in snapshot.performance_delta

    def test_not_degraded_within_tolerance(self, clean_df):
        preds = np.array([0, 1] * 50)
        baseline = {"f1": 0.90}
        current = {"f1": 0.88}  # -2.2% drop < 5% tolerance
        snapshot = ModelMonitor().snapshot(
            clean_df,
            preds,
            current_metrics=current,
            baseline_metrics=baseline,
            degradation_tolerance=0.05,
        )
        assert snapshot.degraded is False

    def test_data_quality_keys(self, dirty_df):
        preds = np.array([0] * len(dirty_df))
        snapshot = ModelMonitor().snapshot(dirty_df, preds)
        dq = snapshot.data_quality
        assert "row_count" in dq
        assert "missing_pct" in dq
        assert "duplicate_pct" in dq
        assert dq["missing_pct"] > 0
        assert dq["duplicate_pct"] > 0

    def test_to_dict_keys(self, clean_df):
        preds = np.array([0, 1] * 50)
        snapshot = ModelMonitor().snapshot(clean_df, preds)
        d = snapshot.to_dict()
        assert "performance_metrics" in d
        assert "performance_delta" in d
        assert "degraded" in d
        assert "prediction_distribution" in d
        assert "feature_distribution" in d
        assert "data_quality" in d
