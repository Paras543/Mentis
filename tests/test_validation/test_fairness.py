"""
Tests for mentis/validation/fairness.py — BiasDetector.

Fairlearn is an optional dependency. Tests are skipped when not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

fairlearn = pytest.importorskip("fairlearn", reason="fairlearn not installed — skipping bias tests")

from mentis.exceptions import ValidationError
from mentis.validation.fairness import BiasDetector, FairnessResult


class TestBiasDetectorValidation:
    def test_empty_sensitive_features_raises(self):
        detector = BiasDetector()
        with pytest.raises(ValidationError, match="sensitive feature"):
            detector.detect(np.array([0, 1]), np.array([0, 1]), {})

    def test_length_mismatch_raises(self):
        detector = BiasDetector()
        with pytest.raises(ValidationError, match="length"):
            detector.detect(
                y_true=np.array([0, 1, 0]),
                y_pred=np.array([0, 1, 0]),
                sensitive_features={"group": np.array(["a", "b"])},  # length 2 vs 3
            )


class TestBiasDetectorRun:
    def test_returns_fairness_result_list(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        detector = BiasDetector()
        results = detector.detect(y_true, y_pred, sensitive_features={"group": groups})

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], FairnessResult)

    def test_result_fields_populated(self):
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 0, 0, 1])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        results = BiasDetector().detect(y_true, y_pred, sensitive_features={"group": groups})
        res = results[0]

        assert res.sensitive_feature == "group"
        assert isinstance(res.demographic_parity_difference, float)
        assert isinstance(res.equal_opportunity_difference, float)
        assert isinstance(res.equalized_odds_difference, float)
        assert "A" in res.selection_rate_by_group
        assert "B" in res.selection_rate_by_group

    def test_to_dict_keys(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        groups = np.array(["A", "A", "B", "B"])

        results = BiasDetector().detect(y_true, y_pred, sensitive_features={"group": groups})
        d = results[0].to_dict()

        for key in ("sensitive_feature", "demographic_parity_difference",
                    "equal_opportunity_difference", "equalized_odds_difference",
                    "selection_rate_by_group"):
            assert key in d
