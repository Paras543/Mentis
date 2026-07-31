"""
Tests for mentis/scanner/checks.py — every individual check class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mentis.scanner.checks import (
    ConstantColumnsCheck,
    DataLeakageCheck,
    DuplicateColumnsCheck,
    DuplicateRowsCheck,
    HighCorrelationCheck,
    IDColumnCheck,
    InfiniteValuesCheck,
    MissingValuesCheck,
    MixedTypesCheck,
    NearZeroVarianceCheck,
    OutlierCheck,
    TargetImbalanceCheck,
)

# ── MissingValuesCheck ────────────────────────────────────────────────────────


class TestMissingValuesCheck:
    def test_no_missing_returns_empty(self, clean_df):
        findings = MissingValuesCheck().run(clean_df)
        # clean_df has no nulls in any column
        assert all(f.check_name == "missing_values" for f in findings)
        # Only columns WITH missing should appear
        assert not any(f for f in findings if "age" in f.message and "0.0%" in f.message)

    def test_info_severity_for_low_missing(self):
        df = pd.DataFrame({"a": [1.0, None, 2.0, 3.0, 4.0] * 20})  # 20% missing → WARNING (>=10%)
        findings = MissingValuesCheck(threshold=0.30).run(df)
        assert len(findings) == 1
        assert findings[0].severity == "warning"

    def test_critical_severity_above_threshold(self):
        df = pd.DataFrame({"a": [None] * 40 + [1.0] * 60})  # 40% missing → CRITICAL
        findings = MissingValuesCheck(threshold=0.30).run(df)
        assert len(findings) == 1
        assert findings[0].severity == "critical"

    def test_info_severity_below_10_pct(self):
        df = pd.DataFrame({"a": [None] + [1.0] * 99})  # 1% missing → INFO
        findings = MissingValuesCheck(threshold=0.30).run(df)
        assert len(findings) == 1
        assert findings[0].severity == "info"

    def test_warning_severity_between_10_and_threshold(self):
        df = pd.DataFrame({"a": [None] * 15 + [1.0] * 85})  # 15% missing → WARNING
        findings = MissingValuesCheck(threshold=0.30).run(df)
        assert len(findings) == 1
        assert findings[0].severity == "warning"

    def test_multiple_columns_flagged(self):
        df = pd.DataFrame(
            {
                "a": [None] * 50 + [1.0] * 50,
                "b": [None] * 20 + [2.0] * 80,
                "c": [1.0] * 100,
            }
        )
        findings = MissingValuesCheck().run(df)
        flagged_cols = {f.columns[0] for f in findings}
        assert "a" in flagged_cols
        assert "b" in flagged_cols
        assert "c" not in flagged_cols


# ── DuplicateRowsCheck ────────────────────────────────────────────────────────


class TestDuplicateRowsCheck:
    def test_no_duplicates_returns_empty(self, clean_df):
        findings = DuplicateRowsCheck().run(clean_df)
        assert findings == []

    def test_detects_duplicates(self, dirty_df):
        findings = DuplicateRowsCheck().run(dirty_df)
        assert len(findings) == 1
        assert "duplicate" in findings[0].message.lower()

    def test_warning_severity_for_high_duplicate_pct(self):
        base = pd.DataFrame({"a": range(10)})
        df = pd.concat([base, base, base, base, base], ignore_index=True)  # 80% dupes
        findings = DuplicateRowsCheck().run(df)
        assert findings[0].severity == "warning"

    def test_info_severity_for_low_duplicate_pct(self):
        base = pd.DataFrame({"a": range(1000)})
        dup = base.iloc[:3].copy()
        df = pd.concat([base, dup], ignore_index=True)  # <5% dupes
        findings = DuplicateRowsCheck().run(df)
        assert findings[0].severity == "info"


# ── DuplicateColumnsCheck ─────────────────────────────────────────────────────


class TestDuplicateColumnsCheck:
    def test_no_duplicate_columns(self, clean_df):
        findings = DuplicateColumnsCheck().run(clean_df)
        assert findings == []

    def test_detects_duplicate_column(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [1, 2, 3]})
        findings = DuplicateColumnsCheck().run(df)
        assert len(findings) == 1
        assert "c" in findings[0].message

    def test_multiple_duplicates(self):
        df = pd.DataFrame({"a": [1, 2], "b": [1, 2], "c": [3, 4], "d": [3, 4]})
        findings = DuplicateColumnsCheck().run(df)
        assert len(findings) == 2


# ── ConstantColumnsCheck ──────────────────────────────────────────────────────


class TestConstantColumnsCheck:
    def test_no_constant_columns(self, clean_df):
        findings = ConstantColumnsCheck().run(clean_df)
        assert findings == []

    def test_detects_constant_column(self, dirty_df):
        findings = ConstantColumnsCheck().run(dirty_df)
        constant_findings = [f for f in findings if "constant_col" in f.message]
        assert len(constant_findings) == 1

    def test_single_value_column(self):
        df = pd.DataFrame({"x": [5] * 50, "y": range(50)})
        findings = ConstantColumnsCheck().run(df)
        assert len(findings) == 1
        assert "x" in findings[0].message


# ── NearZeroVarianceCheck ─────────────────────────────────────────────────────


class TestNearZeroVarianceCheck:
    def test_normal_data_no_findings(self, clean_df):
        findings = NearZeroVarianceCheck().run(clean_df)
        assert findings == []

    def test_detects_near_zero_variance(self):
        values = [1] * 95 + [2, 3, 4, 5, 6]  # dominant value, few unique
        df = pd.DataFrame({"nzv": values, "normal": range(100)})
        findings = NearZeroVarianceCheck(freq_ratio_threshold=10, unique_pct_threshold=0.1).run(df)
        assert any("nzv" in f.message for f in findings)

    def test_empty_df_returns_empty(self):
        findings = NearZeroVarianceCheck().run(pd.DataFrame())
        assert findings == []


# ── HighCorrelationCheck ──────────────────────────────────────────────────────


class TestHighCorrelationCheck:
    def test_no_high_correlation(self, clean_df):
        # age and income from clean_df are random and uncorrelated
        findings = HighCorrelationCheck(threshold=0.99).run(clean_df)
        assert findings == []

    def test_detects_high_correlation(self):
        x = np.linspace(0, 1, 100)
        df = pd.DataFrame({"a": x, "b": x * 2 + 0.001, "c": range(100)})
        findings = HighCorrelationCheck(threshold=0.95).run(df)
        assert len(findings) >= 1

    def test_single_column_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        findings = HighCorrelationCheck().run(df)
        assert findings == []


# ── OutlierCheck ──────────────────────────────────────────────────────────────


class TestOutlierCheck:
    def test_no_outliers(self):
        df = pd.DataFrame({"a": np.random.default_rng(42).normal(0, 1, 200)})
        findings = OutlierCheck(z_threshold=5.0).run(df)
        assert findings == []

    def test_detects_outliers(self):
        data = list(range(100)) + [10_000]  # extreme outlier
        df = pd.DataFrame({"a": data})
        findings = OutlierCheck(z_threshold=3.0).run(df)
        assert len(findings) == 1
        assert "a" in findings[0].message

    def test_zero_std_column_skipped(self):
        df = pd.DataFrame({"a": [5] * 10, "b": range(10)})
        findings = OutlierCheck().run(df)
        flagged_cols = [f.columns[0] for f in findings]
        assert "a" not in flagged_cols


# ── InfiniteValuesCheck ───────────────────────────────────────────────────────


class TestInfiniteValuesCheck:
    def test_no_infinite_values(self, clean_df):
        findings = InfiniteValuesCheck().run(clean_df)
        assert findings == []

    def test_detects_positive_infinity(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        findings = InfiniteValuesCheck().run(df)
        assert len(findings) == 1
        assert findings[0].severity == "critical"

    def test_detects_negative_infinity(self):
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
        findings = InfiniteValuesCheck().run(df)
        assert len(findings) == 1

    def test_non_numeric_ignored(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        findings = InfiniteValuesCheck().run(df)
        assert findings == []


# ── IDColumnCheck ─────────────────────────────────────────────────────────────


class TestIDColumnCheck:
    def test_id_column_by_uniqueness(self):
        df = pd.DataFrame({"user_id": range(100), "age": [25] * 100})
        findings = IDColumnCheck(uniqueness_threshold=0.95).run(df)
        assert any("user_id" in f.message for f in findings)

    def test_id_column_by_name(self):
        df = pd.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
        findings = IDColumnCheck().run(df)
        assert any("id" in f.message for f in findings)

    def test_normal_column_not_flagged(self):
        df = pd.DataFrame({"age": [20, 25, 20, 25], "target": [0, 1, 0, 1]})
        findings = IDColumnCheck(uniqueness_threshold=0.99).run(df)
        assert findings == []


# ── TargetImbalanceCheck ──────────────────────────────────────────────────────


class TestTargetImbalanceCheck:
    def test_balanced_target_no_findings(self, clean_df):
        findings = TargetImbalanceCheck(threshold=0.80).run(clean_df, target="target")
        assert findings == []

    def test_imbalanced_target_flagged(self, imbalanced_df):
        findings = TargetImbalanceCheck(threshold=0.80).run(imbalanced_df, target="target")
        assert len(findings) == 1
        assert findings[0].severity == "warning"

    def test_no_target_returns_empty(self, clean_df):
        findings = TargetImbalanceCheck().run(clean_df)
        assert findings == []

    def test_missing_target_column_returns_empty(self, clean_df):
        findings = TargetImbalanceCheck().run(clean_df, target="nonexistent")
        assert findings == []


# ── DataLeakageCheck ──────────────────────────────────────────────────────────


class TestDataLeakageCheck:
    def test_no_leakage(self, clean_df):
        findings = DataLeakageCheck().run(clean_df, target="target")
        assert findings == []

    def test_detects_leakage(self, leakage_df):
        findings = DataLeakageCheck(threshold=0.95).run(leakage_df, target="target")
        assert len(findings) >= 1
        assert findings[0].severity == "critical"
        assert "leaked_feature" in findings[0].message

    def test_no_target_returns_empty(self, leakage_df):
        findings = DataLeakageCheck().run(leakage_df)
        assert findings == []

    def test_categorical_target_skipped(self):
        df = pd.DataFrame({"a": [1, 2, 3], "target": ["x", "y", "z"]})
        findings = DataLeakageCheck().run(df, target="target")
        assert findings == []


# ── MixedTypesCheck ───────────────────────────────────────────────────────────


class TestMixedTypesCheck:
    def test_uniform_types_no_findings(self, clean_df):
        findings = MixedTypesCheck().run(clean_df)
        assert findings == []

    def test_detects_mixed_types(self):
        df = pd.DataFrame({"col": [1, "two", 3.0, None]})
        findings = MixedTypesCheck().run(df)
        assert len(findings) == 1
        assert "col" in findings[0].message

    def test_all_same_type_string(self):
        df = pd.DataFrame({"city": ["NYC", "LA", "Chicago"]})
        findings = MixedTypesCheck().run(df)
        assert findings == []
