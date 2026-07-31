"""
Tests for mentis/scanner/dataset_scanner.py — the DatasetScanner orchestrator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from mentis.exceptions import DatasetError
from mentis.scanner.dataset_scanner import DatasetScanner
from mentis.scanner.result import ScanResult


class TestDatasetScannerValidation:
    def test_raises_on_non_dataframe(self):
        scanner = DatasetScanner()
        with pytest.raises(DatasetError, match="pandas DataFrame"):
            scanner.scan([1, 2, 3])  # type: ignore[arg-type]

    def test_raises_on_empty_dataframe(self):
        scanner = DatasetScanner()
        with pytest.raises(DatasetError, match="empty"):
            scanner.scan(pd.DataFrame())

    def test_raises_when_target_not_in_columns(self, clean_df):
        scanner = DatasetScanner()
        with pytest.raises(DatasetError, match="not found"):
            scanner.scan(clean_df, target="nonexistent_column")


class TestDatasetScannerResult:
    def test_returns_scan_result(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        assert isinstance(result, ScanResult)

    def test_result_shape_matches_dataframe(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        assert result.n_rows == len(clean_df)
        assert result.n_columns == clean_df.shape[1]

    def test_memory_usage_positive(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        assert result.memory_usage_mb > 0

    def test_summary_keys_present(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        assert "total_findings" in result.summary
        assert "severity_counts" in result.summary
        assert "checks_triggered" in result.summary

    def test_column_profiles_count(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        assert len(result.column_profiles) == clean_df.shape[1]

    def test_column_profiles_names(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        profile_names = {p.name for p in result.column_profiles}
        assert profile_names == set(clean_df.columns)


class TestDatasetScannerFindings:
    def test_dirty_df_has_findings(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        assert result.summary["total_findings"] > 0

    def test_clean_df_fewer_findings_than_dirty(self, clean_df, dirty_df):
        r_clean = DatasetScanner().scan(clean_df)
        r_dirty = DatasetScanner().scan(dirty_df)
        assert r_dirty.summary["total_findings"] > r_clean.summary["total_findings"]

    def test_constant_column_flagged(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        check_names = {f.check_name for f in result.findings}
        assert "constant_columns" in check_names

    def test_duplicate_rows_flagged(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        check_names = {f.check_name for f in result.findings}
        assert "duplicate_rows" in check_names

    def test_missing_values_flagged(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        check_names = {f.check_name for f in result.findings}
        assert "missing_values" in check_names

    def test_target_imbalance_requires_target_arg(self, imbalanced_df):
        # Without target arg: no imbalance finding
        result_no_target = DatasetScanner().scan(imbalanced_df)
        r_with_target = DatasetScanner().scan(imbalanced_df, target="target")
        no_target_names = {f.check_name for f in result_no_target.findings}
        with_target_names = {f.check_name for f in r_with_target.findings}
        assert "target_imbalance" not in no_target_names
        assert "target_imbalance" in with_target_names

    def test_data_leakage_flagged(self, leakage_df):
        result = DatasetScanner().scan(leakage_df, target="target")
        check_names = {f.check_name for f in result.findings}
        assert "data_leakage" in check_names


class TestDatasetScannerFilterMethods:
    def test_critical_findings_method(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        criticals = result.critical_findings()
        assert all(f.severity == "critical" for f in criticals)

    def test_warnings_method(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        warnings = result.warnings()
        assert all(f.severity == "warning" for f in warnings)

    def test_info_findings_method(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        infos = result.info_findings()
        assert all(f.severity == "info" for f in infos)


class TestDatasetScannerColumnProfiles:
    def test_profile_role_target(self):
        df = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "label": [0, 1, 0]})
        result = DatasetScanner().scan(df, target="label")
        label_profile = next(p for p in result.column_profiles if p.name == "label")
        assert label_profile.role == "target"

    def test_profile_role_numerical(self, clean_df):
        result = DatasetScanner().scan(clean_df)
        age_profile = next(p for p in result.column_profiles if p.name == "age")
        assert age_profile.role == "numerical"

    def test_profile_missing_count_accurate(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        age_profile = next(p for p in result.column_profiles if p.name == "age")
        assert age_profile.missing_count == int(dirty_df["age"].isnull().sum())

    def test_constant_profile_flagged(self, dirty_df):
        result = DatasetScanner().scan(dirty_df)
        const_profile = next(p for p in result.column_profiles if p.name == "constant_col")
        assert const_profile.is_constant is True


class TestDatasetScannerCustomChecks:
    def test_custom_check_list(self, clean_df):
        from mentis.scanner.checks import DuplicateRowsCheck

        scanner = DatasetScanner(checks=[DuplicateRowsCheck()])
        result = scanner.scan(clean_df)
        # Only DuplicateRows can fire; clean_df has none
        assert result.summary["total_findings"] == 0

    def test_bad_check_doesnt_crash_scanner(self, clean_df):
        """A check that throws must not prevent others from running."""
        from mentis.scanner.base import BaseCheck

        class BrokenCheck(BaseCheck):
            name = "broken"

            def run(self, df, **ctx):
                raise RuntimeError("intentional failure")

        from mentis.scanner.checks import DuplicateRowsCheck

        scanner = DatasetScanner(checks=[BrokenCheck(), DuplicateRowsCheck()])
        # Should not raise; BrokenCheck failure is logged and ignored
        result = scanner.scan(clean_df)
        assert isinstance(result, ScanResult)
