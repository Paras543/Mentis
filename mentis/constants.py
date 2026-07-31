"""
Global constants used across the Mentis library.
"""

from __future__ import annotations

MENTIS_VERSION: str = "0.1.0"


# Thresholds for the Dataset Scanner these are Based on the Standard Data

DEFAULT_MISSING_THRESHOLD: float = 0.30  # 30% missing -> flagged
DEFAULT_CORRELATION_THRESHOLD: float = 0.90  # |corr| > 0.90 -> flagged
DEFAULT_IMBALANCE_THRESHOLD: float = 0.80  # majority class ratio
DEFAULT_NEAR_ZERO_VARIANCE_FREQ_RATIO: float = 19.0  # most common / 2nd most common
DEFAULT_NEAR_ZERO_VARIANCE_UNIQUE_PCT: float = 0.10  # unique values / n rows
DEFAULT_OUTLIER_ZSCORE_THRESHOLD: float = 3.0
DEFAULT_ID_UNIQUENESS_THRESHOLD: float = 0.95  # % unique values to suspect an ID column
DEFAULT_HIGH_CARDINALITY_THRESHOLD: int = 50  # unique categories in a categorical column

# Column type classification mainly consisting of the Numerical , Datetime or Classifications

NUMERIC_DTYPES: tuple[str, ...] = ("int16", "int32", "int64", "float16", "float32", "float64")
CATEGORICAL_DTYPES: tuple[str, ...] = ("object", "category", "bool")
DATETIME_DTYPES: tuple[str, ...] = ("datetime64[ns]", "datetime64[ns, UTC]")

# Severity levels for scan findings i have marked three signals mainly info warning critical

SEVERITY_INFO: str = "info"
SEVERITY_WARNING: str = "warning"
SEVERITY_CRITICAL: str = "critical"


# Misc

RANDOM_STATE: int = 42
MAX_SAMPLE_ROWS_FOR_HEAVY_CHECKS: int = 100_000
