"""
General-purpose helper functions used across multiple Mentis modules.

These are small, pure, dependency-light utilities -- anything with
heavier logic belongs in its own dedicated module instead.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd

from mentis.constants import CATEGORICAL_DTYPES, DATETIME_DTYPES, NUMERIC_DTYPES


def split_columns_by_type(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Split a dataframe's columns into numerical, categorical, and
    datetime groups based on dtype.

    Args:
        df: The dataframe to inspect.

    Returns:
        A dict with keys "numerical", "categorical", "datetime", each
        mapping to a list of column names.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"age": [1, 2], "city": ["a", "b"]})
        >>> split_columns_by_type(df)
        {'numerical': ['age'], 'categorical': ['city'], 'datetime': []}
    """
    numerical: list[str] = []
    categorical: list[str] = []
    datetime_cols: list[str] = []

    for col in df.columns:
        dtype_str = str(df[col].dtype)
        if dtype_str in DATETIME_DTYPES or "datetime" in dtype_str:
            datetime_cols.append(str(col))
        elif dtype_str in NUMERIC_DTYPES:
            numerical.append(str(col))
        elif dtype_str in CATEGORICAL_DTYPES:
            categorical.append(str(col))

    return {"numerical": numerical, "categorical": categorical, "datetime": datetime_cols}


def bytes_to_mb(n_bytes: int | float) -> float:
    """
    Convert a byte count to megabytes.

    Args:
        n_bytes: Size in bytes.

    Returns:
        Size in megabytes (base-1024).

    Examples:
        >>> bytes_to_mb(1_048_576)
        1.0
    """
    return float(n_bytes) / 1_048_576


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide two numbers, returning `default` instead of raising on
    division by zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value to return if `denominator` is zero.

    Returns:
        `numerator / denominator`, or `default` if denominator is 0.

    Examples:
        >>> safe_divide(10, 0)
        0.0
        >>> safe_divide(10, 2)
        5.0
    """
    if denominator == 0:
        return default
    return numerator / denominator


def ensure_directory(path: str) -> str:
    """
    Ensure a directory exists, creating it (and parents) if needed.

    Args:
        path: Directory path to create.

    Returns:
        The same `path`, for convenient chaining.

    Examples:
        >>> ensure_directory("/tmp/mentis_reports")  # doctest: +SKIP
    """
    os.makedirs(path, exist_ok=True)
    return path


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    """
    Context manager that measures elapsed wall-clock time.

    Yields a dict that is populated with an "elapsed_seconds" key once
    the `with` block exits -- useful for timing model training or
    inference without cluttering business logic with `time.time()`
    calls.

    Examples:
        >>> with timer() as t:
        ...     _ = sum(range(1000))
        >>> "elapsed_seconds" in t
        True
    """
    result: dict[str, float] = {}
    start = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_seconds"] = time.perf_counter() - start


def detect_task_type(y: pd.Series | np.ndarray) -> str:
    """
    Heuristically infer whether a target variable represents a
    classification or regression task.

    Args:
        y: Target values.

    Returns:
        "classification" if `y` has a small number of discrete values
        relative to its length, or a non-numeric/categorical dtype;
        otherwise "regression".

    Examples:
        >>> import pandas as pd
        >>> detect_task_type(pd.Series([0, 1, 0, 1, 1]))
        'classification'
        >>> detect_task_type(pd.Series([1.2, 3.4, 5.6, 7.8]))
        'regression'
    """
    series = pd.Series(y)

    if series.dtype == object or str(series.dtype) == "category" or series.dtype == bool:
        return "classification"

    n_unique = series.nunique(dropna=True)
    n_total = len(series)

    if n_unique <= 20 or (n_total > 0 and n_unique / n_total < 0.05):
        return "classification"

    return "regression"


def truncate_dataframe_for_heavy_ops(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """
    Return a deterministic sample of `df` if it exceeds `max_rows`,
    otherwise return it unchanged. Used to keep expensive checks
    (e.g. pairwise correlation, outlier detection) tractable on very
    large datasets.

    Args:
        df: The dataframe to potentially sample.
        max_rows: Maximum number of rows to keep.

    Returns:
        `df` itself if within `max_rows`, else a random sample of
        exactly `max_rows` rows (fixed random_state for reproducibility).

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"a": range(10)})
        >>> len(truncate_dataframe_for_heavy_ops(df, 5))
        5
    """
    if len(df) <= max_rows:
        return df
    return df.sample(n=max_rows, random_state=42)
