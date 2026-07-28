"""
Shared input-validation helpers used across Mentis modules.

Centralizing validation logic here avoids duplicating the same
`isinstance` / empty-check boilerplate in every scanner, comparator,
and reporter.
"""

from __future__ import annotations

import pandas as pd

from mentis.exceptions import DatasetError, ValidationError


def validate_dataframe(df: pd.DataFrame, allow_empty: bool = False) -> None:
    """
    Validate that `df` is a usable pandas DataFrame.

    Args:
        df: Object to validate.
        allow_empty: If False (default), raises when `df` has zero rows.

    Raises:
        DatasetError: If `df` is not a DataFrame, or is empty and
            `allow_empty` is False.

    Examples:
        >>> import pandas as pd
        >>> validate_dataframe(pd.DataFrame({"a": [1]}))
    """
    if not isinstance(df, pd.DataFrame):
        raise DatasetError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
    if not allow_empty and df.empty:
        raise DatasetError("DataFrame is empty.")


def validate_column_exists(df: pd.DataFrame, column: str, label: str = "column") -> None:
    """
    Validate that a named column exists in `df`.

    Args:
        df: DataFrame to check.
        column: Column name expected to be present.
        label: Human-readable label used in the error message
            (e.g. "target", "prediction column").

    Raises:
        ValidationError: If `column` is not found in `df.columns`.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({"churn": [0, 1]})
        >>> validate_column_exists(df, "churn", label="target")
    """
    if column not in df.columns:
        raise ValidationError(f"Expected {label} '{column}' was not found in dataframe columns.")


def validate_matching_length(*arrays: object, names: list[str] | None = None) -> None:
    """
    Validate that all provided array-likes have the same length.

    Args:
        *arrays: Any objects supporting `len()` (arrays, Series, lists).
        names: Optional human-readable names matching `arrays`, used to
            produce a clearer error message.

    Raises:
        ValidationError: If lengths differ.

    Examples:
        >>> validate_matching_length([1, 2, 3], [4, 5, 6])
    """
    lengths = [len(a) for a in arrays]  
    if len(set(lengths)) > 1:
        labels = names or [f"array_{i}" for i in range(len(arrays))]
        pairs = ", ".join(f"{n}={l}" for n, l in zip(labels, lengths))
        raise ValidationError(f"Length mismatch between inputs: {pairs}.")


def validate_task_type(task: str) -> None:
    """
    Validate that `task` is a supported ML task type.

    Args:
        task: Task type string, expected to be "classification" or
            "regression".

    Raises:
        ValidationError: If `task` is not a recognized value.

    Examples:
        >>> validate_task_type("classification")
    """
    valid_tasks = {"classification", "regression"}
    if task not in valid_tasks:
        raise ValidationError(f"Unsupported task '{task}'. Must be one of {sorted(valid_tasks)}.")
    

    


