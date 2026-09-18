"""Shared input validation."""

from __future__ import annotations

from numbers import Real

import numpy as np
import pandas as pd


def finite_scalar(value: object, name: str, *, positive: bool = False) -> float:
    """Return *value* as a validated finite float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a numeric scalar")

    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return result


def numeric_frame(
    frame: object,
    name: str,
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Validate a DataFrame and its required numeric columns."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} must contain at least one row")

    result = frame.copy()
    for column in required_columns:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise TypeError(f"{name}.{column} must be numeric")
        values = result[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name}.{column} contains missing or non-finite values")
    return result

