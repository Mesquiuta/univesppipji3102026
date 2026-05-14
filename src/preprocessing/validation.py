"""Validation helpers for transaction datasets."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Ensure all required columns are present."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_non_empty(df: pd.DataFrame) -> None:
    """Ensure DataFrame is not empty."""
    if df.empty:
        raise ValueError("DataFrame is empty after ingestion or transformation.")


def validate_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str]) -> None:
    """Ensure numeric columns are convertible to numeric values."""
    invalid_columns = []
    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        if series.isna().all():
            invalid_columns.append(column)
    if invalid_columns:
        raise ValueError(f"Invalid numeric columns: {invalid_columns}")


def validate_datetime_column(df: pd.DataFrame, column: str) -> None:
    """Ensure at least one valid datetime can be parsed."""
    parsed = pd.to_datetime(df[column], errors="coerce")
    if parsed.notna().sum() == 0:
        raise ValueError(f"Column '{column}' has no valid datetime values.")


def validate_non_negative_values(df: pd.DataFrame, columns: Iterable[str]) -> None:
    """Ensure all specified numeric columns are non-negative."""
    negative_columns = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if (values < 0).any():
            negative_columns.append(column)
    if negative_columns:
        raise ValueError(f"Negative values found in columns: {negative_columns}")

