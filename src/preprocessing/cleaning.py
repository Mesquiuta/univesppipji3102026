"""Data cleaning routines for transactional data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config.settings import DataColumns
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningReport:
    """Summary of removed records during cleaning."""

    input_rows: int
    removed_duplicates: int
    removed_invalid_dates_or_keys: int
    removed_invalid_numeric: int
    removed_negative_or_zero_values: int
    output_rows: int

    def as_dict(self) -> dict[str, int]:
        """Return report as a serializable dictionary."""
        return {
            "input_rows": self.input_rows,
            "removed_duplicates": self.removed_duplicates,
            "removed_invalid_dates_or_keys": self.removed_invalid_dates_or_keys,
            "removed_invalid_numeric": self.removed_invalid_numeric,
            "removed_negative_or_zero_values": self.removed_negative_or_zero_values,
            "output_rows": self.output_rows,
        }


def clean_transactions_with_report(df: pd.DataFrame, columns: DataColumns) -> tuple[pd.DataFrame, CleaningReport]:
    """Apply cleaning steps and return cleaned data plus removal report."""
    cleaned = df.copy()
    input_rows = int(cleaned.shape[0])

    dedup_subset = [columns.order_id, columns.product_id]
    before = cleaned.shape[0]
    cleaned = cleaned.drop_duplicates(subset=dedup_subset, keep="last")
    removed_duplicates = int(before - cleaned.shape[0])

    cleaned[columns.order_date] = pd.to_datetime(cleaned[columns.order_date], errors="coerce")
    before = cleaned.shape[0]
    cleaned = cleaned.dropna(
        subset=[columns.customer_id, columns.order_id, columns.product_id, columns.order_date]
    )
    removed_invalid_dates_or_keys = int(before - cleaned.shape[0])

    cleaned[columns.quantity] = pd.to_numeric(cleaned[columns.quantity], errors="coerce")
    cleaned[columns.unit_price] = pd.to_numeric(cleaned[columns.unit_price], errors="coerce")
    before = cleaned.shape[0]
    cleaned = cleaned.dropna(subset=[columns.quantity, columns.unit_price])
    removed_invalid_numeric = int(before - cleaned.shape[0])

    before = cleaned.shape[0]
    cleaned = cleaned[(cleaned[columns.quantity] > 0) & (cleaned[columns.unit_price] >= 0)]
    removed_negative_or_zero_values = int(before - cleaned.shape[0])

    report = CleaningReport(
        input_rows=input_rows,
        removed_duplicates=removed_duplicates,
        removed_invalid_dates_or_keys=removed_invalid_dates_or_keys,
        removed_invalid_numeric=removed_invalid_numeric,
        removed_negative_or_zero_values=removed_negative_or_zero_values,
        output_rows=int(cleaned.shape[0]),
    )
    logger.info("Cleaning report: %s", report.as_dict())
    return cleaned, report


def clean_transactions(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Backward compatible cleaning helper that returns only cleaned data."""
    cleaned, _ = clean_transactions_with_report(df, columns)
    return cleaned

