"""Ingestion and preprocessing pipeline."""

from __future__ import annotations

from typing import Any

from src.config.settings import AppSettings, load_settings
from src.ingestion.loader import load_dataset
from src.preprocessing.cleaning import clean_transactions_with_report
from src.preprocessing.feature_engineering import build_features
from src.preprocessing.transformations import standardize_transaction_types
from src.preprocessing.validation import (
    validate_datetime_column,
    validate_non_empty,
    validate_non_negative_values,
    validate_numeric_columns,
    validate_required_columns,
)
from src.utils.io_helpers import save_dataframe
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def run(settings: AppSettings | None = None) -> dict[str, Any]:
    """Run data ingestion, validation, cleaning, and feature engineering."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    source_path = settings.paths.data_raw / settings.data.source_filename
    interim_path = settings.paths.data_interim / settings.data.interim_filename
    processed_path = settings.paths.data_processed / settings.data.processed_filename
    logger.info("Starting preprocessing with source %s", source_path)

    df = load_dataset(source_path, encoding=settings.data.encoding)
    validate_non_empty(df)
    validate_required_columns(df, settings.columns.required())
    validate_datetime_column(df, settings.columns.order_date)
    validate_numeric_columns(df, [settings.columns.quantity, settings.columns.unit_price])

    cleaned, cleaning_report = clean_transactions_with_report(df, settings.columns)
    transformed = standardize_transaction_types(cleaned, settings.columns)
    featured = build_features(transformed, settings.columns)
    validate_non_empty(featured)
    validate_non_negative_values(featured, [settings.columns.quantity, settings.columns.unit_price, settings.columns.total_value])
    if featured[settings.columns.order_date].isna().any():
        raise ValueError("Invalid order_date values remained after preprocessing.")

    save_dataframe(cleaned, interim_path, index=False)
    save_dataframe(featured, processed_path, index=False)

    result = {
        "source_path": str(source_path),
        "interim_path": str(interim_path),
        "processed_path": str(processed_path),
        "n_rows_processed": int(featured.shape[0]),
        "cleaning_report": cleaning_report.as_dict(),
    }
    logger.info("Preprocessing completed with %s rows.", featured.shape[0])
    return result


if __name__ == "__main__":
    run()

