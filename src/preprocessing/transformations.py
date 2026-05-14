"""Transformações padrão para normalizar esquema de dados."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns


def standardize_transaction_types(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Padroniza tipos fundamentais para processamento analítico."""
    transformed = df.copy()
    transformed[columns.customer_id] = transformed[columns.customer_id].astype(str)
    transformed[columns.order_id] = transformed[columns.order_id].astype(str)
    transformed[columns.product_id] = transformed[columns.product_id].astype(str)
    transformed[columns.quantity] = pd.to_numeric(transformed[columns.quantity], errors="coerce")
    transformed[columns.unit_price] = pd.to_numeric(transformed[columns.unit_price], errors="coerce")
    transformed[columns.order_date] = pd.to_datetime(transformed[columns.order_date], errors="coerce")
    transformed = transformed.sort_values(columns.order_date)
    return transformed.reset_index(drop=True)

