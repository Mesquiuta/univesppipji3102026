"""Cálculo de métricas RFM (Recência, Frequência e Monetário)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.config.settings import DataColumns


def calculate_rfm(
    df: pd.DataFrame,
    columns: DataColumns,
    reference_date: datetime | None = None,
) -> pd.DataFrame:
    """Constrói tabela RFM por cliente."""
    if reference_date is None:
        max_date = pd.to_datetime(df[columns.order_date]).max()
        reference_date = (max_date + pd.Timedelta(days=1)).to_pydatetime()

    customer_group = df.groupby(columns.customer_id)
    rfm = customer_group.agg(
        last_order_date=(columns.order_date, "max"),
        frequency=(columns.order_id, "nunique"),
        monetary=(columns.total_value, "sum"),
    ).reset_index()

    rfm["recency"] = (pd.Timestamp(reference_date) - pd.to_datetime(rfm["last_order_date"])).dt.days
    rfm = rfm[[columns.customer_id, "recency", "frequency", "monetary"]]
    return rfm.sort_values(columns.customer_id).reset_index(drop=True)

