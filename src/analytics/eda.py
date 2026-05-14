"""Exploratory analysis utilities for transactional data."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config.settings import DataColumns
from src.segmentation.rfm import calculate_rfm


def summarize_dataset(df: pd.DataFrame, columns: DataColumns) -> dict[str, Any]:
    """Return core metadata summary."""
    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_customers": int(df[columns.customer_id].nunique()),
        "n_orders": int(df[columns.order_id].nunique()),
        "n_products": int(df[columns.product_id].nunique()),
        "date_min": str(df[columns.order_date].min()),
        "date_max": str(df[columns.order_date].max()),
        "missing_ratio": {k: float(v) for k, v in (df.isna().mean()).to_dict().items()},
    }


def numeric_describe(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Return descriptive stats for numeric metrics."""
    numeric_cols = [columns.quantity, columns.unit_price]
    if columns.total_value in df.columns:
        numeric_cols.append(columns.total_value)
    return df[numeric_cols].describe().transpose().reset_index(names="metric")


def revenue_by_month(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Aggregate revenue by month."""
    monthly = (
        df.assign(month=df[columns.order_date].dt.to_period("M").dt.to_timestamp())
        .groupby("month", as_index=False)[columns.total_value]
        .sum()
        .rename(columns={columns.total_value: "revenue"})
        .sort_values("month")
    )
    return monthly


def average_ticket_by_month(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Calculate average order ticket by month."""
    orders = (
        df.groupby([columns.order_id, columns.order_date], as_index=False)[columns.total_value]
        .sum()
        .assign(month=lambda x: x[columns.order_date].dt.to_period("M").dt.to_timestamp())
    )
    monthly_ticket = (
        orders.groupby("month", as_index=False)[columns.total_value]
        .mean()
        .rename(columns={columns.total_value: "avg_ticket"})
        .sort_values("month")
    )
    return monthly_ticket


def orders_per_customer(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Return distribution of order count per customer."""
    customer_orders = (
        df.groupby(columns.customer_id)[columns.order_id]
        .nunique()
        .rename("orders")
        .reset_index()
        .sort_values("orders", ascending=False)
    )
    return customer_orders


def top_products_by_sales(df: pd.DataFrame, columns: DataColumns, top_n: int = 10) -> pd.DataFrame:
    """Return top products by revenue and quantity."""
    return (
        df.groupby(columns.product_id, as_index=False)
        .agg(
            quantity=(columns.quantity, "sum"),
            revenue=(columns.total_value, "sum"),
            orders=(columns.order_id, "nunique"),
        )
        .sort_values(["revenue", "quantity"], ascending=False)
        .head(top_n)
    )


def top_customers_by_value(df: pd.DataFrame, columns: DataColumns, top_n: int = 10) -> pd.DataFrame:
    """Return top customers by generated revenue."""
    return (
        df.groupby(columns.customer_id, as_index=False)
        .agg(
            revenue=(columns.total_value, "sum"),
            orders=(columns.order_id, "nunique"),
            products=(columns.product_id, "nunique"),
        )
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )


def customer_revenue_concentration(df: pd.DataFrame, columns: DataColumns) -> dict[str, float]:
    """Estimate concentration of revenue by top customer groups."""
    customer_revenue = (
        df.groupby(columns.customer_id)[columns.total_value]
        .sum()
        .sort_values(ascending=False)
    )
    if customer_revenue.empty:
        return {"top_10pct_share": 0.0, "top_20pct_share": 0.0, "gini_like_index": 0.0}

    n_customers = len(customer_revenue)
    top_10_count = max(1, int(np.ceil(n_customers * 0.10)))
    top_20_count = max(1, int(np.ceil(n_customers * 0.20)))
    total_revenue = float(customer_revenue.sum()) or 1.0

    top_10_share = float(customer_revenue.head(top_10_count).sum() / total_revenue)
    top_20_share = float(customer_revenue.head(top_20_count).sum() / total_revenue)

    sorted_values = customer_revenue.sort_values().to_numpy(dtype=float)
    n = len(sorted_values)
    cum = (np.arange(1, n + 1) * sorted_values).sum()
    gini_like = float((2 * cum) / (n * sorted_values.sum()) - (n + 1) / n) if sorted_values.sum() else 0.0

    return {
        "top_10pct_share": top_10_share,
        "top_20pct_share": top_20_share,
        "gini_like_index": gini_like,
    }


def rfm_distribution(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Return customer-level RFM values for distribution analysis."""
    return calculate_rfm(df, columns)


def detect_simple_outliers(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Detect IQR outliers for key numeric fields."""
    numeric_cols = [columns.quantity, columns.unit_price, columns.total_value]
    records: list[dict[str, float]] = []
    for metric in numeric_cols:
        series = pd.to_numeric(df[metric], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((series < lower) | (series > upper)).sum())
        records.append(
            {
                "metric": metric,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": outlier_count,
                "outlier_ratio": float(outlier_count / len(series)),
            }
        )
    return pd.DataFrame(records)

