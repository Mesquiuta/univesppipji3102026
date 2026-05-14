"""Retention and repeat-purchase proxies based on transaction data."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns


def build_repeat_purchase_funnel(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """
    Build a repeat-purchase funnel proxy from order frequency per customer.
    This is intentionally not a full behavioral journey funnel.
    """
    order_counts = df.groupby(columns.customer_id)[columns.order_id].nunique()
    stage_counts = {
        "customers_1plus_order": int((order_counts >= 1).sum()),
        "customers_2plus_orders": int((order_counts >= 2).sum()),
        "customers_3plus_orders": int((order_counts >= 3).sum()),
    }
    funnel = pd.DataFrame({"stage": list(stage_counts.keys()), "count": list(stage_counts.values())})
    funnel["conversion_from_previous"] = funnel["count"] / funnel["count"].shift(1)
    funnel.loc[0, "conversion_from_previous"] = 1.0
    funnel["drop_off_from_previous"] = 1 - funnel["conversion_from_previous"]
    return funnel


def purchase_frequency_distribution(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Return customer frequency distribution by number of unique orders."""
    order_counts = df.groupby(columns.customer_id)[columns.order_id].nunique()
    distribution = (
        order_counts.value_counts()
        .rename_axis("orders_per_customer")
        .reset_index(name="n_customers")
        .sort_values("orders_per_customer")
    )
    distribution["customer_share"] = distribution["n_customers"] / distribution["n_customers"].sum()
    return distribution


def average_days_between_orders(df: pd.DataFrame, columns: DataColumns) -> float:
    """Calculate average gap in days between sequential customer orders."""
    order_level = (
        df.groupby([columns.customer_id, columns.order_id], as_index=False)[columns.order_date]
        .max()
        .sort_values([columns.customer_id, columns.order_date])
    )
    order_level["previous_order_date"] = order_level.groupby(columns.customer_id)[columns.order_date].shift(1)
    order_level["days_since_previous"] = (
        order_level[columns.order_date] - order_level["previous_order_date"]
    ).dt.days
    gaps = order_level["days_since_previous"].dropna()
    if gaps.empty:
        return 0.0
    return float(gaps.mean())


def calculate_repurchase_metrics(df: pd.DataFrame, columns: DataColumns) -> dict[str, float]:
    """Calculate basic repeat-purchase and retention proxy metrics."""
    order_counts = df.groupby(columns.customer_id)[columns.order_id].nunique()
    total_customers = int(order_counts.shape[0]) or 1
    one_order_customers = int((order_counts == 1).sum())
    repeat_customers = int((order_counts >= 2).sum())
    repurchase_rate = repeat_customers / total_customers

    return {
        "total_customers": float(total_customers),
        "one_order_customers": float(one_order_customers),
        "repeat_customers": float(repeat_customers),
        "repurchase_rate": float(repurchase_rate),
        "avg_days_between_orders": average_days_between_orders(df, columns),
    }

