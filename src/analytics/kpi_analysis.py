"""Indicadores de negócio para e-commerce."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import DataColumns


def calculate_kpis(df: pd.DataFrame, columns: DataColumns) -> dict[str, Any]:
    """Calcula KPIs iniciais de operação e receita."""
    orders = df.groupby(columns.order_id).agg(
        order_value=(columns.total_value, "sum"),
        item_count=(columns.product_id, "count"),
        customer_id=(columns.customer_id, "first"),
    )
    customer_orders = orders.groupby("customer_id").size()

    gross_revenue = float(orders["order_value"].sum())
    n_orders = int(orders.shape[0])
    n_customers = int(df[columns.customer_id].nunique())
    avg_order_value = gross_revenue / n_orders if n_orders else 0.0
    avg_items_per_order = float(orders["item_count"].mean()) if n_orders else 0.0
    repeat_customer_rate = float((customer_orders >= 2).mean()) if not customer_orders.empty else 0.0

    return {
        "gross_revenue": gross_revenue,
        "n_orders": n_orders,
        "n_customers": n_customers,
        "avg_order_value": avg_order_value,
        "avg_items_per_order": avg_items_per_order,
        "repeat_customer_rate": repeat_customer_rate,
    }

