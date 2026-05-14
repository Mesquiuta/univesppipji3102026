"""Feature engineering helpers."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns


def add_total_value(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Create or reconstruct transaction total value."""
    featured = df.copy()
    featured[columns.quantity] = pd.to_numeric(featured[columns.quantity], errors="coerce")
    featured[columns.unit_price] = pd.to_numeric(featured[columns.unit_price], errors="coerce")

    computed = featured[columns.quantity] * featured[columns.unit_price]
    if columns.total_value not in featured.columns:
        featured[columns.total_value] = computed
    else:
        base_total = pd.to_numeric(featured[columns.total_value], errors="coerce")
        featured[columns.total_value] = base_total.fillna(computed)

    featured[columns.total_value] = pd.to_numeric(featured[columns.total_value], errors="coerce")
    featured[columns.total_value] = featured[columns.total_value].fillna(computed).fillna(0.0)
    return featured


def add_time_features(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Add simple temporal features."""
    featured = df.copy()
    featured[columns.order_date] = pd.to_datetime(featured[columns.order_date], errors="coerce")
    order_date = featured[columns.order_date]
    featured["order_year"] = order_date.dt.year
    featured["order_month"] = order_date.dt.month
    featured["order_dayofweek"] = order_date.dt.dayofweek
    featured["is_weekend"] = (order_date.dt.dayofweek >= 5).astype(int)
    return featured


def build_features(df: pd.DataFrame, columns: DataColumns) -> pd.DataFrame:
    """Run base feature engineering pipeline."""
    featured = add_total_value(df, columns)
    featured = add_time_features(featured, columns)
    return featured

