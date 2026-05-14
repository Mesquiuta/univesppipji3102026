"""Simple fallback recommendation strategies."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns


def build_popularity_ranking(df: pd.DataFrame, columns: DataColumns) -> pd.Series:
    """Build product popularity ranking using buyers, orders, and revenue."""
    popularity = (
        df.groupby(columns.product_id)
        .agg(
            buyers=(columns.customer_id, "nunique"),
            orders=(columns.order_id, "nunique"),
            revenue=(columns.total_value, "sum"),
        )
        .assign(score=lambda x: x["buyers"] * 0.4 + x["orders"] * 0.3 + x["revenue"] * 0.3)
        .sort_values("score", ascending=False)["score"]
    )
    return popularity


def build_recency_ranking(df: pd.DataFrame, columns: DataColumns) -> pd.Series:
    """Build ranking of recently purchased products."""
    latest = df.groupby(columns.product_id)[columns.order_date].max().sort_values(ascending=False)
    ranked = pd.Series(
        data=list(range(len(latest), 0, -1)),
        index=latest.index,
        dtype=float,
    )
    return ranked


def recommend_popular(
    popularity_ranking: pd.Series,
    seen_items: set[str],
    top_n: int,
) -> list[tuple[str, float]]:
    """Return top-N popular items not yet seen by the user."""
    filtered = [
        (str(item), float(score))
        for item, score in popularity_ranking.items()
        if str(item) not in seen_items
    ]
    return filtered[:top_n]


def recommend_recent(
    recency_ranking: pd.Series,
    seen_items: set[str],
    top_n: int,
) -> list[tuple[str, float]]:
    """Return top-N recent items not yet seen by the user."""
    filtered = [
        (str(item), float(score))
        for item, score in recency_ranking.items()
        if str(item) not in seen_items
    ]
    return filtered[:top_n]

