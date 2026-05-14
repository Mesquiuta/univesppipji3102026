"""Unit tests for recommendation behavior."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns
from src.recommendation.recommend import generate_recommendations
from src.recommendation.user_item_matrix import build_user_item_matrix, get_seen_items


def test_generate_recommendations(sample_transactions: pd.DataFrame) -> None:
    """Recommendations should avoid seen items and preserve ranking order."""
    df = sample_transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    columns = DataColumns()

    recommendations, _similarity, _popularity = generate_recommendations(
        df,
        columns,
        top_n=3,
        alpha=0.7,
        matrix_mode="binary",
    )
    matrix = build_user_item_matrix(df, columns, value_mode="binary")

    assert recommendations
    for user_id, items in recommendations.items():
        assert len(items) <= 3
        seen = get_seen_items(matrix, user_id)
        recommended_products = {item for item, _score in items}
        assert not (recommended_products & seen)

        scores = [score for _, score in items]
        assert scores == sorted(scores, reverse=True)
        assert all(score >= 0 for score in scores)

