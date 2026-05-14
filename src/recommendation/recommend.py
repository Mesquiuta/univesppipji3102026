"""Recommendation orchestration."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from src.config.settings import DataColumns
from src.recommendation.collaborative_filtering import compute_item_similarity, recommend_for_user
from src.recommendation.content_based import (
    build_popularity_ranking,
    build_recency_ranking,
    recommend_popular,
    recommend_recent,
)
from src.recommendation.hybrid import combine_hybrid_recommendations
from src.recommendation.user_item_matrix import build_user_item_matrix, get_seen_items


def _build_fallback_recommendations(
    seen_items: set[str],
    popularity: pd.Series,
    recency: pd.Series,
    top_n: int,
    recent_weight: float,
) -> list[tuple[str, float]]:
    """Combine popularity and recency fallback rankings."""
    popularity_items = recommend_popular(popularity, seen_items=seen_items, top_n=top_n * 2)
    recent_items = recommend_recent(recency, seen_items=seen_items, top_n=top_n * 2)
    alpha = max(0.0, min(1.0, 1.0 - recent_weight))
    return combine_hybrid_recommendations(
        collaborative=popularity_items,
        fallback=recent_items,
        alpha=alpha,
        top_n=top_n,
    )


def generate_recommendations(
    df: pd.DataFrame,
    columns: DataColumns,
    top_n: int = 5,
    alpha: float = 0.7,
    matrix_mode: str = "binary",
    fallback_recent_weight: float = 0.35,
    user_ids: Iterable[str] | None = None,
) -> tuple[dict[str, list[tuple[str, float]]], pd.DataFrame, pd.Series]:
    """Generate hybrid recommendations for selected users."""
    matrix = build_user_item_matrix(df, columns, value_mode=matrix_mode)
    similarity = compute_item_similarity(matrix)
    popularity = build_popularity_ranking(df, columns)
    recency = build_recency_ranking(df, columns)

    if user_ids is None:
        target_user_ids = list(matrix.index.astype(str))
    else:
        target_user_ids = [str(user_id) for user_id in user_ids]

    recommendations: dict[str, list[tuple[str, float]]] = {}
    for user_id in target_user_ids:
        seen_items = get_seen_items(matrix, user_id)
        collaborative = recommend_for_user(user_id, matrix, similarity, top_n=top_n)
        fallback = _build_fallback_recommendations(
            seen_items=seen_items,
            popularity=popularity,
            recency=recency,
            top_n=top_n,
            recent_weight=fallback_recent_weight,
        )
        hybrid = combine_hybrid_recommendations(collaborative, fallback, alpha=alpha, top_n=top_n)
        hybrid = [(item, score) for item, score in hybrid if item not in seen_items][:top_n]
        recommendations[user_id] = hybrid
    return recommendations, similarity, popularity


def recommendations_to_frame(recommendations: dict[str, list[tuple[str, float]]]) -> pd.DataFrame:
    """Convert recommendations dictionary to tabular format."""
    records: list[dict[str, object]] = []
    for user_id, items in recommendations.items():
        for rank, (item_id, score) in enumerate(items, start=1):
            records.append(
                {
                    "customer_id": user_id,
                    "product_id": item_id,
                    "score": float(score),
                    "rank": rank,
                }
            )
    return pd.DataFrame.from_records(records)

