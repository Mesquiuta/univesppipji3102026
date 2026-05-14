"""Item-item collaborative filtering helpers."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_item_similarity(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute cosine similarity between items."""
    if user_item_matrix.empty:
        return pd.DataFrame()
    similarity = cosine_similarity(user_item_matrix.T)
    return pd.DataFrame(similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)


def recommend_for_user(
    user_id: str,
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    top_n: int,
) -> list[tuple[str, float]]:
    """Generate top-N collaborative recommendations for a user."""
    if user_id not in user_item_matrix.index or item_similarity.empty:
        return []

    user_vector = user_item_matrix.loc[user_id]
    raw_scores = item_similarity.dot(user_vector).astype(float)
    seen_mask = user_vector > 0
    candidates = raw_scores[~seen_mask]
    candidates = candidates[candidates > 0]
    if candidates.empty:
        return []
    top_items = candidates.sort_values(ascending=False).head(top_n)
    return [(str(item), float(score)) for item, score in top_items.items()]


def recommend_for_users(
    user_ids: Sequence[str],
    user_item_matrix: pd.DataFrame,
    item_similarity: pd.DataFrame,
    top_n: int,
) -> dict[str, list[tuple[str, float]]]:
    """Generate collaborative recommendations for multiple users."""
    return {
        str(user_id): recommend_for_user(str(user_id), user_item_matrix, item_similarity, top_n)
        for user_id in user_ids
    }

