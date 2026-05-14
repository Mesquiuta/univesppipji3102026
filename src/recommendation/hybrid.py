"""Hybrid recommendation score fusion."""

from __future__ import annotations


def _normalize_ranked_scores(items: list[tuple[str, float]]) -> dict[str, float]:
    """Normalize scores to [0, 1] while keeping relative ranking."""
    if not items:
        return {}
    max_score = max(max(score, 0.0) for _, score in items) or 1.0
    return {item: max(score, 0.0) / max_score for item, score in items}


def combine_hybrid_recommendations(
    collaborative: list[tuple[str, float]],
    fallback: list[tuple[str, float]],
    alpha: float,
    top_n: int,
) -> list[tuple[str, float]]:
    """Merge collaborative and fallback scores into a single ranking."""
    collab_norm = _normalize_ranked_scores(collaborative)
    fallback_norm = _normalize_ranked_scores(fallback)

    all_items = set(collab_norm) | set(fallback_norm)
    merged: list[tuple[str, float]] = []
    for item in all_items:
        score = alpha * collab_norm.get(item, 0.0) + (1 - alpha) * fallback_norm.get(item, 0.0)
        merged.append((item, float(score)))

    merged.sort(key=lambda x: (-x[1], x[0]))
    return merged[:top_n]

