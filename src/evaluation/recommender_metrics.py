"""Métricas de ranking para sistemas de recomendação."""

from __future__ import annotations

from typing import Iterable


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Calcula precisão@k."""
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(top_k)


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Calcula recall@k."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Calcula AP@k para uma lista recomendada."""
    if not relevant:
        return 0.0
    top_k = recommended[:k]
    score = 0.0
    hits = 0
    for idx, item in enumerate(top_k, start=1):
        if item in relevant:
            hits += 1
            score += hits / idx
    return score / min(len(relevant), k) if k > 0 else 0.0


def evaluate_recommendations(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
) -> dict[str, float]:
    """Agrega métricas de recomendação em nível global."""
    users = [user for user in predictions if user in ground_truth]
    if not users:
        return {"precision_at_k": 0.0, "recall_at_k": 0.0, "map_at_k": 0.0}

    precision_scores = []
    recall_scores = []
    map_scores = []
    for user in users:
        recommended = predictions[user]
        relevant = ground_truth[user]
        precision_scores.append(precision_at_k(recommended, relevant, k))
        recall_scores.append(recall_at_k(recommended, relevant, k))
        map_scores.append(average_precision_at_k(recommended, relevant, k))

    return {
        "precision_at_k": float(sum(precision_scores) / len(precision_scores)),
        "recall_at_k": float(sum(recall_scores) / len(recall_scores)),
        "map_at_k": float(sum(map_scores) / len(map_scores)),
    }

