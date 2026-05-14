"""Indicadores de negócio para recomendações."""

from __future__ import annotations

from itertools import combinations


def catalog_coverage(predictions: dict[str, list[str]], catalog_size: int) -> float:
    """Percentual do catálogo recomendado para pelo menos um usuário."""
    if catalog_size <= 0:
        return 0.0
    recommended_items = {item for recs in predictions.values() for item in recs}
    return len(recommended_items) / catalog_size


def personalization(predictions: dict[str, list[str]]) -> float:
    """Mede diversidade entre listas de usuários via dissimilaridade média."""
    user_lists = list(predictions.values())
    if len(user_lists) < 2:
        return 0.0

    dissimilarities = []
    for a, b in combinations(user_lists, 2):
        sa, sb = set(a), set(b)
        union = sa | sb
        if not union:
            dissimilarities.append(0.0)
            continue
        jaccard = len(sa & sb) / len(union)
        dissimilarities.append(1 - jaccard)
    return float(sum(dissimilarities) / len(dissimilarities))


def average_recommendation_list_size(predictions: dict[str, list[str]]) -> float:
    """Tamanho médio das listas recomendadas."""
    if not predictions:
        return 0.0
    return float(sum(len(items) for items in predictions.values()) / len(predictions))

