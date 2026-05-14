"""Métricas para avaliação de segmentação por clusters."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate_clustering(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Calcula métricas clássicas de clusterização."""
    unique_labels = np.unique(labels)
    if unique_labels.size < 2:
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "note": "Métricas indisponíveis com menos de 2 clusters.",
        }

    return {
        "silhouette_score": float(silhouette_score(features, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(features, labels)),
    }

