"""Unit tests for clustering utilities."""

from __future__ import annotations

import pandas as pd

from src.segmentation.clustering import assign_clusters, choose_optimal_k, evaluate_k_range, train_kmeans


def test_kmeans_training_and_assignment() -> None:
    """Train model and assign cluster labels on synthetic RFM data."""
    rfm_df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4"],
            "recency": [1, 5, 20, 30],
            "frequency": [10, 7, 3, 1],
            "monetary": [1000, 700, 200, 50],
        }
    )
    artifacts = train_kmeans(rfm_df, n_clusters=2, random_state=42)
    clustered = assign_clusters(rfm_df, artifacts.labels)

    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() == 2


def test_k_range_diagnostics_and_selection() -> None:
    """Evaluate candidate Ks and pick an optimal one."""
    rfm_df = pd.DataFrame(
        {
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "recency": [1, 3, 15, 20, 35],
            "frequency": [10, 8, 4, 2, 1],
            "monetary": [1100, 900, 350, 200, 80],
        }
    )
    diagnostics = evaluate_k_range(rfm_df, k_values=range(2, 5), random_state=42)
    selected_k = choose_optimal_k(diagnostics, fallback_k=3)

    assert set(diagnostics.columns) == {"k", "inertia", "silhouette"}
    assert selected_k in diagnostics["k"].tolist()

