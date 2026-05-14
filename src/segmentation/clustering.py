"""K-Means training and diagnostics for RFM clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class ClusteringArtifacts:
    """Container for fitted clustering artifacts."""

    scaler: StandardScaler
    model: KMeans
    features: pd.DataFrame
    labels: pd.Series


def prepare_rfm_features(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Select numeric RFM features."""
    return rfm_df[["recency", "frequency", "monetary"]].copy()


def evaluate_k_range(
    rfm_df: pd.DataFrame,
    k_values: Iterable[int],
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate candidate K values using inertia and silhouette."""
    features = prepare_rfm_features(rfm_df)
    if features.empty:
        return pd.DataFrame(columns=["k", "inertia", "silhouette"])

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    n_samples = scaled.shape[0]
    rows: list[dict[str, float | int | None]] = []

    for k in sorted(set(int(value) for value in k_values)):
        if k < 2 or k >= n_samples:
            rows.append({"k": k, "inertia": None, "silhouette": None})
            continue
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(scaled)
        silhouette = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 else None
        rows.append({"k": k, "inertia": float(model.inertia_), "silhouette": silhouette})
    return pd.DataFrame(rows)


def choose_optimal_k(diagnostics: pd.DataFrame, fallback_k: int) -> int:
    """Pick the best K from diagnostics using silhouette as primary criterion."""
    if diagnostics.empty:
        return fallback_k
    valid_silhouette = diagnostics.dropna(subset=["silhouette"])
    if not valid_silhouette.empty:
        best_row = valid_silhouette.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]
        return int(best_row["k"])

    valid_inertia = diagnostics.dropna(subset=["inertia"])
    if not valid_inertia.empty:
        best_row = valid_inertia.sort_values(["inertia", "k"], ascending=[True, True]).iloc[0]
        return int(best_row["k"])

    return fallback_k


def train_kmeans(
    rfm_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
) -> ClusteringArtifacts:
    """Train K-Means on standardized RFM features."""
    features = prepare_rfm_features(rfm_df)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = pd.Series(model.fit_predict(scaled), name="cluster")
    return ClusteringArtifacts(scaler=scaler, model=model, features=features, labels=labels)


def assign_clusters(rfm_df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Attach cluster assignments to the RFM table."""
    clustered = rfm_df.copy()
    clustered["cluster"] = labels.values
    return clustered

