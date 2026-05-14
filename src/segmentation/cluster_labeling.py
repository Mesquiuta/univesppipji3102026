"""Interpretive labeling rules for customer clusters."""

from __future__ import annotations

import pandas as pd


def summarize_clusters(clustered_rfm: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean profile per cluster."""
    return (
        clustered_rfm.groupby("cluster")[["recency", "frequency", "monetary"]]
        .mean()
        .reset_index()
        .sort_values("cluster")
    )


def _flag_relative_position(cluster_summary: pd.DataFrame) -> pd.DataFrame:
    """Create relative flags for high/low recency, frequency, and monetary."""
    summary = cluster_summary.copy()
    summary["low_recency"] = summary["recency"] <= summary["recency"].quantile(0.33)
    summary["high_recency"] = summary["recency"] >= summary["recency"].quantile(0.67)
    summary["high_frequency"] = summary["frequency"] >= summary["frequency"].quantile(0.67)
    summary["low_frequency"] = summary["frequency"] <= summary["frequency"].quantile(0.33)
    summary["high_monetary"] = summary["monetary"] >= summary["monetary"].quantile(0.67)
    summary["low_monetary"] = summary["monetary"] <= summary["monetary"].quantile(0.33)
    return summary


def create_cluster_labels(cluster_summary: pd.DataFrame) -> dict[int, str]:
    """
    Label clusters with explicit business rules:
    - low recency + high frequency + high monetary => loyal_high_value
    - high recency + low frequency => at_risk
    - low frequency + low monetary => occasional_low_value
    - high monetary => big_spenders
    - high frequency => frequent_buyers
    - fallback => regular
    """
    labeled_summary = _flag_relative_position(cluster_summary)
    labels: dict[int, str] = {}

    for _, row in labeled_summary.iterrows():
        cluster_id = int(row["cluster"])
        if row["low_recency"] and row["high_frequency"] and row["high_monetary"]:
            label = "loyal_high_value"
        elif row["high_recency"] and row["low_frequency"]:
            label = "at_risk"
        elif row["low_frequency"] and row["low_monetary"]:
            label = "occasional_low_value"
        elif row["high_monetary"]:
            label = "big_spenders"
        elif row["high_frequency"]:
            label = "frequent_buyers"
        else:
            label = "regular"
        labels[cluster_id] = label
    return labels


def attach_cluster_labels(clustered_rfm: pd.DataFrame) -> pd.DataFrame:
    """Attach human-readable labels to clustered RFM data."""
    cluster_summary = summarize_clusters(clustered_rfm)
    mapping = create_cluster_labels(cluster_summary)
    labeled = clustered_rfm.copy()
    labeled["cluster_label"] = labeled["cluster"].map(mapping)
    return labeled

