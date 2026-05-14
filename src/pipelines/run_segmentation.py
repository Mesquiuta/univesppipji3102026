"""RFM + K-Means customer segmentation pipeline."""

from __future__ import annotations

from typing import Any

from src.config.settings import AppSettings, load_settings
from src.reporting.export_csv import export_dataframe_csv
from src.reporting.export_json import export_json
from src.segmentation.cluster_labeling import attach_cluster_labels, summarize_clusters
from src.segmentation.clustering import (
    assign_clusters,
    choose_optimal_k,
    evaluate_k_range,
    train_kmeans,
)
from src.segmentation.rfm import calculate_rfm
from src.utils.io_helpers import load_dataframe, save_model
from src.utils.logger import configure_logging, get_logger
from src.visualization.plots import plot_cluster_distribution

logger = get_logger(__name__)


def run(settings: AppSettings | None = None) -> dict[str, Any]:
    """Run segmentation and save artifacts."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    processed_path = settings.paths.data_processed / settings.data.processed_filename
    logger.info("Starting segmentation from %s", processed_path)
    df = load_dataframe(processed_path, parse_dates=[settings.columns.order_date])

    rfm_df = calculate_rfm(df, settings.columns)
    k_values = range(settings.model.cluster_k_min, settings.model.cluster_k_max + 1)
    diagnostics = evaluate_k_range(
        rfm_df,
        k_values=k_values,
        random_state=settings.model.random_state,
    )
    diagnostics_path = export_dataframe_csv(
        diagnostics,
        settings.paths.outputs_tables / "k_selection_diagnostics.csv",
        index=False,
    )

    if settings.model.auto_select_k:
        selected_k = choose_optimal_k(diagnostics, fallback_k=settings.model.n_clusters)
    else:
        selected_k = settings.model.n_clusters

    artifacts = train_kmeans(rfm_df, n_clusters=selected_k, random_state=settings.model.random_state)
    clustered = assign_clusters(rfm_df, artifacts.labels)
    labeled = attach_cluster_labels(clustered)
    cluster_summary = summarize_clusters(labeled)

    clustered_path = export_dataframe_csv(labeled, settings.paths.outputs_tables / "rfm_clusters.csv", index=False)
    summary_path = export_dataframe_csv(cluster_summary, settings.paths.outputs_tables / "cluster_summary.csv", index=False)

    model_path = settings.paths.models / "kmeans_rfm.joblib"
    scaler_path = settings.paths.models / "rfm_scaler.joblib"
    save_model(artifacts.model, model_path)
    save_model(artifacts.scaler, scaler_path)

    selection_metadata_path = export_json(
        {
            "selected_k": selected_k,
            "auto_select_k": settings.model.auto_select_k,
            "fallback_k": settings.model.n_clusters,
        },
        settings.paths.outputs_reports / "k_selection_summary.json",
    )

    figure_path = settings.paths.outputs_figures / "cluster_distribution.png"
    plot_cluster_distribution(labeled, output_path=figure_path)

    return {
        "clustered_path": str(clustered_path),
        "summary_path": str(summary_path),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "figure_path": str(figure_path),
        "diagnostics_path": str(diagnostics_path),
        "selection_metadata_path": str(selection_metadata_path),
        "selected_k": int(selected_k),
        "inertia": float(artifacts.model.inertia_),
    }


if __name__ == "__main__":
    run()

