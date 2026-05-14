"""Technical and business evaluation pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config.settings import AppSettings, load_settings
from src.evaluation.business_metrics import (
    average_recommendation_list_size,
    catalog_coverage,
    personalization,
)
from src.evaluation.clustering_metrics import evaluate_clustering
from src.evaluation.recommender_metrics import evaluate_recommendations
from src.recommendation.recommend import generate_recommendations
from src.reporting.export_json import export_json
from src.reporting.report_builder import build_run_report
from src.utils.io_helpers import load_dataframe
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def temporal_train_test_split(
    df: pd.DataFrame,
    customer_col: str,
    order_col: str,
    date_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split transactions by time:
    - test: most recent order per user with at least 2 orders
    - train: all older orders + users with only one order
    """
    order_dates = (
        df.groupby([customer_col, order_col], as_index=False)[date_col]
        .max()
        .sort_values([customer_col, date_col, order_col])
    )
    order_counts = order_dates.groupby(customer_col)[order_col].nunique()
    eligible_customers = set(order_counts[order_counts >= 2].index.astype(str))

    test_orders = (
        order_dates[order_dates[customer_col].astype(str).isin(eligible_customers)]
        .groupby(customer_col, as_index=False)
        .tail(1)
    )
    test_order_ids = set(test_orders[order_col].astype(str))
    test_customer_ids = set(test_orders[customer_col].astype(str))

    test_mask = df[order_col].astype(str).isin(test_order_ids) & df[customer_col].astype(str).isin(test_customer_ids)
    test_df = df[test_mask].copy()
    train_df = df[~test_mask].copy()
    return train_df, test_df


def build_ground_truth(
    test_df: pd.DataFrame,
    customer_col: str,
    item_col: str,
) -> dict[str, set[str]]:
    """Build relevant items by customer from holdout test orders."""
    if test_df.empty:
        return {}
    return (
        test_df.groupby(customer_col)[item_col]
        .apply(lambda x: set(x.astype(str)))
        .to_dict()
    )


def run(settings: AppSettings | None = None) -> dict[str, Any]:
    """Run clustering and recommender evaluation metrics."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    processed_path = settings.paths.data_processed / settings.data.processed_filename
    clustered_path = settings.paths.outputs_tables / "rfm_clusters.csv"

    df = load_dataframe(processed_path, parse_dates=[settings.columns.order_date])
    clustered_df = load_dataframe(clustered_path)

    clustering_scores = evaluate_clustering(
        clustered_df[["recency", "frequency", "monetary"]].to_numpy(),
        clustered_df["cluster"].to_numpy(),
    )

    train_df, test_df = temporal_train_test_split(
        df,
        customer_col=settings.columns.customer_id,
        order_col=settings.columns.order_id,
        date_col=settings.columns.order_date,
    )
    ground_truth = build_ground_truth(
        test_df,
        customer_col=settings.columns.customer_id,
        item_col=settings.columns.product_id,
    )

    target_users = list(ground_truth.keys())
    recommendations, _similarity, _popularity = generate_recommendations(
        train_df,
        settings.columns,
        top_n=settings.model.recommendation_top_n,
        alpha=settings.model.hybrid_alpha,
        matrix_mode=settings.model.recommendation_matrix_mode,
        fallback_recent_weight=settings.model.fallback_recent_weight,
        user_ids=target_users,
    )
    predicted_items = {
        user_id: [item for item, _score in items] for user_id, items in recommendations.items()
    }

    rec_metrics = evaluate_recommendations(
        predictions=predicted_items,
        ground_truth=ground_truth,
        k=settings.model.recommendation_top_n,
    )

    business = {
        "catalog_coverage": catalog_coverage(
            predicted_items,
            catalog_size=int(train_df[settings.columns.product_id].nunique()),
        ),
        "personalization": personalization(predicted_items),
        "avg_recommendation_list_size": average_recommendation_list_size(predicted_items),
    }

    split_info = {
        "train_rows": int(train_df.shape[0]),
        "test_rows": int(test_df.shape[0]),
        "train_orders": int(train_df[settings.columns.order_id].nunique()),
        "test_orders": int(test_df[settings.columns.order_id].nunique()),
        "evaluated_users": int(len(target_users)),
    }

    full_metrics = {
        "clustering": clustering_scores,
        "recommender": rec_metrics,
        "business": business,
        "split": split_info,
    }
    report = build_run_report("evaluation", full_metrics)
    report_path = export_json(report, settings.paths.outputs_reports / "evaluation_report.json")
    holdout_path = export_json(
        {
            "ground_truth_users": target_users,
            "predicted_items": predicted_items,
        },
        settings.paths.outputs_reports / "holdout_recommendation_eval.json",
    )
    return {"report_path": str(report_path), "holdout_path": str(holdout_path), "metrics": full_metrics}


if __name__ == "__main__":
    run()

