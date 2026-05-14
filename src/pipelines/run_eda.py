"""EDA and business KPI pipeline."""

from __future__ import annotations

from typing import Any

from src.analytics.eda import (
    average_ticket_by_month,
    customer_revenue_concentration,
    detect_simple_outliers,
    numeric_describe,
    orders_per_customer,
    revenue_by_month,
    rfm_distribution,
    summarize_dataset,
    top_customers_by_value,
    top_products_by_sales,
)
from src.analytics.funnel_analysis import (
    build_repeat_purchase_funnel,
    calculate_repurchase_metrics,
    purchase_frequency_distribution,
)
from src.analytics.kpi_analysis import calculate_kpis
from src.config.settings import AppSettings, load_settings
from src.reporting.export_csv import export_dataframe_csv
from src.reporting.export_json import export_json
from src.utils.io_helpers import load_dataframe
from src.utils.logger import configure_logging, get_logger
from src.visualization.plots import plot_revenue_over_time

logger = get_logger(__name__)


def run(settings: AppSettings | None = None) -> dict[str, Any]:
    """Run EDA and save core analytical outputs."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    processed_path = settings.paths.data_processed / settings.data.processed_filename
    logger.info("Starting EDA from %s", processed_path)
    df = load_dataframe(processed_path, parse_dates=[settings.columns.order_date])

    summary = summarize_dataset(df, settings.columns)
    describe_df = numeric_describe(df, settings.columns)
    monthly_revenue_df = revenue_by_month(df, settings.columns)
    monthly_ticket_df = average_ticket_by_month(df, settings.columns)
    orders_customer_df = orders_per_customer(df, settings.columns)
    top_products_df = top_products_by_sales(df, settings.columns, top_n=10)
    top_customers_df = top_customers_by_value(df, settings.columns, top_n=10)
    outliers_df = detect_simple_outliers(df, settings.columns)
    rfm_dist_df = rfm_distribution(df, settings.columns)

    repeat_funnel_df = build_repeat_purchase_funnel(df, settings.columns)
    frequency_dist_df = purchase_frequency_distribution(df, settings.columns)
    repurchase_metrics = calculate_repurchase_metrics(df, settings.columns)
    concentration_metrics = customer_revenue_concentration(df, settings.columns)
    kpis = calculate_kpis(df, settings.columns)

    paths = {
        "describe_path": export_dataframe_csv(describe_df, settings.paths.outputs_tables / "eda_describe.csv", index=False),
        "monthly_revenue_path": export_dataframe_csv(monthly_revenue_df, settings.paths.outputs_tables / "revenue_by_month.csv", index=False),
        "monthly_ticket_path": export_dataframe_csv(monthly_ticket_df, settings.paths.outputs_tables / "avg_ticket_by_month.csv", index=False),
        "orders_per_customer_path": export_dataframe_csv(orders_customer_df, settings.paths.outputs_tables / "orders_per_customer.csv", index=False),
        "top_products_path": export_dataframe_csv(top_products_df, settings.paths.outputs_tables / "top_products.csv", index=False),
        "top_customers_path": export_dataframe_csv(top_customers_df, settings.paths.outputs_tables / "top_customers.csv", index=False),
        "rfm_distribution_path": export_dataframe_csv(rfm_dist_df, settings.paths.outputs_tables / "rfm_distribution.csv", index=False),
        "outliers_path": export_dataframe_csv(outliers_df, settings.paths.outputs_tables / "outlier_summary.csv", index=False),
        "repeat_funnel_path": export_dataframe_csv(repeat_funnel_df, settings.paths.outputs_tables / "repeat_purchase_funnel.csv", index=False),
        "frequency_distribution_path": export_dataframe_csv(frequency_dist_df, settings.paths.outputs_tables / "purchase_frequency_distribution.csv", index=False),
    }

    summary_path = export_json(summary, settings.paths.outputs_reports / "eda_summary.json")
    kpis_path = export_json(kpis, settings.paths.outputs_reports / "kpis.json")
    retention_path = export_json(repurchase_metrics, settings.paths.outputs_reports / "retention_proxy_metrics.json")
    concentration_path = export_json(concentration_metrics, settings.paths.outputs_reports / "revenue_concentration.json")
    figure_path = settings.paths.outputs_figures / "revenue_over_time.png"
    plot_revenue_over_time(df, settings.columns, output_path=figure_path)

    return {
        **{key: str(value) for key, value in paths.items()},
        "summary_path": str(summary_path),
        "kpis_path": str(kpis_path),
        "retention_metrics_path": str(retention_path),
        "concentration_path": str(concentration_path),
        "figure_path": str(figure_path),
    }


if __name__ == "__main__":
    run()

