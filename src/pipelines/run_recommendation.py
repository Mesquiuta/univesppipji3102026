"""Top-N recommendation generation pipeline."""

from __future__ import annotations

from typing import Any

from src.config.settings import AppSettings, load_settings
from src.recommendation.recommend import generate_recommendations, recommendations_to_frame
from src.reporting.export_csv import export_dataframe_csv
from src.reporting.export_json import export_json
from src.utils.io_helpers import load_dataframe
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def run(settings: AppSettings | None = None) -> dict[str, Any]:
    """Run hybrid recommendation for all users in processed data."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    processed_path = settings.paths.data_processed / settings.data.processed_filename
    logger.info("Starting recommendation from %s", processed_path)
    df = load_dataframe(processed_path, parse_dates=[settings.columns.order_date])

    recommendations, similarity, popularity = generate_recommendations(
        df,
        settings.columns,
        top_n=settings.model.recommendation_top_n,
        alpha=settings.model.hybrid_alpha,
        matrix_mode=settings.model.recommendation_matrix_mode,
        fallback_recent_weight=settings.model.fallback_recent_weight,
    )

    recommendation_frame = recommendations_to_frame(recommendations)
    recs_path = export_dataframe_csv(
        recommendation_frame,
        settings.paths.outputs_predictions / "recommendations.csv",
        index=False,
    )

    recommendation_items = {
        user_id: [item for item, _score in items] for user_id, items in recommendations.items()
    }
    recs_json_path = export_json(recommendation_items, settings.paths.outputs_predictions / "recommendations.json")

    similarity_path = settings.paths.outputs_tables / "item_similarity.csv"
    popularity_path = settings.paths.outputs_tables / "product_popularity.csv"
    similarity.to_csv(similarity_path, index=True)
    popularity.rename("score").reset_index().to_csv(popularity_path, index=False)

    return {
        "recommendations_csv_path": str(recs_path),
        "recommendations_json_path": str(recs_json_path),
        "item_similarity_path": str(similarity_path),
        "popularity_path": str(popularity_path),
        "n_users_scored": int(len(recommendations)),
        "matrix_mode": settings.model.recommendation_matrix_mode,
    }


if __name__ == "__main__":
    run()

