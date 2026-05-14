"""Pipeline mestre para execução fim-a-fim."""

from __future__ import annotations

from typing import Any

from src.config.settings import AppSettings, load_settings
from src.pipelines import run_eda, run_evaluation, run_preprocessing, run_recommendation, run_segmentation
from src.reporting.export_json import export_json
from src.reporting.report_builder import build_run_report
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def run(settings: AppSettings | None = None) -> dict[str, Any]:
    """Executa todos os pipelines em sequência."""
    settings = settings or load_settings()
    configure_logging(settings.log_level)

    logger.info("Executando pipeline completo.")
    preprocess_result = run_preprocessing.run(settings)
    eda_result = run_eda.run(settings)
    segmentation_result = run_segmentation.run(settings)
    recommendation_result = run_recommendation.run(settings)
    evaluation_result = run_evaluation.run(settings)

    artifacts = {
        "preprocessing": preprocess_result,
        "eda": eda_result,
        "segmentation": segmentation_result,
        "recommendation": recommendation_result,
        "evaluation": evaluation_result,
    }
    report = build_run_report("run_all", metrics=evaluation_result["metrics"], artifacts=artifacts)
    report_path = export_json(report, settings.paths.outputs_reports / "run_all_report.json")

    return {"report_path": str(report_path), "artifacts": artifacts}


if __name__ == "__main__":
    run()
