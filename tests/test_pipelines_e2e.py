"""End-to-end pipeline tests with temporary project directories."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.paths import build_paths, ensure_directories
from src.config.settings import AppSettings, DataColumns, DataConfig, ModelParams
from src.pipelines import run_all, run_preprocessing, run_recommendation, run_segmentation
from src.utils.io_helpers import load_dataframe, load_json


def _build_temp_settings(tmp_path: Path) -> AppSettings:
    """Create isolated settings rooted in a temporary directory."""
    paths = build_paths(root=tmp_path)
    ensure_directories(paths)
    return AppSettings(
        paths=paths,
        columns=DataColumns(),
        model=ModelParams(
            n_clusters=3,
            auto_select_k=True,
            cluster_k_min=2,
            cluster_k_max=4,
            recommendation_top_n=3,
            recommendation_matrix_mode="binary",
            hybrid_alpha=0.7,
            fallback_recent_weight=0.35,
        ),
        data=DataConfig(
            source_filename="transactions.csv",
            interim_filename="transactions_clean.csv",
            processed_filename="transactions_processed.csv",
            encoding="utf-8",
        ),
        log_level="WARNING",
    )


def _write_input_data(settings: AppSettings, sample_transactions: pd.DataFrame) -> Path:
    """Write synthetic transactions into temporary raw directory."""
    source_path = settings.paths.data_raw / settings.data.source_filename
    sample_transactions.to_csv(source_path, index=False)
    return source_path


def test_run_preprocessing_pipeline(tmp_path: Path, sample_transactions: pd.DataFrame) -> None:
    """Validate preprocessing pipeline output and cleaning report."""
    settings = _build_temp_settings(tmp_path)
    _write_input_data(settings, sample_transactions)

    result = run_preprocessing.run(settings)
    processed = load_dataframe(Path(result["processed_path"]))

    assert Path(result["interim_path"]).exists()
    assert Path(result["processed_path"]).exists()
    assert not processed.empty
    assert "cleaning_report" in result
    assert "total_value" in processed.columns


def test_run_segmentation_pipeline(tmp_path: Path, sample_transactions: pd.DataFrame) -> None:
    """Validate segmentation pipeline artifacts and diagnostics."""
    settings = _build_temp_settings(tmp_path)
    _write_input_data(settings, sample_transactions)
    run_preprocessing.run(settings)

    result = run_segmentation.run(settings)
    clustered = load_dataframe(Path(result["clustered_path"]))

    assert Path(result["clustered_path"]).exists()
    assert Path(result["diagnostics_path"]).exists()
    assert clustered["cluster"].nunique() >= 2
    assert result["selected_k"] >= 2


def test_run_recommendation_pipeline(tmp_path: Path, sample_transactions: pd.DataFrame) -> None:
    """Validate recommendation pipeline outputs."""
    settings = _build_temp_settings(tmp_path)
    _write_input_data(settings, sample_transactions)
    run_preprocessing.run(settings)

    result = run_recommendation.run(settings)
    recs_df = load_dataframe(Path(result["recommendations_csv_path"]))
    recs_json = load_json(Path(result["recommendations_json_path"]))

    assert Path(result["recommendations_csv_path"]).exists()
    assert isinstance(recs_json, dict)
    assert set(recs_df.columns) >= {"customer_id", "product_id", "score", "rank"}
    assert result["matrix_mode"] == "binary"


def test_run_all_pipeline(tmp_path: Path, sample_transactions: pd.DataFrame) -> None:
    """Validate full run_all execution including evaluation holdout report."""
    settings = _build_temp_settings(tmp_path)
    _write_input_data(settings, sample_transactions)

    result = run_all.run(settings)
    report_path = Path(result["report_path"])
    evaluation_path = settings.paths.outputs_reports / "evaluation_report.json"
    holdout_path = settings.paths.outputs_reports / "holdout_recommendation_eval.json"

    assert report_path.exists()
    assert evaluation_path.exists()
    assert holdout_path.exists()

