"""Central configuration for the analytics project."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config.paths import ProjectPaths, build_paths, ensure_directories


@dataclass(frozen=True)
class DataColumns:
    """Column names expected by the project."""

    customer_id: str = "customer_id"
    order_id: str = "order_id"
    product_id: str = "product_id"
    order_date: str = "order_date"
    quantity: str = "quantity"
    unit_price: str = "unit_price"
    total_value: str = "total_value"

    def required(self) -> list[str]:
        """Minimum required columns for preprocessing."""
        return [
            self.customer_id,
            self.order_id,
            self.product_id,
            self.order_date,
            self.quantity,
            self.unit_price,
        ]


@dataclass(frozen=True)
class ModelParams:
    """Initial model and evaluation parameters."""

    n_clusters: int = 4
    auto_select_k: bool = True
    cluster_k_min: int = 2
    cluster_k_max: int = 8
    random_state: int = 42
    recommendation_top_n: int = 5
    hybrid_alpha: float = 0.7
    recommendation_matrix_mode: str = "binary"
    fallback_recent_weight: float = 0.35


@dataclass(frozen=True)
class DataConfig:
    """Dataset IO parameters."""

    source_filename: str = "transactions.csv"
    interim_filename: str = "transactions_clean.csv"
    processed_filename: str = "transactions_processed.csv"
    encoding: str = "utf-8"


@dataclass(frozen=True)
class AppSettings:
    """Main settings object used by pipelines."""

    paths: ProjectPaths
    columns: DataColumns
    model: ModelParams
    data: DataConfig
    log_level: str = "INFO"


def _override_dataclass(instance: Any, updates: dict[str, Any]) -> Any:
    """Override dataclass fields from a dictionary when keys match."""
    valid = {key: value for key, value in updates.items() if hasattr(instance, key)}
    return type(instance)(**{**instance.__dict__, **valid})


def load_settings(config_path: str | Path | None = None) -> AppSettings:
    """Load default settings and optional JSON overrides."""
    paths = build_paths()
    ensure_directories(paths)

    settings = AppSettings(
        paths=paths,
        columns=DataColumns(),
        model=ModelParams(),
        data=DataConfig(),
    )

    if not config_path:
        return settings

    cfg_path = Path(config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    columns = _override_dataclass(settings.columns, payload.get("columns", {}))
    model = _override_dataclass(settings.model, payload.get("model", {}))
    data = _override_dataclass(settings.data, payload.get("data", {}))
    log_level = payload.get("log_level", settings.log_level)
    return AppSettings(paths=paths, columns=columns, model=model, data=data, log_level=log_level)

