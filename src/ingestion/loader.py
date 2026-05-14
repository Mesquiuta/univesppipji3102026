"""Leitura de datasets transacionais."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_dataset(path: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Carrega dataset em CSV/parquet a partir de caminho local."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = file_path.suffix.lower()
    logger.info("Carregando dataset: %s", file_path)
    if suffix == ".csv":
        return pd.read_csv(file_path, encoding=encoding)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)

    raise ValueError(f"Formato de arquivo não suportado: {file_path.suffix}")

