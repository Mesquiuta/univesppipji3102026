"""Helpers de I/O para dados e modelos."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_dir(path: Path) -> None:
    """Garante que um diretório exista."""
    path.mkdir(parents=True, exist_ok=True)


def load_dataframe(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Carrega DataFrame de CSV ou parquet com base na extensão."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    raise ValueError(f"Formato não suportado: {path.suffix}")


def save_dataframe(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    """Salva DataFrame em CSV ou parquet de acordo com extensão."""
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
        return
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=index)
        return
    raise ValueError(f"Formato não suportado: {path.suffix}")


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Salva dicionário em JSON com indentação."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Carrega JSON para dicionário."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def save_model(model: Any, path: Path) -> None:
    """Serializa objeto Python com joblib."""
    ensure_dir(path.parent)
    joblib.dump(model, path)


def load_model(path: Path) -> Any:
    """Desserializa objeto salvo com joblib."""
    return joblib.load(path)

