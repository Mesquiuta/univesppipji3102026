"""Exportação padronizada de DataFrames para CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_dataframe_csv(df: pd.DataFrame, output_path: Path, index: bool = False) -> Path:
    """Salva DataFrame em CSV e retorna caminho salvo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=index)
    return output_path

