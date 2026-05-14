"""Testes de ingestão de dados."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.loader import load_dataset


def test_load_dataset_csv(tmp_path: Path) -> None:
    """Valida leitura de CSV local."""
    file_path = tmp_path / "transactions.csv"
    file_path.write_text("customer_id,order_id\nC1,O1\n", encoding="utf-8")

    df = load_dataset(file_path)
    assert df.shape == (1, 2)
    assert list(df.columns) == ["customer_id", "order_id"]

