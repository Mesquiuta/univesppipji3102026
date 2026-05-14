"""Testes de cálculo RFM."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns
from src.segmentation.rfm import calculate_rfm


def test_calculate_rfm(sample_transactions: pd.DataFrame) -> None:
    """Valida estrutura e consistência básica da tabela RFM."""
    df = sample_transactions.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    rfm = calculate_rfm(df, DataColumns())
    assert {"customer_id", "recency", "frequency", "monetary"} <= set(rfm.columns)
    assert (rfm["recency"] >= 0).all()
    assert (rfm["frequency"] >= 1).all()

