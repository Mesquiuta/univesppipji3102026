"""Fixtures compartilhadas de teste."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def sample_transactions() -> pd.DataFrame:
    """Cria dataset sintético simples para testes."""
    return pd.DataFrame(
        {
            "customer_id": ["C1", "C1", "C2", "C2", "C3", "C3", "C4", "C5"],
            "order_id": ["O1", "O2", "O3", "O3", "O4", "O5", "O6", "O7"],
            "product_id": ["P1", "P2", "P2", "P3", "P1", "P4", "P3", "P5"],
            "order_date": [
                "2024-01-01",
                "2024-01-05",
                "2024-01-03",
                "2024-01-03",
                "2024-01-07",
                "2024-01-08",
                "2024-01-02",
                "2024-01-09",
            ],
            "quantity": [1, 2, 1, 1, 3, 1, 2, 1],
            "unit_price": [10.0, 15.0, 15.0, 20.0, 10.0, 30.0, 20.0, 50.0],
            "total_value": [10.0, 30.0, 15.0, 20.0, 30.0, 30.0, 40.0, 50.0],
        }
    )

