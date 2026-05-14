"""Testes de validação de colunas."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.validation import validate_required_columns


def test_validate_required_columns_success() -> None:
    """Não deve falhar quando colunas obrigatórias existem."""
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    validate_required_columns(df, ["a", "b"])


def test_validate_required_columns_missing() -> None:
    """Deve falhar quando faltar coluna obrigatória."""
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError):
        validate_required_columns(df, ["a", "b", "c"])

