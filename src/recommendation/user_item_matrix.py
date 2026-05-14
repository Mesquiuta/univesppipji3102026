"""Build user-item interaction matrices."""

from __future__ import annotations

import pandas as pd

from src.config.settings import DataColumns


VALID_MATRIX_MODES = {"binary", "quantity", "total_value"}


def build_user_item_matrix(
    df: pd.DataFrame,
    columns: DataColumns,
    value_mode: str = "binary",
    implicit: bool | None = None,
) -> pd.DataFrame:
    """
    Build a user-item matrix using one of:
    - binary: interaction presence (0/1)
    - quantity: purchased quantity sum
    - total_value: purchased monetary sum
    """
    if implicit is not None:
        value_mode = "binary" if implicit else "total_value"
    if value_mode not in VALID_MATRIX_MODES:
        raise ValueError(f"Invalid value_mode '{value_mode}'. Expected one of {sorted(VALID_MATRIX_MODES)}")

    if value_mode == "quantity":
        value_column = columns.quantity
    else:
        value_column = columns.total_value

    matrix = pd.pivot_table(
        df,
        index=columns.customer_id,
        columns=columns.product_id,
        values=value_column,
        aggfunc="sum",
        fill_value=0.0,
    )
    if value_mode == "binary":
        matrix = (matrix > 0).astype(float)
    return matrix


def get_seen_items(matrix: pd.DataFrame, user_id: str) -> set[str]:
    """Return items already consumed by a user."""
    if user_id not in matrix.index:
        return set()
    consumed = matrix.loc[user_id]
    return set(consumed[consumed > 0].index.astype(str))

