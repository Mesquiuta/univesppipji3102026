"""Funções de plot para apoio à análise."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config.settings import DataColumns  # noqa: E402


def plot_revenue_over_time(df: pd.DataFrame, columns: DataColumns, output_path: Path | None = None) -> None:
    """Plota receita diária e salva figura opcionalmente."""
    daily = df.groupby(pd.Grouper(key=columns.order_date, freq="D"))[columns.total_value].sum().reset_index()
    plt.figure(figsize=(10, 4))
    sns.lineplot(data=daily, x=columns.order_date, y=columns.total_value, marker="o")
    plt.title("Receita ao Longo do Tempo")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120)
    plt.close()


def plot_cluster_distribution(clustered_rfm: pd.DataFrame, output_path: Path | None = None) -> None:
    """Plota distribuição de clientes por cluster."""
    plt.figure(figsize=(8, 4))
    sns.countplot(data=clustered_rfm, x="cluster")
    plt.title("Distribuição de Clientes por Cluster")
    plt.tight_layout()
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=120)
    plt.close()

