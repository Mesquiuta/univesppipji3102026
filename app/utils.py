"""Helpers compartilhados pelas páginas do frontend Streamlit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.paths import build_paths, ensure_directories  # noqa: E402


def get_paths():
    """Retorna os caminhos canônicos do projeto."""
    paths = build_paths(root=PROJECT_ROOT)
    ensure_directories(paths)
    return paths


@st.cache_data(show_spinner=False)
def load_csv(path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Carrega CSV com cache do Streamlit."""
    return pd.read_csv(path, parse_dates=parse_dates) if parse_dates else pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict[str, Any]:
    """Carrega JSON com cache do Streamlit."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def file_exists(path: Path | str) -> bool:
    """Verifica existência de arquivo."""
    return Path(path).exists()


def pipeline_status() -> dict[str, bool]:
    """Retorna status booleano de presença dos artefatos principais."""
    paths = get_paths()
    return {
        "raw_dataset": (paths.data_raw / "transactions.csv").exists(),
        "processed_dataset": (paths.data_processed / "transactions_processed.csv").exists(),
        "eda_summary": (paths.outputs_reports / "eda_summary.json").exists(),
        "kpis": (paths.outputs_reports / "kpis.json").exists(),
        "clusters": (paths.outputs_tables / "rfm_clusters.csv").exists(),
        "recommendations": (paths.outputs_predictions / "recommendations.csv").exists(),
        "evaluation": (paths.outputs_reports / "evaluation_report.json").exists(),
        "model": (paths.models / "kmeans_rfm.joblib").exists(),
    }


def render_pipeline_warning(missing_artifact_label: str) -> None:
    """Renderiza aviso amigável quando artefato necessário não existe."""
    st.warning(
        f"Artefato '{missing_artifact_label}' ainda não foi gerado. "
        "Execute o pipeline completo na Home ou rode `python main.py --pipeline all` no terminal."
    )


def render_run_pipeline_button(key: str = "run_pipeline_btn") -> None:
    """Botão que dispara o pipeline completo em subprocess."""
    if st.button("Executar pipeline completo agora", key=key, type="primary"):
        with st.spinner("Executando pipeline completo... isso pode levar alguns segundos."):
            try:
                result = subprocess.run(
                    [sys.executable, "main.py", "--pipeline", "all"],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode == 0:
                    st.success("Pipeline executado com sucesso! Recarregue a página para ver os novos resultados.")
                    st.cache_data.clear()
                else:
                    st.error("Pipeline falhou.")
                    st.code(result.stderr or result.stdout, language="text")
            except subprocess.TimeoutExpired:
                st.error("Pipeline atingiu o tempo limite (10 min).")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Erro ao executar pipeline: {exc}")


def format_currency(value: float) -> str:
    """Formata valor monetário no padrão BR."""
    return f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_number(value: float, decimals: int = 0) -> str:
    """Formata número com decimais."""
    fmt = f"{{:,.{decimals}f}}"
    return fmt.format(value).replace(",", "X").replace(".", ",").replace("X", ".")


CLUSTER_LABEL_DESCRIPTIONS = {
    "loyal_high_value": "Clientes fiéis, recentes, frequentes e de alto valor. Foco em retenção VIP.",
    "at_risk": "Clientes com baixa atividade recente. Foco em campanhas de reativação.",
    "occasional_low_value": "Clientes esporádicos e de baixo ticket. Foco em incentivos de recompra.",
    "big_spenders": "Clientes com alto valor monetário. Foco em up-sell e cross-sell premium.",
    "frequent_buyers": "Clientes com alta frequência de compra. Foco em fidelização e recorrência.",
    "regular": "Clientes com comportamento padrão. Foco em comunicação massificada e nutrição.",
}


def cluster_description(label: str) -> str:
    """Retorna descrição em português para o label do cluster."""
    return CLUSTER_LABEL_DESCRIPTIONS.get(label, "Perfil sem descrição específica cadastrada.")
