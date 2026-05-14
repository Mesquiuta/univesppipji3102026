"""Home do dashboard analítico de e-commerce - UNIVESP PI."""

from __future__ import annotations

import streamlit as st

from utils import (
    format_currency,
    format_number,
    get_paths,
    load_json,
    pipeline_status,
    render_pipeline_warning,
    render_run_pipeline_button,
)

st.set_page_config(
    page_title="Analytics E-commerce | UNIVESP PI",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    """Renderiza a página inicial do dashboard."""
    paths = get_paths()
    status = pipeline_status()

    st.title("Sistema Analítico de E-commerce")
    st.caption("Projeto Integrador - UNIVESP | EDA, segmentação RFM + K-Means, recomendação híbrida e avaliação")

    st.markdown(
        """
        Este dashboard apresenta os resultados de um pipeline completo de **ciência de dados** aplicado a dados
        transacionais de e-commerce. Use a barra lateral para navegar entre as etapas:

        - **Dataset**: preview e qualidade dos dados.
        - **EDA**: visão exploratória, KPIs e tendências.
        - **Segmentação**: perfis de clientes via RFM + K-Means.
        - **Recomendações**: top-N por cliente (filtragem colaborativa + fallback).
        - **Avaliação**: métricas técnicas e de negócio.
        - **Artefatos**: arquivos gerados pelo pipeline.
        """
    )

    st.divider()
    st.subheader("Status do pipeline")
    cols = st.columns(4)
    labels = [
        ("Dataset bruto", "raw_dataset"),
        ("Dataset processado", "processed_dataset"),
        ("Resumo EDA", "eda_summary"),
        ("KPIs", "kpis"),
        ("Clusters", "clusters"),
        ("Recomendações", "recommendations"),
        ("Avaliação", "evaluation"),
        ("Modelo K-Means", "model"),
    ]
    for idx, (label, key) in enumerate(labels):
        icon = ":white_check_mark:" if status[key] else ":x:"
        cols[idx % 4].markdown(f"{icon} **{label}**")

    if not all(status.values()):
        render_pipeline_warning("um ou mais artefatos do pipeline")
    st.markdown("")
    render_run_pipeline_button()

    st.divider()
    st.subheader("Resumo do dataset")

    if not status["eda_summary"]:
        render_pipeline_warning("eda_summary.json")
    else:
        summary = load_json(str(paths.outputs_reports / "eda_summary.json"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Linhas (transações)", format_number(summary["n_rows"]))
        c2.metric("Clientes únicos", format_number(summary["n_customers"]))
        c3.metric("Pedidos únicos", format_number(summary["n_orders"]))
        c4.metric("Produtos distintos", format_number(summary["n_products"]))

        c5, c6 = st.columns(2)
        c5.write(f"**Data mínima:** {summary['date_min']}")
        c6.write(f"**Data máxima:** {summary['date_max']}")

    st.divider()
    st.subheader("KPIs principais")

    if not status["kpis"]:
        render_pipeline_warning("kpis.json")
    else:
        kpis = load_json(str(paths.outputs_reports / "kpis.json"))
        c1, c2, c3 = st.columns(3)
        c1.metric("Receita bruta", format_currency(kpis["gross_revenue"]))
        c2.metric("Pedidos", format_number(kpis["n_orders"]))
        c3.metric("Clientes", format_number(kpis["n_customers"]))

        c4, c5, c6 = st.columns(3)
        c4.metric("Ticket médio", format_currency(kpis["avg_order_value"]))
        c5.metric("Itens / pedido", format_number(kpis["avg_items_per_order"], decimals=2))
        c6.metric("Taxa de recompra", f"{kpis['repeat_customer_rate'] * 100:.1f}%")

    st.divider()
    with st.expander("Sobre o projeto"):
        st.markdown(
            """
            **Objetivo:** apoiar decisões de marketing, retenção e recomendação em um e-commerce
            usando dados transacionais.

            **Pipeline:**
            1. *Preprocessing*: validação, limpeza, padronização e feature engineering.
            2. *EDA*: KPIs, tendências de receita, top clientes e produtos, distribuição RFM.
            3. *Segmentação*: clusterização K-Means com seleção automática de k via silhouette.
            4. *Recomendação*: matriz user-item, filtragem colaborativa item-item + fallback (popularidade e recência).
            5. *Avaliação*: split temporal holdout, precision@k, recall@k, MAP@k, cobertura e personalização.

            **Stack:** Python, pandas, scikit-learn, matplotlib, seaborn, Streamlit, Plotly.
            """
        )


if __name__ == "__main__":
    main()
else:
    main()
