"""Página de inspeção do dataset transacional."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import file_exists, format_number, get_paths, load_csv, render_pipeline_warning

st.set_page_config(page_title="Dataset | UNIVESP PI", page_icon=":card_index_dividers:", layout="wide")


def main() -> None:
    """Renderiza a página de exploração do dataset."""
    paths = get_paths()

    st.title("Dataset")
    st.caption("Visão geral, qualidade e estatísticas do conjunto de transações.")

    raw_path = paths.data_raw / "transactions.csv"
    processed_path = paths.data_processed / "transactions_processed.csv"

    st.sidebar.subheader("Fonte do dataset")
    fonte = st.sidebar.radio(
        "Selecionar arquivo",
        ["Processado (após limpeza)", "Bruto (original)"],
        index=0,
    )
    chosen_path = processed_path if fonte == "Processado (após limpeza)" else raw_path

    if not file_exists(chosen_path):
        render_pipeline_warning(str(chosen_path.name))
        st.stop()

    df = load_csv(str(chosen_path), parse_dates=["order_date"]) if "processado" in fonte.lower() else load_csv(str(chosen_path))

    st.success(f"Arquivo carregado: `{chosen_path}`")

    st.divider()
    st.subheader("Métricas básicas")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", format_number(df.shape[0]))
    c2.metric("Colunas", format_number(df.shape[1]))
    if "customer_id" in df.columns:
        c3.metric("Clientes únicos", format_number(df["customer_id"].nunique()))
    if "product_id" in df.columns:
        c4.metric("Produtos únicos", format_number(df["product_id"].nunique()))

    st.divider()
    st.subheader("Amostra do dataset")
    n_preview = st.slider("Quantas linhas exibir?", min_value=5, max_value=200, value=20, step=5)
    st.dataframe(df.head(n_preview), use_container_width=True)

    st.divider()
    st.subheader("Tipos de dados")
    dtypes = pd.DataFrame({"coluna": df.columns, "tipo": df.dtypes.astype(str).values})
    st.dataframe(dtypes, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Valores ausentes")
    missing = (
        pd.DataFrame(
            {
                "coluna": df.columns,
                "n_faltantes": df.isna().sum().values,
                "ratio_faltantes": (df.isna().mean() * 100).values,
            }
        )
        .assign(ratio_faltantes=lambda x: x["ratio_faltantes"].round(2))
    )
    st.dataframe(missing, hide_index=True, use_container_width=True)

    if df.select_dtypes(include="number").shape[1] > 0:
        st.divider()
        st.subheader("Estatísticas descritivas (numéricas)")
        describe = df.describe().transpose().reset_index().rename(columns={"index": "métrica"})
        st.dataframe(describe, hide_index=True, use_container_width=True)

    st.divider()
    st.download_button(
        "Baixar amostra exibida (CSV)",
        data=df.head(n_preview).to_csv(index=False).encode("utf-8"),
        file_name="sample_transactions.csv",
        mime="text/csv",
    )


main()
