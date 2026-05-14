"""Página de exibição de recomendações híbridas por cliente."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    file_exists,
    format_currency,
    format_number,
    get_paths,
    load_csv,
    render_pipeline_warning,
)

st.set_page_config(page_title="Recomendações | UNIVESP PI", page_icon=":sparkles:", layout="wide")


def main() -> None:
    """Renderiza a página de exploração das recomendações geradas."""
    paths = get_paths()

    required = {
        "recommendations": paths.outputs_predictions / "recommendations.csv",
        "popularity": paths.outputs_tables / "product_popularity.csv",
        "processed": paths.data_processed / "transactions_processed.csv",
    }
    missing = [name for name, path in required.items() if not file_exists(path)]
    if missing:
        render_pipeline_warning(", ".join(missing))
        st.stop()

    st.title("Recomendações por Cliente")
    st.caption("Top-N híbridas (filtragem colaborativa item-item + fallback de popularidade e recência).")

    recs = load_csv(str(required["recommendations"]))
    popularity = load_csv(str(required["popularity"]))
    transactions = load_csv(str(required["processed"]), parse_dates=["order_date"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes recomendados", format_number(recs["customer_id"].nunique()))
    c2.metric("Produtos no mix de saída", format_number(recs["product_id"].nunique()))
    c3.metric("Tamanho médio da lista", format_number(recs.groupby("customer_id").size().mean(), decimals=2))

    st.divider()
    st.subheader("Selecione um cliente")
    customer_ids = sorted(recs["customer_id"].astype(str).unique().tolist())
    default_idx = 0
    selected_customer = st.selectbox(
        "Cliente",
        customer_ids,
        index=default_idx,
        help=f"{len(customer_ids)} clientes com recomendações disponíveis.",
    )

    if not selected_customer:
        st.stop()

    customer_recs = recs[recs["customer_id"].astype(str) == selected_customer].sort_values("rank")
    customer_history = transactions[transactions["customer_id"].astype(str) == selected_customer]

    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Pedidos do cliente", format_number(customer_history["order_id"].nunique()))
    h2.metric("Produtos comprados", format_number(customer_history["product_id"].nunique()))
    h3.metric("Receita gerada", format_currency(customer_history["total_value"].sum()))
    if not customer_history.empty:
        last_date = customer_history["order_date"].max()
        h4.metric("Última compra", last_date.strftime("%d/%m/%Y"))
    else:
        h4.metric("Última compra", "—")

    st.divider()
    st.subheader(f"Top {len(customer_recs)} recomendações para {selected_customer}")

    if customer_recs.empty:
        st.warning(
            "Este cliente não recebeu recomendações no pipeline atual "
            "(possivelmente histórico insuficiente ou ausência de candidatos)."
        )
    else:
        display = customer_recs.copy()
        display["score"] = display["score"].round(4)
        st.dataframe(
            display[["rank", "product_id", "score"]],
            hide_index=True,
            use_container_width=True,
        )

        fig = px.bar(
            display.sort_values("score"),
            x="score",
            y="product_id",
            orientation="h",
            text="score",
            title=f"Pontuação das recomendações - {selected_customer}",
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Histórico de compras do cliente")
    if customer_history.empty:
        st.info("Sem histórico processado para este cliente.")
    else:
        history_summary = (
            customer_history.groupby("product_id", as_index=False)
            .agg(
                pedidos=("order_id", "nunique"),
                quantidade=("quantity", "sum"),
                receita=("total_value", "sum"),
                ultima_compra=("order_date", "max"),
            )
            .sort_values("receita", ascending=False)
        )
        st.dataframe(history_summary, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Produtos mais populares (fallback global)")
    st.dataframe(popularity.head(20), hide_index=True, use_container_width=True)

    st.divider()
    st.download_button(
        "Baixar recomendações deste cliente (CSV)",
        data=customer_recs.to_csv(index=False).encode("utf-8"),
        file_name=f"recommendations_{selected_customer}.csv",
        mime="text/csv",
    )
    st.download_button(
        "Baixar TODAS as recomendações (CSV)",
        data=recs.to_csv(index=False).encode("utf-8"),
        file_name="recommendations_all.csv",
        mime="text/csv",
    )


main()
