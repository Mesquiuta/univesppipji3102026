"""Análise Exploratória de Dados (EDA)."""

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
    load_json,
    render_pipeline_warning,
)

st.set_page_config(page_title="EDA | UNIVESP PI", page_icon=":bar_chart:", layout="wide")


def main() -> None:
    """Renderiza a página de EDA com KPIs, tendências e top rankings."""
    paths = get_paths()

    st.title("Análise Exploratória")
    st.caption("Padrões de comportamento, KPIs e tendências de receita.")

    required = {
        "kpis": paths.outputs_reports / "kpis.json",
        "retention": paths.outputs_reports / "retention_proxy_metrics.json",
        "concentration": paths.outputs_reports / "revenue_concentration.json",
        "summary": paths.outputs_reports / "eda_summary.json",
        "monthly_revenue": paths.outputs_tables / "revenue_by_month.csv",
        "monthly_ticket": paths.outputs_tables / "avg_ticket_by_month.csv",
        "top_products": paths.outputs_tables / "top_products.csv",
        "top_customers": paths.outputs_tables / "top_customers.csv",
        "orders_per_customer": paths.outputs_tables / "orders_per_customer.csv",
        "frequency_dist": paths.outputs_tables / "purchase_frequency_distribution.csv",
        "rfm_dist": paths.outputs_tables / "rfm_distribution.csv",
        "funnel": paths.outputs_tables / "repeat_purchase_funnel.csv",
        "outliers": paths.outputs_tables / "outlier_summary.csv",
    }

    missing = [name for name, path in required.items() if not file_exists(path)]
    if missing:
        render_pipeline_warning(", ".join(missing))
        st.stop()

    kpis = load_json(str(required["kpis"]))
    retention = load_json(str(required["retention"]))
    concentration = load_json(str(required["concentration"]))

    st.subheader("KPIs principais")
    c1, c2, c3 = st.columns(3)
    c1.metric("Receita bruta", format_currency(kpis["gross_revenue"]))
    c2.metric("Pedidos", format_number(kpis["n_orders"]))
    c3.metric("Clientes", format_number(kpis["n_customers"]))

    c4, c5, c6 = st.columns(3)
    c4.metric("Ticket médio", format_currency(kpis["avg_order_value"]))
    c5.metric("Itens / pedido", format_number(kpis["avg_items_per_order"], decimals=2))
    c6.metric("Taxa de recompra", f"{kpis['repeat_customer_rate'] * 100:.1f}%")

    st.divider()
    st.subheader("Receita ao longo do tempo")
    rev = load_csv(str(required["monthly_revenue"]), parse_dates=["month"])
    fig = px.line(rev, x="month", y="revenue", markers=True, title="Receita mensal")
    fig.update_layout(yaxis_title="Receita (R$)", xaxis_title="Mês")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Interpretação"):
        var = rev["revenue"].pct_change().abs().mean() * 100
        st.write(
            f"Receita mensal varia em média **{var:.1f}%** entre meses consecutivos. "
            f"O mês de maior receita foi **{rev.loc[rev['revenue'].idxmax(), 'month'].strftime('%Y-%m')}** "
            f"com **{format_currency(rev['revenue'].max())}**."
        )

    st.divider()
    st.subheader("Ticket médio mensal")
    ticket = load_csv(str(required["monthly_ticket"]), parse_dates=["month"])
    fig = px.line(ticket, x="month", y="avg_ticket", markers=True, title="Ticket médio por mês")
    fig.update_layout(yaxis_title="Ticket médio (R$)", xaxis_title="Mês")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 produtos por receita")
        top_products = load_csv(str(required["top_products"]))
        fig = px.bar(
            top_products.sort_values("revenue"),
            x="revenue",
            y="product_id",
            orientation="h",
            text="revenue",
            title="Top produtos",
        )
        fig.update_traces(texttemplate="R$ %{text:,.2f}", textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title="Receita (R$)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_products, hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Top 10 clientes por receita")
        top_customers = load_csv(str(required["top_customers"]))
        fig = px.bar(
            top_customers.sort_values("revenue"),
            x="revenue",
            y="customer_id",
            orientation="h",
            text="revenue",
            title="Top clientes",
        )
        fig.update_traces(texttemplate="R$ %{text:,.2f}", textposition="outside")
        fig.update_layout(yaxis_title="", xaxis_title="Receita (R$)")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_customers, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Frequência de pedidos por cliente")
    freq = load_csv(str(required["frequency_dist"]))
    fig = px.bar(
        freq,
        x="orders_per_customer",
        y="n_customers",
        title="Distribuição: nº de pedidos por cliente",
    )
    fig.update_layout(xaxis_title="Pedidos por cliente", yaxis_title="Nº de clientes")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Funil proxy de retenção")
    funnel = load_csv(str(required["funnel"]))
    fig = px.funnel(funnel, x="count", y="stage", title="Retenção via recompra")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(funnel, hide_index=True, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total de clientes", format_number(retention["total_customers"]))
    c2.metric("Taxa de recompra", f"{retention['repurchase_rate'] * 100:.1f}%")
    c3.metric("Média de dias entre pedidos", format_number(retention["avg_days_between_orders"], decimals=1))

    st.divider()
    st.subheader("Concentração de receita")
    c1, c2, c3 = st.columns(3)
    c1.metric("Top 10% clientes", f"{concentration['top_10pct_share'] * 100:.1f}% da receita")
    c2.metric("Top 20% clientes", f"{concentration['top_20pct_share'] * 100:.1f}% da receita")
    c3.metric("Índice tipo Gini", format_number(concentration["gini_like_index"], decimals=3))

    st.divider()
    st.subheader("Distribuição RFM")
    rfm = load_csv(str(required["rfm_dist"]))
    tab1, tab2, tab3 = st.tabs(["Recência", "Frequência", "Monetário"])
    with tab1:
        st.plotly_chart(px.histogram(rfm, x="recency", nbins=40, title="Distribuição de Recência (dias)"), use_container_width=True)
    with tab2:
        st.plotly_chart(px.histogram(rfm, x="frequency", nbins=40, title="Distribuição de Frequência"), use_container_width=True)
    with tab3:
        st.plotly_chart(px.histogram(rfm, x="monetary", nbins=40, title="Distribuição de Valor Monetário (R$)"), use_container_width=True)

    st.divider()
    st.subheader("Outliers nas variáveis numéricas (IQR)")
    outliers = load_csv(str(required["outliers"]))
    st.dataframe(outliers, hide_index=True, use_container_width=True)


main()
