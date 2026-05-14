"""Segmentação RFM + K-Means."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from utils import (
    cluster_description,
    file_exists,
    format_currency,
    format_number,
    get_paths,
    load_csv,
    load_json,
    render_pipeline_warning,
)

st.set_page_config(page_title="Segmentação | UNIVESP PI", page_icon=":busts_in_silhouette:", layout="wide")


def main() -> None:
    """Renderiza visualização da segmentação RFM por K-Means."""
    paths = get_paths()
    required = {
        "rfm_clusters": paths.outputs_tables / "rfm_clusters.csv",
        "cluster_summary": paths.outputs_tables / "cluster_summary.csv",
        "diagnostics": paths.outputs_tables / "k_selection_diagnostics.csv",
        "selection_meta": paths.outputs_reports / "k_selection_summary.json",
    }
    missing = [name for name, path in required.items() if not file_exists(path)]
    if missing:
        render_pipeline_warning(", ".join(missing))
        st.stop()

    st.title("Segmentação de Clientes")
    st.caption("Recência, Frequência e Monetário (RFM) com K-Means e seleção automática de k.")

    clusters = load_csv(str(required["rfm_clusters"]))
    summary = load_csv(str(required["cluster_summary"]))
    diagnostics = load_csv(str(required["diagnostics"]))
    selection_meta = load_json(str(required["selection_meta"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes segmentados", format_number(clusters.shape[0]))
    c2.metric("Clusters selecionados", selection_meta["selected_k"])
    c3.metric("Seleção automática?", "Sim" if selection_meta.get("auto_select_k") else "Não")
    c4.metric("K de fallback", selection_meta.get("fallback_k", "—"))

    st.divider()
    st.subheader("Diagnóstico da escolha de K")
    fig = px.line(
        diagnostics.dropna(subset=["silhouette"]),
        x="k",
        y="silhouette",
        markers=True,
        title="Silhouette score por número de clusters",
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        diagnostics.dropna(subset=["inertia"]),
        x="k",
        y="inertia",
        markers=True,
        title="Inércia (elbow) por número de clusters",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(diagnostics, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Distribuição de clientes por cluster")
    if "cluster_label" in clusters.columns:
        dist = (
            clusters.groupby(["cluster", "cluster_label"], as_index=False)
            .size()
            .rename(columns={"size": "n_clientes"})
            .sort_values("cluster")
        )
        fig = px.bar(
            dist,
            x="cluster",
            y="n_clientes",
            color="cluster_label",
            text="n_clientes",
            title="Tamanho dos clusters",
        )
    else:
        dist = clusters.groupby("cluster", as_index=False).size().rename(columns={"size": "n_clientes"})
        fig = px.bar(dist, x="cluster", y="n_clientes", text="n_clientes", title="Tamanho dos clusters")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Resumo dos clusters")
    summary_render = summary.copy()
    if "cluster_label" in clusters.columns:
        label_map = clusters.groupby("cluster")["cluster_label"].first().to_dict()
        summary_render["cluster_label"] = summary_render["cluster"].map(label_map)
        cols = ["cluster", "cluster_label", "recency", "frequency", "monetary"]
        summary_render = summary_render[cols]
    st.dataframe(summary_render, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Interpretação dos clusters")
    if "cluster_label" in clusters.columns:
        for _, row in summary_render.iterrows():
            label = row.get("cluster_label", "regular")
            st.markdown(
                f"**Cluster {int(row['cluster'])} — {label}**  \n"
                f":blue[Recência média:] {row['recency']:.1f} dias &nbsp;|&nbsp; "
                f":blue[Frequência média:] {row['frequency']:.2f} pedidos &nbsp;|&nbsp; "
                f":blue[Valor médio:] {format_currency(row['monetary'])}  \n"
                f"_{cluster_description(label)}_"
            )

    st.divider()
    st.subheader("Visualização 3D dos clusters (RFM)")
    fig = px.scatter_3d(
        clusters,
        x="recency",
        y="frequency",
        z="monetary",
        color="cluster_label" if "cluster_label" in clusters.columns else "cluster",
        hover_data=["customer_id"],
        title="Clientes no espaço RFM",
        opacity=0.7,
    )
    fig.update_traces(marker=dict(size=4))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Tabela completa de clientes segmentados")
    label_filter = None
    if "cluster_label" in clusters.columns:
        labels_disp = ["(todos)"] + sorted(clusters["cluster_label"].dropna().unique().tolist())
        label_filter = st.selectbox("Filtrar por perfil:", labels_disp)
    filtered = clusters if (label_filter in (None, "(todos)")) else clusters[clusters["cluster_label"] == label_filter]
    st.dataframe(filtered, hide_index=True, use_container_width=True)

    st.download_button(
        "Baixar segmentação completa (CSV)",
        data=clusters.to_csv(index=False).encode("utf-8"),
        file_name="rfm_clusters.csv",
        mime="text/csv",
    )


main()
