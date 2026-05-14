"""Métricas de avaliação técnica e de negócio."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from utils import (
    file_exists,
    format_number,
    get_paths,
    load_json,
    render_pipeline_warning,
)

st.set_page_config(page_title="Avaliação | UNIVESP PI", page_icon=":chart_with_upwards_trend:", layout="wide")


def main() -> None:
    """Renderiza métricas de clustering, recomendação e negócio."""
    paths = get_paths()
    report_path = paths.outputs_reports / "evaluation_report.json"
    holdout_path = paths.outputs_reports / "holdout_recommendation_eval.json"

    if not file_exists(report_path):
        render_pipeline_warning("evaluation_report.json")
        st.stop()

    report = load_json(str(report_path))
    metrics = report["metrics"]

    st.title("Avaliação")
    st.caption("Métricas técnicas e de negócio do pipeline analítico.")

    st.subheader("Métricas de clustering")
    clu = metrics["clustering"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Silhouette Score", format_number(clu["silhouette_score"], decimals=4) if clu["silhouette_score"] is not None else "—")
    c2.metric("Davies-Bouldin (↓)", format_number(clu["davies_bouldin_score"], decimals=4) if clu["davies_bouldin_score"] is not None else "—")
    c3.metric("Calinski-Harabasz (↑)", format_number(clu["calinski_harabasz_score"], decimals=2) if clu["calinski_harabasz_score"] is not None else "—")
    with st.expander("Como interpretar"):
        st.markdown(
            """
            - **Silhouette score**: varia de -1 a 1. Quanto maior, mais bem definidos os clusters.
              Valores > 0.5 indicam estruturas razoavelmente separadas.
            - **Davies-Bouldin**: quanto menor, melhor. Mede o quão similares são os clusters próximos
              em relação ao tamanho deles.
            - **Calinski-Harabasz**: quanto maior, melhor. Razão entre dispersão entre-clusters
              e dentro-de-cluster.
            """
        )

    st.divider()
    st.subheader("Métricas do recomendador (holdout temporal)")
    rec = metrics["recommender"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Precision@k", f"{rec['precision_at_k'] * 100:.2f}%")
    c2.metric("Recall@k", f"{rec['recall_at_k'] * 100:.2f}%")
    c3.metric("MAP@k", f"{rec['map_at_k'] * 100:.2f}%")
    with st.expander("Como interpretar"):
        st.markdown(
            """
            O recomendador é avaliado em um *holdout* temporal: o último pedido de cada cliente
            com 2+ pedidos é removido e o sistema tenta prever esses itens com base nos pedidos anteriores.

            - **Precision@k**: fração das k recomendações que estão no holdout.
            - **Recall@k**: fração dos itens do holdout que aparecem nas k recomendações.
            - **MAP@k**: precisão média sensível ao ranking (penaliza acertos no fim da lista).
            """
        )

    st.divider()
    st.subheader("Métricas de negócio")
    biz = metrics["business"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Cobertura do catálogo", f"{biz['catalog_coverage'] * 100:.2f}%")
    c2.metric("Personalização (1 - Jaccard)", format_number(biz["personalization"], decimals=4))
    c3.metric("Tamanho médio da lista", format_number(biz["avg_recommendation_list_size"], decimals=2))
    with st.expander("Como interpretar"):
        st.markdown(
            """
            - **Cobertura do catálogo**: percentual de produtos do catálogo que aparecem em alguma recomendação.
              Cobertura mais alta = sistema explora mais do catálogo.
            - **Personalização**: 1 - similaridade Jaccard média entre listas de usuários distintos.
              Valores próximos de 1 indicam recomendações altamente personalizadas.
            - **Tamanho médio**: número médio de itens efetivamente recomendados por usuário.
            """
        )

    st.divider()
    st.subheader("Split treino/teste")
    split = metrics["split"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Linhas de treino", format_number(split["train_rows"]))
    c2.metric("Linhas de teste", format_number(split["test_rows"]))
    c3.metric("Pedidos de treino", format_number(split["train_orders"]))
    c4.metric("Pedidos de teste", format_number(split["test_orders"]))
    c5.metric("Usuários avaliados", format_number(split["evaluated_users"]))

    st.divider()
    st.subheader("Comparação visual das métricas")
    cluster_chart = pd.DataFrame(
        {
            "métrica": ["Silhouette", "Davies-Bouldin (↓)", "Calinski-Harabasz/1000"],
            "valor": [
                clu["silhouette_score"] or 0,
                clu["davies_bouldin_score"] or 0,
                (clu["calinski_harabasz_score"] or 0) / 1000.0,
            ],
        }
    )
    fig = px.bar(cluster_chart, x="métrica", y="valor", title="Métricas de clustering (normalizadas)")
    st.plotly_chart(fig, use_container_width=True)

    rec_chart = pd.DataFrame(
        {
            "métrica": ["Precision@k", "Recall@k", "MAP@k"],
            "valor": [rec["precision_at_k"], rec["recall_at_k"], rec["map_at_k"]],
        }
    )
    fig = px.bar(rec_chart, x="métrica", y="valor", title="Métricas do recomendador (escala 0-1)")
    st.plotly_chart(fig, use_container_width=True)

    biz_chart = pd.DataFrame(
        {
            "métrica": ["Cobertura catálogo", "Personalização", "Tamanho médio lista (norm)"],
            "valor": [
                biz["catalog_coverage"],
                biz["personalization"],
                biz["avg_recommendation_list_size"] / 10.0,
            ],
        }
    )
    fig = px.bar(biz_chart, x="métrica", y="valor", title="Métricas de negócio")
    st.plotly_chart(fig, use_container_width=True)

    if file_exists(holdout_path):
        st.divider()
        with st.expander("Ver detalhes do holdout"):
            holdout = load_json(str(holdout_path))
            st.json(
                {
                    "n_ground_truth_users": len(holdout.get("ground_truth_users", [])),
                    "exemplo_predicoes": dict(list(holdout.get("predicted_items", {}).items())[:5]),
                }
            )

    st.divider()
    st.caption(f"Relatório gerado em: {report.get('generated_at', '—')}")


main()
