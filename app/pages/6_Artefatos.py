"""Listagem dos artefatos gerados pelo pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from utils import format_number, get_paths

st.set_page_config(page_title="Artefatos | UNIVESP PI", page_icon=":file_folder:", layout="wide")


def list_files(directory: Path) -> pd.DataFrame:
    """Lista arquivos de um diretório em DataFrame com tamanho em KB."""
    if not directory.exists():
        return pd.DataFrame(columns=["arquivo", "tamanho_kb", "caminho"])
    records = []
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            records.append(
                {
                    "arquivo": path.name,
                    "tamanho_kb": round(path.stat().st_size / 1024.0, 2),
                    "caminho": str(path.relative_to(directory.parent)),
                }
            )
    return pd.DataFrame(records)


def show_section(title: str, directory: Path) -> None:
    """Exibe seção com tabela de arquivos e botões de download."""
    st.subheader(title)
    df = list_files(directory)
    if df.empty:
        st.info(f"Sem arquivos em `{directory}` ainda. Rode o pipeline.")
        return
    st.dataframe(df, hide_index=True, use_container_width=True)

    extension_filter = st.multiselect(
        f"Filtrar por extensão em {title}",
        sorted({Path(f).suffix.lstrip(".") for f in df["arquivo"]}),
        default=[],
        key=f"ext_{title}",
    )
    if extension_filter:
        df = df[df["arquivo"].apply(lambda x: Path(x).suffix.lstrip(".") in extension_filter)]

    selected = st.selectbox(
        f"Baixar arquivo de {title}",
        ["(nenhum)"] + df["arquivo"].tolist(),
        key=f"dl_{title}",
    )
    if selected and selected != "(nenhum)":
        full_path = directory / selected
        with open(full_path, "rb") as fp:
            st.download_button(
                f"Baixar {selected}",
                data=fp.read(),
                file_name=selected,
                mime="application/octet-stream",
                key=f"dlbtn_{title}_{selected}",
            )


def main() -> None:
    """Renderiza explorador de artefatos."""
    paths = get_paths()

    st.title("Artefatos gerados pelo pipeline")
    st.caption("Tabelas, figuras, relatórios, predições e modelos exportados.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tabelas", format_number(len(list(paths.outputs_tables.glob('*.csv')))))
    c2.metric("Figuras", format_number(len(list(paths.outputs_figures.glob('*.png')))))
    c3.metric("Relatórios", format_number(len(list(paths.outputs_reports.glob('*.json')))))
    c4.metric("Predições", format_number(len(list(paths.outputs_predictions.glob('*')))))

    st.divider()
    show_section("Tabelas (CSV)", paths.outputs_tables)
    st.divider()
    show_section("Figuras (PNG)", paths.outputs_figures)
    st.divider()
    show_section("Relatórios (JSON)", paths.outputs_reports)
    st.divider()
    show_section("Predições", paths.outputs_predictions)
    st.divider()
    show_section("Modelos persistidos", paths.models)
    st.divider()
    show_section("Dados intermediários", paths.data_interim)
    st.divider()
    show_section("Dados processados", paths.data_processed)

    st.divider()
    st.subheader("Visualizar figuras")
    figs = sorted(paths.outputs_figures.glob("*.png"))
    if not figs:
        st.info("Nenhuma figura disponível ainda.")
    else:
        for fig in figs:
            st.markdown(f"**{fig.name}**")
            st.image(str(fig), use_container_width=True)

main()
