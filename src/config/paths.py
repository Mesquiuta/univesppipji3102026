"""Gerenciamento de caminhos do projeto."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Agrupa caminhos físicos usados pelo projeto."""

    root: Path
    data_raw: Path
    data_interim: Path
    data_processed: Path
    data_external: Path
    models: Path
    outputs: Path
    outputs_figures: Path
    outputs_tables: Path
    outputs_reports: Path
    outputs_predictions: Path


def get_project_root() -> Path:
    """Retorna o diretório raiz do repositório."""
    return Path(__file__).resolve().parents[2]


def build_paths(root: Path | None = None) -> ProjectPaths:
    """Constrói a estrutura de caminhos a partir da raiz."""
    base = root or get_project_root()
    return ProjectPaths(
        root=base,
        data_raw=base / "data" / "raw",
        data_interim=base / "data" / "interim",
        data_processed=base / "data" / "processed",
        data_external=base / "data" / "external",
        models=base / "models",
        outputs=base / "outputs",
        outputs_figures=base / "outputs" / "figures",
        outputs_tables=base / "outputs" / "tables",
        outputs_reports=base / "outputs" / "reports",
        outputs_predictions=base / "outputs" / "predictions",
    )


def ensure_directories(paths: ProjectPaths) -> None:
    """Garante que todos os diretórios necessários existam."""
    for directory in (
        paths.data_raw,
        paths.data_interim,
        paths.data_processed,
        paths.data_external,
        paths.models,
        paths.outputs,
        paths.outputs_figures,
        paths.outputs_tables,
        paths.outputs_reports,
        paths.outputs_predictions,
    ):
        directory.mkdir(parents=True, exist_ok=True)

