"""Composição de relatórios de execução de pipelines."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def build_run_report(
    run_name: str,
    metrics: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Monta estrutura padrão de relatório."""
    return {
        "run_name": run_name,
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": metrics,
        "artifacts": artifacts or {},
    }

