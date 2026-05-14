"""Exportação padronizada de estruturas em JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_json(payload: dict[str, Any], output_path: Path) -> Path:
    """Salva dicionário em arquivo JSON e retorna caminho salvo."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    return output_path

