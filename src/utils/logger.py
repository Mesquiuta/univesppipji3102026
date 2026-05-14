"""Funções de logging reutilizáveis."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configura logging global do projeto."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Retorna logger nomeado."""
    return logging.getLogger(name)

