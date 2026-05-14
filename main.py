"""Ponto de entrada principal para execução de pipelines."""

from __future__ import annotations

import argparse
import json
from typing import Callable

from src.config.settings import load_settings
from src.pipelines import run_all, run_eda, run_evaluation, run_preprocessing, run_recommendation, run_segmentation


def parse_args() -> argparse.Namespace:
    """Configura argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Pipeline analítico de e-commerce")
    parser.add_argument(
        "--pipeline",
        choices=["all", "preprocessing", "eda", "segmentation", "recommendation", "evaluation"],
        default="all",
        help="Pipeline a executar.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Caminho opcional para arquivo JSON de configuração.",
    )
    return parser.parse_args()


def main() -> None:
    """Executa pipeline selecionado pelo usuário."""
    args = parse_args()
    settings = load_settings(args.config)

    pipeline_map: dict[str, Callable] = {
        "all": run_all.run,
        "preprocessing": run_preprocessing.run,
        "eda": run_eda.run,
        "segmentation": run_segmentation.run,
        "recommendation": run_recommendation.run,
        "evaluation": run_evaluation.run,
    }
    result = pipeline_map[args.pipeline](settings)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

