# main.py
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from loguru import logger

from biomedxpro.core.domain import PromptEnsemble
from biomedxpro.engine import builder
from biomedxpro.engine.config import MasterConfig
from biomedxpro.utils import reporting
from biomedxpro.utils.history import HistoryRecorder
from biomedxpro.utils.logging import setup_logging

app = typer.Typer(help="BioMedXPro Evolutionary Engine")


@app.command()
def run(
    config_path: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="Path to the experiment YAML config (e.g., experiments/melanoma_dg.yaml)",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    exp_name: Optional[str] = typer.Option(
        None, "--exp-name", "-n", help="Optional experiment name override"
    ),
) -> None:
    """
    Executes the Concept-Driven Island Evolution pipeline based on a YAML configuration.
    """
    # 0. Load environment variables (API Keys from .env)
    load_dotenv()

    # 1. Load Configuration
    try:
        config = MasterConfig.from_yaml(config_path)
    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        sys.exit(1)

    experiment_name = exp_name or config.dataset.name

    # 2. Setup Logging & Persistence
    setup_logging(experiment_name, console_level="DEBUG")
    recorder = HistoryRecorder(experiment_name=experiment_name)
    logger.info(f"Loaded config from {config_path}")
    logger.info(f"Starting experiment: {experiment_name}")

    # 3. Build World (Factory)
    train_ds, val_ds = builder.load_datasets(config)
    orchestrator = builder.build_orchestrator(
        config, train_ds, val_ds, recorder=recorder
    )

    # 4. Evolution Phase (Execution)
    logger.info("Entering Phase I & II: Archipelago Evolution")
    # Initialize handles Phase I (Concept Discovery) fallback if concepts is None
    orchestrator.initialize(concepts=config.task.concepts)

    champions = orchestrator.run()

    # 5. Reporting Phase (Evolution)
    reporting.print_champion_summary(champions, config.evolution.target_metric)

    # 6. Deployment Phase (Execution)
    logger.info("Constructing Prompt Ensemble from champions...")

    # 1. Instantiate Domain Entity
    ensemble = PromptEnsemble.from_individuals(
        champions,
        metric=config.evolution.target_metric,
        temperature=0.05,
    )

    # 2. Infrastructure Work (Heavy Lifting)
    # Note: access evaluator via orchestrator public property
    metrics = orchestrator.evaluator.evaluate_ensemble(ensemble, val_ds)

    # 7. Reporting Phase (Deployment)
    reporting.print_ensemble_results(metrics)


if __name__ == "__main__":
    app()
