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
    # 1. The Problem
    task_config: Path = typer.Option(
        ..., "--task", "-t", help="Path to Task/Dataset config", exists=True
    ),
    # 2. The Method
    evo_config: Path = typer.Option(
        ..., "--evolution", "-e", help="Path to Evolution params", exists=True
    ),
    # 3. The Intelligence
    llm_config: Path = typer.Option(
        ..., "--llm", "-l", help="Path to LLM config", exists=True
    ),
    # 4. The Hardware
    exec_config: Path = typer.Option(
        ..., "--exec", "-x", help="Path to Execution config", exists=True
    ),
    # 5. Overrides
    shots: int = typer.Option(16, "--shots", "-s", help="Number of few-shot examples"),
    metric: str = typer.Option(
        "f1_macro", "--metric", "-m", help="Optimization target metric"
    ),
    exp_name: Optional[str] = typer.Option(
        None, "--exp-name", "-n", help="Optional experiment name override"
    ),
) -> None:
    """
    Executes the Concept-Driven Island Evolution pipeline using composable configurations.
    """
    # 0. Load environment variables (API Keys from .env)
    load_dotenv()

    # 1. Load Configuration (Assembly Phase)
    try:
        config = MasterConfig.from_composable(
            task_path=task_config,
            evo_path=evo_config,
            llm_path=llm_config,
            exec_path=exec_config,
            target_metric=metric,
            shots=shots,
        )
    except Exception as e:
        logger.error(f"Failed to parse config: {e}")
        sys.exit(1)

    experiment_name = exp_name or config.dataset.name

    # 2. Setup Logging & Persistence
    setup_logging(experiment_name, console_level="DEBUG")
    recorder = HistoryRecorder(experiment_name=experiment_name)
    logger.info(f"Loaded task config from {task_config}")
    logger.info(f"Loaded evolution config from {evo_config}")
    logger.info(f"Loaded LLM config from {llm_config}")
    logger.info(f"Loaded execution config from {exec_config}")
    logger.info(f"Override: Shots = {shots}")
    logger.info(f"Override: Metric = {metric}")
    logger.info(f"Starting experiment: {experiment_name}")

    # 3. Build World (Factory)
    train_ds, val_ds, test_ds = builder.load_datasets(config)
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
        temperature=1.0,
    )

    # 2. Infrastructure Work (Heavy Lifting)
    # Note: access evaluator via orchestrator public property
    metrics = orchestrator.evaluator.evaluate_ensemble(ensemble, test_ds)

    # 7. Reporting Phase (Deployment)
    reporting.print_ensemble_results(metrics)


if __name__ == "__main__":
    app()
