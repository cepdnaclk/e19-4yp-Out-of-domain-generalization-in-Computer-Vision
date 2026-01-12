# main.py
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from loguru import logger

from biomedxpro.core.domain import DataSplit
from biomedxpro.engine.config import MasterConfig
from biomedxpro.engine.orchestrator import Orchestrator
from biomedxpro.impl.adapters import get_adapter
from biomedxpro.impl.data_loader import BiomedDataLoader
from biomedxpro.impl.evaluator import FitnessEvaluator
from biomedxpro.impl.llm_client import create_llm_client
from biomedxpro.impl.llm_operator import LLMOperator
from biomedxpro.impl.selection import RouletteWheelSelector
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

    # 3. Initialize Data Layer (The Translator & Processor)
    # Step 1: Get Adapter from the Registry
    logger.info(f"Initializing adapter: {config.dataset.adapter}")
    adapter = get_adapter(
        config.dataset.adapter,
        root=config.dataset.root,
        shots=config.dataset.shots,
        **config.dataset.adapter_params,
    )

    # Step 2: Load Samples (Standardization)
    logger.info(f"Loading samples from {config.dataset.root}...")
    train_samples = adapter.load_samples(DataSplit.TRAIN)
    val_samples = adapter.load_samples(DataSplit.VAL)
    logger.info(
        f"Found {len(train_samples)} training and {len(val_samples)} validation samples."
    )

    # Step 3, 4, 5: Process, Encode and Cache (The Loader)
    loader = BiomedDataLoader(
        cache_dir=config.dataset.cache_dir,
        batch_size=config.execution.batch_size,
        device=config.execution.device,
        num_workers=config.execution.dataloader_cpu_workers,
    )

    logger.info("Encapsulating data into optimized EncodedDataset artifacts...")
    train_dataset = loader.load_encoded_dataset(
        name=f"{config.dataset.name}_train",
        samples=train_samples,
        class_names=config.dataset.class_names,
    )

    val_dataset = loader.load_encoded_dataset(
        name=f"{config.dataset.name}_val",
        samples=val_samples,
        class_names=config.dataset.class_names,
    )

    # 5. Initialize Evolutionary Components
    logger.info("Bootstrapping evolutionary components...")
    llm_client = create_llm_client(config.llm)
    operator = LLMOperator(
        llm=llm_client, strategy=config.strategy, task_def=config.task
    )
    evaluator = FitnessEvaluator(
        device=config.execution.device, batch_size=config.execution.batch_size
    )
    selector = RouletteWheelSelector()

    # 6. Build & Run Orchestrator
    orchestrator = Orchestrator(
        evaluator=evaluator,
        operator=operator,
        selector=selector,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        params=config.evolution,
        recorder=recorder,
        exec_config=config.execution,
    )

    # Phases of the Island-Model Evolutionary Algorithm
    logger.info("Entering Phase I & II: Archipelago Initialization and Evolution")
    # Initialize handles Phase I (Concept Discovery) fallback if concepts is None
    orchestrator.initialize(concepts=config.task.concepts)

    champions = orchestrator.run()

    # 7. Final Championship Summary
    print("\n" + "=" * 60)
    logger.success(
        f"Evolution complete. Discovered {len(champions)} expert concept prompts."
    )
    print("=" * 60)
    for ind in champions:
        score = ind.get_fitness(config.evolution.target_metric)
        logger.info(f"Expert Concept: {ind.concept}")
        logger.info(
            f"   Validation Score ({config.evolution.target_metric}): {score:.4f}"
        )
        logger.info(f"   Positive Prompt: {ind.genotype.positive_prompt}")
        logger.info(f"   Negative Prompt: {ind.genotype.negative_prompt}")
        print("-" * 60)


if __name__ == "__main__":
    app()
