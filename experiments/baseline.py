#!/usr/bin/env python3
"""
Baseline Prompt Evaluation.

This script evaluates a simple baseline prompt strategy using the format:
"An {image_modality} of a {class_name}"

This provides a performance floor to compare against evolved prompts.

Usage:
    uv run experiments/baseline.py --task configs/tasks/btmri.yaml
    uv run experiments/baseline.py --task configs/tasks/derm7pt-wConcepts.yaml --shots 32
"""

from pathlib import Path

import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from biomedxpro.core.domain import CreationOperation, Individual, PromptGenotype
from biomedxpro.engine import builder
from biomedxpro.engine.config import ExecutionConfig, MasterConfig
from biomedxpro.impl.evaluator import FitnessEvaluator

app = typer.Typer(help="Baseline Prompt Evaluation")
console = Console()


def create_baseline_individual(
    image_modality: str,
    class_names: list[str],
) -> Individual:
    """
    Create a baseline individual with simple prompt format.

    Args:
        image_modality: The imaging modality (e.g., "MRI scan", "histopathology image")
        class_names: List of class names

    Returns:
        Individual with baseline prompts
    """
    # Generate simple baseline prompts
    prompts = [f"A {image_modality} of a {class_name}" for class_name in class_names]

    genotype = PromptGenotype(prompts=tuple(prompts))
    individual = Individual(
        id="baseline",
        genotype=genotype,
        concept="baseline",
        generation_born=0,
        operation=CreationOperation.INITIALIZATION,
    )

    logger.info("Created baseline individual with prompts:")
    for class_name, prompt in zip(class_names, prompts):
        logger.info(f"  {class_name}: {prompt}")

    return individual


@app.command()
def evaluate(
    task_config: Path = typer.Option(
        ...,
        "--task",
        "-t",
        help="Path to task config (required)",
        exists=True,
    ),
    exec_config: Path = typer.Option(
        Path("configs/execution/gpu_default.yaml"),
        "--exec",
        "-x",
        help="Path to execution config",
        exists=True,
    ),
    split: str = typer.Option(
        "test",
        "--split",
        help="Which split to evaluate on: val or test",
    ),
) -> None:
    """
    Evaluate baseline prompts on a dataset.

    Example:
        uv run experiments/baseline.py --task configs/tasks/btmri.yaml
        uv run experiments/baseline.py --task configs/tasks/btmri.yaml --split val
    """
    logger.info("=" * 80)
    logger.info("BASELINE PROMPT EVALUATION")
    logger.info("=" * 80)

    # Load configurations
    logger.info(f"Loading task config from {task_config}")

    with open(task_config, "r") as f:
        task_data = yaml.safe_load(f)

    with open(exec_config, "r") as f:
        exec_data = yaml.safe_load(f)

    # Build minimal config
    from biomedxpro.core.domain import EvolutionParams, TaskDefinition
    from biomedxpro.impl.config import DatasetConfig

    config = MasterConfig(
        task=TaskDefinition.from_dict(task_data["task"]),
        dataset=DatasetConfig.from_dict(task_data["dataset"]),
        evolution=EvolutionParams(
            initial_pop_size=10,
            generations=1,
            target_metric="accuracy",
        ),
        llm=None,  # type: ignore[arg-type]
        execution=ExecutionConfig.from_dict(exec_data.get("execution", {})),
    )

    # Load datasets
    logger.info("Loading encoded datasets...")
    train_ds, val_ds, test_ds = builder.load_datasets(config)

    # Select evaluation split
    split_map = {
        "val": val_ds,
        "test": test_ds,
    }

    if split not in split_map:
        logger.error(f"Invalid split '{split}'. Choose from: val, test")
        raise typer.Exit(code=1)

    eval_ds = split_map[split]
    logger.info(f"Evaluating on {split} split ({eval_ds.num_samples} samples)")

    # Create baseline individual
    logger.info("=" * 80)
    logger.info("CREATING BASELINE INDIVIDUAL")
    logger.info("=" * 80)

    baseline = create_baseline_individual(
        image_modality=config.task.image_modality,
        class_names=config.dataset.class_names,
    )

    # Build evaluator
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATING BASELINE")
    logger.info("=" * 80)

    evaluator = FitnessEvaluator(
        device=config.execution.device,
        batch_size=config.execution.batch_size,
    )

    # Evaluate the baseline individual
    evaluator.evaluate([baseline], eval_ds)

    # Print results
    if baseline.metrics is None:
        logger.error("Evaluation failed - no metrics available")
        raise typer.Exit(code=1)

    metrics = baseline.metrics

    print("\n" + "=" * 80)
    logger.success("BASELINE RESULTS")
    print("=" * 80)

    # Create results table
    table = Table(
        title="Baseline Performance", show_header=True, header_style="bold cyan"
    )
    table.add_column("Metric", style="yellow")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Accuracy", f"{metrics['accuracy']:.4f}")
    table.add_row("F1 Score (Macro)", f"{metrics['f1_macro']:.4f}")
    table.add_row("F1 Score (Weighted)", f"{metrics['f1_weighted']:.4f}")
    table.add_row("AUROC", f"{metrics['auc']:.4f}")
    table.add_row("Inverted BCE", f"{metrics['inverted_bce']:.4f}")

    console.print(table)

    # Show confusion matrix
    cm = metrics.get("confusion_matrix")
    if cm is not None:
        print("\n" + "-" * 80)
        logger.info("Confusion Matrix:")
        print("-" * 80)

        # Print header
        header = "         " + " ".join(
            f"{name:>10s}" for name in config.dataset.class_names
        )
        logger.info(header)

        # Print rows
        for i, row in enumerate(cm):
            row_str = f"{config.dataset.class_names[i]:>10s} " + " ".join(
                f"{val:>10d}" for val in row
            )
            logger.info(row_str)

    # Summary
    print("\n" + "=" * 80)
    logger.info("SUMMARY")
    print("=" * 80)
    logger.info(f"Task: {config.task.task_name}")
    logger.info(f"Split: {split} ({eval_ds.num_samples} samples)")
    logger.info("Baseline Strategy: 'An {{image_modality}} of a {{class_name}}'")
    logger.success(f"✅ Baseline Accuracy: {metrics['accuracy']:.4f}")
    logger.success(f"✅ Baseline F1 Score: {metrics['f1_macro']:.4f}")
    logger.success(f"✅ Baseline Inverted BCE: {metrics['inverted_bce']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    app()
