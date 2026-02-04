#!/usr/bin/env python3
"""
Temperature Sweep Experiment for Ensemble Evaluation.

This script validates the hypothesis that the ensemble underperformance is caused
by high temperature (T=1.0) averaging high-quality specialists with low-performing prompts.

Usage:
    uv run experiments/evaluate_temperature.py logs/your_history.jsonl
    uv run experiments/evaluate_temperature.py logs/your_history.jsonl --temperatures 0.1 0.5 1.0
    uv run experiments/evaluate_temperature.py logs/your_history.jsonl --metric f1_score
"""

from pathlib import Path
from typing import TypedDict

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from biomedxpro.analysis import EvolutionHistory, HistoryRecord
from biomedxpro.core.domain import (
    EvaluationMetrics,
    Individual,
    MetricName,
    PromptEnsemble,
)
from biomedxpro.engine import builder
from biomedxpro.engine.config import MasterConfig


class TemperatureResult(TypedDict):
    """Type definition for temperature experiment results."""

    temperature: float
    accuracy: float
    f1_score: float
    precision: float
    recall: float
    metric_value: float
    metrics: EvaluationMetrics


app = typer.Typer(help="Ensemble Temperature Evaluation Experiment")
console = Console()


def reconstruct_individuals_from_champions(
    champions: dict[str, HistoryRecord], metric: MetricName
) -> list[Individual]:
    """
    Reconstruct Individual domain objects from history records.

    Args:
        champions: Dictionary mapping concept -> champion HistoryRecord
        metric: Metric name to use for logging

    Returns:
        List of Individual objects with metrics loaded
    """
    individuals = []

    for concept, record in champions.items():
        if record.metrics is None:
            logger.warning(f"Champion for '{concept}' has no metrics, skipping")
            continue

        # Reconstruct Individual
        from biomedxpro.core.domain import CreationOperation, Individual, PromptGenotype

        genotype = PromptGenotype(prompts=tuple(record.prompts))
        individual = Individual(
            id=record.id,
            genotype=genotype,
            concept=concept,
            generation_born=record.generation_born,
            operation=CreationOperation.LLM_MUTATION,  # Historical record
        )

        # Load metrics
        from biomedxpro.core.domain import EvaluationMetrics

        # Cast to EvaluationMetrics for type safety
        metrics: EvaluationMetrics = record.metrics  # type: ignore[assignment]
        individual.update_metrics(metrics)

        individuals.append(individual)
        logger.info(
            f"✓ Loaded champion '{concept}': "
            f"{metric}={metrics[metric]:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    return individuals


@app.command()
def evaluate(
    history_file: Path = typer.Argument(
        ...,
        help="Path to *_history.jsonl file from evolution run",
        exists=True,
    ),
    task_config: Path = typer.Option(
        Path("configs/tasks/btmri.yaml"),
        "--task",
        "-t",
        help="Path to task config (for dataset loading)",
        exists=True,
    ),
    exec_config: Path = typer.Option(
        Path("configs/execution/cpu_local.yaml"),
        "--exec",
        "-x",
        help="Path to execution config",
        exists=True,
    ),
    temperatures: list[float] = typer.Option(
        [2.0, 1.0, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01],
        "--temperatures",
        "-T",
        help="Temperature values to test",
    ),
    metric: MetricName = typer.Option(
        "accuracy",
        "--metric",
        "-m",
        help="Metric to use for ensemble weighting",
    ),
    shots: int = typer.Option(
        16,
        "--shots",
        "-s",
        help="Number of few-shot examples for dataset loading",
    ),
) -> None:
    """
    Re-evaluate ensemble with different temperature settings.
    
    This experiment loads champions from a completed evolution run and
    tests how different temperature values affect ensemble performance.
    
    Example:
        uv run experiments/evaluate_temperature.py \\
            logs/texture_ablation_v1_20260204_085949_history.jsonl \\
            --temperatures 1.0 0.5 0.1 0.05 \\
            --metric accuracy
    """
    logger.info("=" * 80)
    logger.info("ENSEMBLE TEMPERATURE SWEEP EXPERIMENT")
    logger.info("=" * 80)

    # Step 1: Load evolutionary history
    logger.info(f"Loading evolutionary history from {history_file}")
    history = EvolutionHistory(history_file)

    # Step 2: Extract champions
    logger.info("Extracting champions from all islands...")
    champion_records = history.get_champions()

    if not champion_records:
        logger.error("No champions found in history! Cannot proceed.")
        raise typer.Exit(code=1)

    logger.success(f"Found {len(champion_records)} champions")

    # Step 3: Reconstruct Individuals
    logger.info("Reconstructing Individual objects from champions...")
    individuals = reconstruct_individuals_from_champions(
        champion_records, metric=metric
    )

    if not individuals:
        logger.error("Could not reconstruct any valid individuals!")
        raise typer.Exit(code=1)

    # Step 4: Load test dataset (need config for this)
    logger.info(f"Loading dataset from task config: {task_config}")

    # Build a minimal config for dataset loading
    # We need evolution and llm configs but they won't be used, so create minimal ones
    import yaml

    from biomedxpro.engine.config import ExecutionConfig

    with open(task_config, "r") as f:
        task_data = yaml.safe_load(f)

    with open(exec_config, "r") as f:
        exec_data = yaml.safe_load(f)

    # Inject shots
    task_data.setdefault("dataset", {})
    task_data["dataset"]["shots"] = shots

    # Build minimal config
    from biomedxpro.core.domain import EvolutionParams, MetricName, TaskDefinition
    from biomedxpro.impl.config import DatasetConfig

    config = MasterConfig(
        task=TaskDefinition.from_dict(task_data["task"]),
        dataset=DatasetConfig.from_dict(task_data["dataset"]),
        evolution=EvolutionParams(
            initial_pop_size=10,
            generations=1,
            target_metric=metric,
        ),
        llm=None,  # type: ignore[arg-type]  # Not needed for evaluation
        execution=ExecutionConfig.from_dict(exec_data.get("execution", {})),
    )

    logger.info("Loading encoded datasets...")
    train_ds, val_ds, test_ds = builder.load_datasets(config)

    # Step 5: Build evaluator
    logger.info("Building fitness evaluator...")
    from biomedxpro.impl.evaluator import FitnessEvaluator

    evaluator = FitnessEvaluator(
        device=config.execution.device,
        batch_size=config.execution.batch_size,
    )

    # Step 6: Temperature sweep
    logger.info("=" * 80)
    logger.info("BEGINNING TEMPERATURE SWEEP")
    logger.info("=" * 80)

    results: list[TemperatureResult] = []

    for temp in temperatures:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing Temperature = {temp}")
        logger.info(f"{'=' * 60}")

        ensemble = PromptEnsemble.from_individuals(
            individuals=individuals,
            metric=metric,
            temperature=temp,
        )

        # Log weights distribution
        logger.info(f"Ensemble weights (temp={temp}):")
        for ind, weight in zip(individuals, ensemble.weights):
            logger.info(
                f"  {ind.concept:20s}: weight={weight.item():.6f}, "
                f"{metric}={ind.get_fitness(metric):.4f}"
            )

        # Evaluate
        logger.info(
            f"Evaluating ensemble on test set ({test_ds.num_samples} samples)..."
        )
        metrics = evaluator.evaluate_ensemble(ensemble, test_ds)

        # Store results
        metric_value = float(metrics.get(metric, 0.0))
        result: TemperatureResult = {
            "temperature": temp,
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_macro"],
            "precision": metrics.get("precision", 0.0),
            "recall": metrics.get("recall", 0.0),
            "metric_value": metric_value,
            "metrics": metrics,
        }
        results.append(result)

        # Print immediate results
        logger.success(f"Results for T={temp}:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score:  {metrics['f1_macro']:.4f}")
        logger.info(f"  AUC:       {metrics.get('auc', 0.0):.4f}")
        logger.info(f"  {metric}:    {metric_value:.4f}")

        # Show confusion matrix
        logger.info("  Confusion Matrix:")
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            for i, row in enumerate(cm):
                row_str = "    " + " ".join(f"{val:5d}" for val in row)
                logger.info(f"    Class {i}: {row_str}")

    # Step 7: Generate comparison table
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON TABLE")
    logger.info("=" * 80 + "\n")

    table = Table(
        title="Temperature Sweep Results", show_header=True, header_style="bold magenta"
    )
    table.add_column("Temperature", style="cyan", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("F1 Score", justify="center")
    table.add_column(f"{metric}", justify="center")
    table.add_column("Δ Acc vs T=1.0", justify="center")

    baseline_acc = next(r["accuracy"] for r in results if r["temperature"] == 1.0)

    for result in results:
        temp = float(result["temperature"])
        acc = float(result["accuracy"])
        f1 = float(result["f1_score"])
        metric_val = float(result["metric_value"])
        delta = acc - baseline_acc

        delta_str = f"{delta:+.4f}" if temp != 1.0 else "baseline"
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "white"

        table.add_row(
            f"{temp:.2f}",
            f"{acc:.4f}",
            f"{f1:.4f}",
            f"{metric_val:.4f}",
            f"[{delta_style}]{delta_str}[/{delta_style}]",
        )

    console.print(table)

    # Step 8: Print best configuration
    best_result = max(results, key=lambda r: float(r["accuracy"]))
    logger.info("\n" + "=" * 80)
    logger.success(f"🏆 BEST CONFIGURATION: Temperature = {best_result['temperature']}")
    logger.success(f"   Accuracy: {best_result['accuracy']:.4f}")
    logger.success(f"   F1 Score: {best_result['f1_score']:.4f}")
    logger.info("=" * 80)

    # Step 9: Hypothesis validation
    logger.info("\n" + "=" * 80)
    logger.info("HYPOTHESIS VALIDATION")
    logger.info("=" * 80)

    if float(best_result["temperature"]) < 1.0:
        improvement = float(best_result["accuracy"]) - baseline_acc
        logger.success(
            f"✅ HYPOTHESIS CONFIRMED: Lowering temperature from 1.0 to "
            f"{best_result['temperature']} improved accuracy by {improvement:+.4f}"
        )
        logger.success(
            "   This validates that T=1.0 was averaging high-quality specialists "
            "with low-performing prompts."
        )
    else:
        logger.warning(
            "⚠️  HYPOTHESIS NOT CONFIRMED: T=1.0 remains the best setting. "
            "The ensemble underperformance may have a different root cause."
        )


if __name__ == "__main__":
    app()
