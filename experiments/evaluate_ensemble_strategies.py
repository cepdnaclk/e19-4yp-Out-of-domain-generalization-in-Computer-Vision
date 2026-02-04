#!/usr/bin/env python3
"""
Ensemble Strategy Comparison Experiment.

This script validates different ensemble aggregation strategies:
- Linear Opinion Pool (LOP): Standard arithmetic mean (current approach)
- Logarithmic Opinion Pool (LogOP): Geometric mean with veto power
- Entropy-Dynamic Weighting: Confidence-aware dynamic weighting

The hypothesis is that logarithmic pooling will outperform linear pooling
by enforcing consensus and penalizing disagreement.

Usage:
    uv run experiments/evaluate_ensemble_strategies.py logs/your_history.jsonl
    uv run experiments/evaluate_ensemble_strategies.py logs/your_history.jsonl --temperature 0.1
    uv run experiments/evaluate_ensemble_strategies.py logs/your_history.jsonl --metric f1_macro
"""

from pathlib import Path
from typing import Literal, TypedDict

import torch
import torch.nn.functional as F
import typer
import yaml
from loguru import logger
from rich.console import Console
from rich.table import Table

from biomedxpro.analysis import EvolutionHistory, HistoryRecord
from biomedxpro.core.domain import (
    CreationOperation,
    EvaluationMetrics,
    Individual,
    PromptEnsemble,
    PromptGenotype,
)
from biomedxpro.engine import builder
from biomedxpro.engine.config import ExecutionConfig, MasterConfig

app = typer.Typer(help="Ensemble Strategy Evaluation Experiment")
console = Console()

EnsembleStrategy = Literal["linear", "logarithmic", "entropy_dynamic"]


class StrategyResult(TypedDict):
    """Type definition for strategy evaluation results."""

    strategy: EnsembleStrategy
    accuracy: float
    f1_macro: float
    margin_score: float
    auc: float
    metrics: EvaluationMetrics


class AdvancedEnsemble:
    """
    Enhanced ensemble with multiple aggregation strategies.
    Does NOT modify the domain.py file - this is experiment-specific.
    """

    def __init__(
        self,
        experts: list[Individual],
        weights: torch.Tensor,
        strategy: EnsembleStrategy = "linear",
    ):
        self.experts = experts
        self.weights = weights
        self.strategy = strategy

    @property
    def prompts(self) -> list[list[str]]:
        return [list(ind.genotype.prompts) for ind in self.experts]

    def apply(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Dispatch to the correct aggregation strategy.

        Args:
            expert_probs: (N_Samples, N_Experts, N_Classes)

        Returns:
            (N_Samples, N_Classes) final probability distribution
        """
        if self.strategy == "linear":
            return self._apply_linear(expert_probs)
        elif self.strategy == "logarithmic":
            return self._apply_logarithmic(expert_probs)
        elif self.strategy == "entropy_dynamic":
            return self._apply_entropy_dynamic(expert_probs)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _apply_linear(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Standard Linear Opinion Pool (LOP) - Arithmetic Mean.

        Good for: Noise reduction, stable predictions
        Bad for: Sharp decision boundaries, minority expert signals

        Math: P_final = Sum(w_i * P_i)
        """
        w = self.weights.to(expert_probs.device).view(1, -1, 1)
        weighted = expert_probs * w
        return weighted.sum(dim=1)

    def _apply_logarithmic(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic Opinion Pool (LogOP) - Geometric Mean.

        Good for: Consensus enforcement, veto power
        Bad for: Can be overly conservative if one expert is wrong

        Math: P_final = Softmax(Sum(w_i * Log(P_i)))

        Why: If one expert says prob=0, the ensemble respects that veto.
        All experts must agree for high confidence.
        """
        # Numerical stability: avoid log(0)
        eps = 1e-9
        log_probs = torch.log(expert_probs + eps)

        w = self.weights.to(expert_probs.device).view(1, -1, 1)

        # Weighted sum of LOG probabilities
        weighted_log_sum = (log_probs * w).sum(dim=1)

        # Re-normalize back to probability distribution
        return F.softmax(weighted_log_sum, dim=1)

    def _apply_entropy_dynamic(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Dynamic Uncertainty-Aware Weighting.

        Good for: Adapting to per-sample expert confidence
        Bad for: More complex, can be unstable

        Logic: Combine global fitness (static weight) with local confidence (entropy).
        If an expert is unsure (high entropy) for THIS specific sample,
        their vote counts for less.

        Math:
            H_i = -Sum(p * log(p))  # Entropy per expert
            confidence_i = exp(-H_i)  # Low entropy → High confidence
            w_dynamic = w_global * confidence
            w_normalized = w_dynamic / Sum(w_dynamic)
        """
        # 1. Calculate Entropy per expert per sample
        # H = -Sum(p * log(p))
        # Shape: (N_Samples, N_Experts)
        eps = 1e-9
        entropy = -torch.sum(expert_probs * torch.log(expert_probs + eps), dim=2)

        # 2. Invert Entropy (Low entropy = High confidence)
        confidence = torch.exp(-entropy)

        # 3. Combine with global static weights
        # Shape: (N_Samples, N_Experts)
        global_w = self.weights.to(expert_probs.device).view(1, -1)

        # The final weight is: Global_Quality * Local_Confidence
        combined_weights = global_w * confidence

        # 4. Normalize weights per sample so they sum to 1
        # Shape: (N_Samples, N_Experts, 1)
        norm_weights = (
            combined_weights / combined_weights.sum(dim=1, keepdim=True)
        ).unsqueeze(2)

        # 5. Apply Weighted Sum
        return (expert_probs * norm_weights).sum(dim=1)


def reconstruct_individuals_from_champions(
    champions: dict[str, HistoryRecord],
) -> list[Individual]:
    """
    Reconstruct Individual domain objects from history records.

    Args:
        champions: Dictionary mapping concept -> champion HistoryRecord

    Returns:
        List of Individual objects with metrics loaded
    """
    individuals = []

    for concept, record in champions.items():
        if record.metrics is None:
            logger.warning(f"Champion for '{concept}' has no metrics, skipping")
            continue

        # Reconstruct Individual
        genotype = PromptGenotype(prompts=tuple(record.prompts))
        individual = Individual(
            id=record.id,
            genotype=genotype,
            concept=concept,
            generation_born=record.generation_born,
            operation=CreationOperation.LLM_MUTATION,
        )

        # Load metrics
        metrics: EvaluationMetrics = record.metrics  # type: ignore[assignment]
        individual.update_metrics(metrics)

        individuals.append(individual)
        logger.info(
            f"✓ Loaded champion '{concept}': "
            f"margin_score={metrics['margin_score']:.4f}, "
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
    temperature: float = typer.Option(
        0.1,
        "--temperature",
        "-T",
        help="Temperature for softmax weighting (lower = more meritocratic)",
    ),
    metric: str = typer.Option(
        "margin_score",
        "--metric",
        "-m",
        help="Metric to use for ensemble weighting",
    ),
) -> None:
    """
    Compare ensemble strategies with different aggregation methods.

    This experiment loads champions from a completed evolution run and
    tests Linear, Logarithmic, and Entropy-Dynamic ensemble strategies.

    Example:
        uv run experiments/evaluate_ensemble_strategies.py \\
            logs/texture_ablation_v1_20260204_085949_history.jsonl \\
            --temperature 0.1 \\
            --metric margin_score
    """
    logger.info("=" * 80)
    logger.info("ENSEMBLE STRATEGY COMPARISON EXPERIMENT")
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
    individuals = reconstruct_individuals_from_champions(champion_records)

    if not individuals:
        logger.error("Could not reconstruct any valid individuals!")
        raise typer.Exit(code=1)

    # Step 4: Load test dataset
    logger.info(f"Loading dataset from task config: {task_config}")

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
            target_metric=metric,  # type: ignore[arg-type]
        ),
        llm=None,  # type: ignore[arg-type]
        execution=ExecutionConfig.from_dict(exec_data.get("execution", {})),
    )

    logger.info("Loading encoded datasets...")
    train_ds, val_ds, test_ds = builder.load_datasets(config)

    # Step 5: Compute global weights
    logger.info(f"Computing ensemble weights with temperature={temperature}...")
    scores = torch.tensor(
        [ind.get_fitness(metric) for ind in individuals],  # type: ignore[arg-type]
        dtype=torch.float32,
    )
    weights = torch.softmax(scores / temperature, dim=0)

    logger.info("Global fitness weights:")
    for ind, weight in zip(individuals, weights):
        logger.info(
            f"  {ind.concept:20s}: weight={weight.item():.6f}, "
            f"margin_score={ind.get_fitness('margin_score'):.4f}"
        )

    # Step 6: Build evaluator
    logger.info("Building fitness evaluator...")
    from biomedxpro.impl.evaluator import FitnessEvaluator

    evaluator = FitnessEvaluator(
        device=config.execution.device,
        batch_size=config.execution.batch_size,
    )

    # Step 7: Strategy comparison
    logger.info("\n" + "=" * 80)
    logger.info("BEGINNING STRATEGY COMPARISON")
    logger.info("=" * 80)

    strategies: list[EnsembleStrategy] = ["linear", "logarithmic", "entropy_dynamic"]
    results: list[StrategyResult] = []

    for strategy in strategies:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing Strategy: {strategy.upper()}")
        logger.info(f"{'=' * 60}")

        # Create advanced ensemble with this strategy
        ensemble = AdvancedEnsemble(
            experts=individuals,
            weights=weights,
            strategy=strategy,
        )

        # Compute raw probabilities from all experts
        logger.info(
            f"Computing expert opinions on test set ({test_ds.num_samples} samples)..."
        )
        raw_probs = evaluator.compute_batch_probabilities(ensemble.prompts, test_ds)

        # Apply strategy-specific aggregation
        logger.info(f"Aggregating with {strategy} strategy...")
        final_probs = ensemble.apply(raw_probs)

        # Calculate metrics
        from biomedxpro.utils.metrics import calculate_classification_metrics

        y_prob_dist = final_probs.cpu().numpy()
        y_true = test_ds.labels.cpu().numpy()
        y_pred = y_prob_dist.argmax(axis=1)

        metrics = calculate_classification_metrics(y_true, y_pred, y_prob_dist)

        # Store results
        results.append(
            {
                "strategy": strategy,
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
                "margin_score": metrics["margin_score"],
                "auc": metrics["auc"],
                "metrics": metrics,
            }
        )

        # Print immediate results
        logger.success(f"Results for {strategy.upper()}:")
        logger.info(f"  Accuracy:     {metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score:     {metrics['f1_macro']:.4f}")
        logger.info(f"  Margin Score: {metrics['margin_score']:.4f}")
        logger.info(f"  AUROC:        {metrics['auc']:.4f}")

        # Show confusion matrix
        cm = metrics.get("confusion_matrix")
        if cm is not None:
            logger.info("  Confusion Matrix:")
            for i, row in enumerate(cm):
                row_str = "    " + " ".join(f"{val:5d}" for val in row)
                logger.info(f"    Class {i}: {row_str}")

    # Step 8: Generate comparison table
    logger.info("\n" + "=" * 80)
    logger.info("FINAL COMPARISON TABLE")
    logger.info("=" * 80 + "\n")

    table = Table(
        title="Ensemble Strategy Comparison",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Strategy", style="cyan", justify="left")
    table.add_column("Accuracy", justify="center")
    table.add_column("F1 Score", justify="center")
    table.add_column("Margin Score", justify="center")
    table.add_column("AUROC", justify="center")
    table.add_column("Δ Acc vs Linear", justify="center")

    baseline_acc = next(r["accuracy"] for r in results if r["strategy"] == "linear")

    for result in results:
        strategy = result["strategy"]
        acc = result["accuracy"]
        f1 = result["f1_macro"]
        margin = result["margin_score"]
        auc = result["auc"]
        delta = acc - baseline_acc

        delta_str = f"{delta:+.4f}" if strategy != "linear" else "baseline"
        delta_style = "green" if delta > 0 else "red" if delta < 0 else "white"

        table.add_row(
            strategy.upper(),
            f"{acc:.4f}",
            f"{f1:.4f}",
            f"{margin:.4f}",
            f"{auc:.4f}",
            f"[{delta_style}]{delta_str}[/{delta_style}]",
        )

    console.print(table)

    # Step 9: Identify best strategy
    best_result = max(results, key=lambda r: r["accuracy"])
    logger.info("\n" + "=" * 80)
    logger.success(f"🏆 BEST STRATEGY: {best_result['strategy'].upper()}")
    logger.success(f"   Accuracy:     {best_result['accuracy']:.4f}")
    logger.success(f"   F1 Score:     {best_result['f1_macro']:.4f}")
    logger.success(f"   Margin Score: {best_result['margin_score']:.4f}")
    logger.info("=" * 80)

    # Step 10: Hypothesis validation
    logger.info("\n" + "=" * 80)
    logger.info("HYPOTHESIS VALIDATION")
    logger.info("=" * 80)

    if best_result["strategy"] == "logarithmic":
        improvement = best_result["accuracy"] - baseline_acc
        logger.success(
            f"✅ HYPOTHESIS CONFIRMED: Logarithmic pooling improved accuracy by {improvement:+.4f}"
        )
        logger.success(
            "   This validates that geometric mean provides better consensus enforcement "
            "and veto power for medical imaging tasks."
        )
    elif best_result["strategy"] == "entropy_dynamic":
        improvement = best_result["accuracy"] - baseline_acc
        logger.success(
            f"✅ DYNAMIC WEIGHTING WINS: Entropy-aware strategy improved accuracy by {improvement:+.4f}"
        )
        logger.success(
            "   This suggests that per-sample confidence adaptation is valuable "
            "for handling varying expert certainty."
        )
    else:
        logger.warning(
            "⚠️  LINEAR POOLING REMAINS BEST: Advanced strategies did not improve performance."
        )
        logger.info(
            "   Possible reasons: (1) Experts already well-calibrated, "
            "(2) Task doesn't benefit from veto power, (3) Need more diverse experts."
        )

    logger.info("=" * 80)


if __name__ == "__main__":
    app()
