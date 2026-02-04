from loguru import logger

from biomedxpro.core.domain import EvaluationMetrics, Individual, MetricName


def print_champion_summary(champions: list[Individual], metric: MetricName) -> None:
    """
    Prints a formatted summary of the discovered experts.
    """
    print("\n" + "=" * 60)
    logger.success(
        f"Evolution complete. Discovered {len(champions)} expert concept prompts."
    )
    print("=" * 60)

    for ind in champions:
        score = ind.get_fitness(metric)
        logger.info(f"Expert Concept: {ind.concept}")
        # Assuming metric is a valid key in the metrics dict, otherwise might crash if key not found
        # But Indivudal.get_fitness handles it.
        logger.info(f"    Validation Score ({metric}): {score:.4f}")
        logger.info("    Prompts:")
        for i, prompt in enumerate(ind.genotype.prompts):
            logger.info(f"        Class {i}: {prompt}")
        print("-" * 60)


def print_ensemble_results(metrics: EvaluationMetrics) -> None:
    """
    Prints the final report card for the deployed model.
    """
    print("\n" + "=" * 60)
    logger.success("FINAL ENSEMBLE RESULTS (TEST SET)")
    print("=" * 60)

    # Use direct keys using the TypedDict
    logger.success(f"Ensemble Accuracy:   {metrics['accuracy']:.4f}")
    logger.success(f"Ensemble F1 (Macro): {metrics['f1_macro']:.4f}")
    logger.success(f"Ensemble AUROC:      {metrics['auc']:.4f}")

    if "confusion_matrix" in metrics:
        print("-" * 60)
        logger.info(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
