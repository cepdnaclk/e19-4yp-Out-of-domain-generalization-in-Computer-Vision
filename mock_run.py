# mock_run.py
from loguru import logger

from src.biomedxpro.core.domain import EvolutionParams, TaskDefinition
from src.biomedxpro.engine.orchestrator import Orchestrator

# Implementation (Using mock implementations for testing)
from src.biomedxpro.impl.mocks import (
    MockEvaluator,
    MockOperator,
    RandomSelector,
    create_dummy_dataset,
)


def main():
    logger.info("Starting Mock Evolution Test...")

    # 3. Define Domain Parameters (Science)
    # Replaces 'EvolutionConfig'
    evo_params = EvolutionParams(
        generations=3,  # Short run for testing
        island_capacity=10,
        initial_pop_size=5,
        num_parents=2,
        offspring_per_gen=2,
        target_metric="inverted_bce",
    )

    # Replaces 'TaskContext' - purely informative for the MockOperator
    task_def = TaskDefinition(
        task_name="Mock Melanoma",
        image_modality="Synthetic Embeddings",
        positive_class="Malignant",
        negative_class="Benign",
        role="Tester",
    )

    # 4. Initialize Mock Components
    # Note: MockOperator is "pure logic" here, so it doesn't need LLMSettings or PromptStrategy
    mock_operator = MockOperator(task_def=task_def)
    mock_evaluator = MockEvaluator()
    mock_selector = RandomSelector()

    # Create fake tensors to prevent crashes
    dummy_data = create_dummy_dataset()

    # 5. Build the Orchestrator
    orchestrator = Orchestrator(
        evaluator=mock_evaluator,
        operator=mock_operator,
        selector=mock_selector,
        train_dataset=dummy_data,
        val_dataset=dummy_data,
        params=evo_params,  # Injecting the Science Parameters
    )

    # 6. Run the Evolutionary Loop
    logger.info("Phase 1: Initialization")
    # We let the mock operator 'discover' its own fixed concepts
    orchestrator.initialize(concepts=None)

    logger.info("Phase 2: Evolution")
    best_individuals = orchestrator.run()

    # 7. Print Results
    print("\n" + "=" * 40)
    logger.success(f"Test Complete. Found {len(best_individuals)} champions.")

    for ind in best_individuals:
        score = ind.get_fitness("inverted_bce")
        print(f"\n🏆 Concept Island: {ind.concept}")
        print(f"   Fitness Score: {score:.4f}")
        print(f"   Genotype: {ind.genotype}")
        print(f"   Source: {ind.metadata.get('source', 'unknown')}")


if __name__ == "__main__":
    main()
