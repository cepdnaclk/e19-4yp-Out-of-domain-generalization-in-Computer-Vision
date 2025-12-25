# src/biomedxpro/engine/orchestrator.py
import time
from typing import Sequence

from loguru import logger

from biomedxpro.core.domain import (
    EncodedDataset,
    EvolutionConfig,
    Individual,
    Population,
)
from biomedxpro.core.interfaces import (
    IFitnessEvaluator,
    IOperator,
    SelectionStrategy,
)


class Orchestrator:
    """
    The Conductor of the Evolutionary Symphony.
    It manages the islands, schedules the operators, and handles the main loop.
    """

    def __init__(
        self,
        evaluator: IFitnessEvaluator,
        operator: IOperator,
        selector: SelectionStrategy,
        train_dataset: EncodedDataset,
        val_dataset: EncodedDataset,
        config: EvolutionConfig,
    ) -> None:
        self.evaluator = evaluator
        self.operator = operator
        self.selector = selector
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # The Archipelago: A list of isolated populations
        self.islands: list[Population] = []

    def initialize(self, concepts: list[str] | None) -> None:
        """
        Phase 1: Genesis.
        Creates an island for each medical concept and populates it.
        """

        # Discover concepts if not provided
        if concepts is None:
            concepts = self.operator.discover_concepts()
            logger.info(f"Discovered concepts: {concepts}")

        logger.info(f"Initializing archipelago with {len(concepts)} islands.")
        for concept in concepts:
            # Create a logging context for this specific operation
            island_logger = logger.bind(phase="init", island=concept)
            island_logger.info("Creating island and generating initial population...")

            # 1. Create the Island Container
            island = Population(concept=concept, capacity=self.config.island_capacity)

            # 2. Generate Adam & Eve (Initial Prompts) via LLM
            # Note: We pass the concept so the LLM knows what to generate
            initial_individuals: Sequence[Individual] = (
                self.operator.initialize_population(
                    num_offsprings=self.config.initial_pop_size, concept=concept
                )
            )

            # 3. Evaluate them immediately to establish a baseline
            self.evaluator.evaluate(initial_individuals, self.train_dataset)

            # 4. Populate the island
            island.add_individuals(initial_individuals)

            self.islands.append(island)

            # Log initial stats
            stats = island.get_stats(self.config.target_metric)
            logger.info(
                f"Island '{concept}' initialized with {len(island.individuals)} individuals. "
                f"Stats: {stats}"
            )

    def run(self) -> list[Individual]:
        """
        Phase 2: The Evolutionary Loop.
        Executing T generations across all islands.
        """
        logger.info("Starting evolutionary run...")

        for gen in range(1, self.config.generations + 1):
            self._run_generation(gen)

            # Optional: Checkpoint logic could go here
            if self.config.save_checkpoints:
                pass  # Implementation would dump self.islands to disk

        logger.info("Evolution complete.")
        return self._collect_best_individuals()

    def _run_generation(self, gen: int) -> None:
        """
        Executes a single generation step for all islands.
        """
        start_time = time.time()

        # Bind 'generation' to all logs in this scope
        gen_logger = logger.bind(generation=gen)
        gen_logger.info(f"=== Starting Generation {gen}/{self.config.generations} ===")

        target_metric = self.config.target_metric

        for island in self.islands:
            # Bind 'island' to logs so we know where we are
            island_logger = gen_logger.bind(island=island.concept)

            # 1. Parent Selection
            parents = self.selector.select(
                island, k=self.config.num_parents, metric=target_metric
            )

            if not parents:
                island_logger.warning(
                    "Island has no viable parents! Skipping generation."
                )
                continue

            # 2. Reproduction (Directed Mutation)
            # The LLM is instructed to maintain the island's concept
            offspring = self.operator.reproduce(
                parents=parents,
                concept=island.concept,
                num_offsprings=self.config.offspring_per_gen,
            )

            # 3. Evaluation
            # Updates the offspring metrics in-place
            self.evaluator.evaluate(offspring, self.train_dataset)

            # 4. Add to Population
            island.add_individuals(offspring)

            # 5. Survival of the Fittest (Sort & Trim)
            # We sort at the start to ensure we select parents from the best available
            island.keep_elites(target_metric)

            # 6. Logging
            stats = island.get_stats(target_metric)
            elapsed = time.time() - start_time

            island_logger.info(
                f"Generation {gen} complete. Stats: {stats} | Time Elapsed: {elapsed:.2f}s"
            )

    def _collect_best_individuals(self) -> list[Individual]:
        """
        Phase 3: Harvest.
        Gather the champions from every island to form the final ensemble.
        """
        champions: list[Individual] = []
        target = self.config.target_metric

        for island in self.islands:
            elites = island.get_elite_individuals(k=1, metric=target)
            if elites:
                best = elites[0]
                champions.append(best)
                logger.info(
                    f"Champion for '{island.concept}': {best.id} (Score: {best.get_fitness(target):.4f})"
                )
            else:
                logger.warning(f"Island '{island.concept}' produced no individuals.")

        return champions
