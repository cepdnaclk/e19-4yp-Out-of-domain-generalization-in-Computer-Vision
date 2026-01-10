# src/biomedxpro/engine/orchestrator.py
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Sequence

from loguru import logger

from biomedxpro.core.domain import (
    EncodedDataset,
    EvolutionParams,
    Individual,
    Population,
)
from biomedxpro.core.interfaces import (
    IFitnessEvaluator,
    IHistoryRecorder,
    IOperator,
    SelectionStrategy,
)
from biomedxpro.engine.config import ExecutionConfig


class Orchestrator:
    """
    The central engine managing the evolutionary lifecycle.

    This class coordinates the parallel execution of evolutionary operators across
    multiple islands (populations), manages the global fitness evaluation queue,
    and synchronizes state updates. It enforces isolation between the 'Thinking'
    (LLM generation) and 'Judging' (GPU evaluation) stages to maximize resource
    throughput.
    """

    def __init__(
        self,
        evaluator: IFitnessEvaluator,
        operator: IOperator,
        selector: SelectionStrategy,
        train_dataset: EncodedDataset,
        val_dataset: EncodedDataset,
        params: EvolutionParams,
        recorder: IHistoryRecorder,
        exec_config: ExecutionConfig,
    ) -> None:
        self.evaluator = evaluator
        self.operator = operator
        self.selector = selector
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.params = params

        # Configuration for hardware and concurrency
        self.exec_config = exec_config
        self.executor = ThreadPoolExecutor(max_workers=self.exec_config.max_workers)

        # The Archipelago: A collection of isolated populations evolving independently
        self.islands: list[Population] = []

        # Persistence layer for tracking lineage and metrics
        self.recorder = recorder

    def _process_futures_and_evaluate(
        self, future_map: dict[Future[Sequence[Individual]], str]
    ) -> dict[str, Sequence[Individual]]:
        """
        Synchronizes parallel tasks and performs vectorized fitness evaluation.

        This method acts as a synchronization barrier, collecting results from
        asynchronous LLM tasks. Once collected, it aggregates all new individuals
        into a single batch for high-throughput GPU evaluation.

        Args:
            future_map: A dictionary mapping pending Futures to their corresponding
                        concept name.

        Returns:
            A dictionary mapping concept names to their evaluated individuals.
        """
        results_map: dict[str, Sequence[Individual]] = {}
        all_individuals_flat: list[Individual] = []

        # Collect results from the thread pool as they complete
        for future in as_completed(future_map):
            concept = future_map[future]
            try:
                individuals = future.result()
                if individuals:
                    results_map[concept] = individuals
                    all_individuals_flat.extend(individuals)
                else:
                    logger.warning(
                        f"Task for island '{concept}' returned no individuals."
                    )
            except Exception as e:
                logger.error(f"Task for island '{concept}' failed: {e}")

        # Execute vectorized evaluation on the GPU if candidates exist
        if all_individuals_flat:
            logger.debug(
                f"Evaluating batch of {len(all_individuals_flat)} individuals..."
            )
            self.evaluator.evaluate(all_individuals_flat, self.train_dataset)
        else:
            logger.warning("No individuals produced across all tasks.")

        return results_map

    def initialize(self, concepts: list[str] | None) -> None:
        """
        Bootstraps the archipelago by creating and populating islands.

        If no concepts are provided, it uses the operator to discover them.
        It then triggers parallel generation of the initial population (Adam & Eve)
        for all islands, evaluates them in a batch, and persists the initial state.
        """
        if concepts is None:
            concepts = self.operator.discover_concepts()
            logger.info(f"Discovered concepts: {concepts}")

        logger.info(f"Initializing archipelago with {len(concepts)} islands...")

        # Dispatch initialization tasks to the thread pool for parallel execution
        future_to_concept = {}
        for concept in concepts:
            future = self.executor.submit(
                self.operator.initialize_population,
                num_offsprings=self.params.initial_pop_size,
                concept=concept,
            )
            future_to_concept[future] = concept

        # Synchronize threads and perform batch GPU evaluation
        results_map = self._process_futures_and_evaluate(future_to_concept)

        # Distribute evaluated individuals back to their respective islands
        for concept in concepts:
            if concept not in results_map:
                continue

            # Instantiate the population container
            island = Population(concept=concept, capacity=self.params.island_capacity)
            island.add_individuals(results_map[concept])

            self.islands.append(island)

            # Log initial population statistics
            stats = island.get_stats(self.params.target_metric)
            logger.bind(phase="init", island=concept).info(
                f"Initialized with {len(island)} individuals. Stats: {stats}"
            )

        self.recorder.record_generation(self.islands)

    def run(self) -> list[Individual]:
        """
        Executes the main evolutionary loop.

        Iterates through the specified number of generations, coordinating selection,
        reproduction, and evaluation. Records the state of the archipelago at each
        generation.
        """
        logger.info("Starting evolutionary run...")

        if not self.islands:
            logger.info("Orchestrator not initialized. Bootstrapping...")
            self.initialize()

        for gen in range(1, self.params.generations + 1):
            self._run_generation(gen)
            self.recorder.record_generation(self.islands)

        logger.info("Evolution complete.")
        return self._collect_best_individuals()

    def _run_generation(self, gen: int) -> None:
        """
        Executes a single evolutionary step across all islands.

        This method follows a Fan-Out/Fan-In pattern:
        1. Select parents sequentially (CPU-bound).
        2. Fan-Out reproduction tasks to threads (Network I/O bound).
        3. Fan-In results and Batch Evaluate on GPU (Compute bound).
        4. Scatter evaluated offspring back to islands for survival selection.
        """
        start_time = time.time()
        gen_logger = logger.bind(generation=gen)
        gen_logger.info(f"=== Starting Generation {gen}/{self.params.generations} ===")

        target_metric = self.params.target_metric

        # Fan-Out: Submit reproduction tasks to the thread pool
        future_to_concept: dict[Future[Sequence[Individual]], str] = {}

        for island in self.islands:
            # Parent selection (CPU-bound operation)
            parents = self.selector.select(
                island, k=self.params.num_parents, metric=target_metric
            )
            if not parents:
                gen_logger.warning(f"Island '{island.concept}' skipped (no parents).")
                continue

            # Submit reproduction task (Network I/O bound operation)
            future = self.executor.submit(
                self.operator.reproduce,
                parents=parents,
                concept=island.concept,
                num_offsprings=self.params.offspring_per_gen,
                target_metric=target_metric,
                current_generation=gen,
            )
            # Map Future to Concept for tracking results
            future_to_concept[future] = island.concept

        # Barrier & Evaluation: Synchronize threads and batch evaluate
        results_map = self._process_futures_and_evaluate(future_to_concept)

        if not results_map:
            return

        # Scatter: Distribute offspring and perform survival selection
        for island in self.islands:
            if island.concept not in results_map:
                continue

            # Integrate new offspring into the population
            island.add_individuals(results_map[island.concept])
            island.keep_elites(target_metric)
            island.increment_generation()

            # Log generation statistics
            stats = island.get_stats(target_metric)
            gen_logger.bind(island=island.concept).info(f"Stats: {stats}")

        elapsed = time.time() - start_time
        gen_logger.info(f"Generation {gen} complete in {elapsed:.2f}s")

    def _collect_best_individuals(self) -> list[Individual]:
        """
        Harvests the top-performing individual from each island.

        This forms the final ensemble of expert prompts discovered during the
        evolutionary process.
        """
        champions: list[Individual] = []
        target = self.params.target_metric

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
