# src/biomedxpro/impl/selection.py
import random
from typing import Sequence

from loguru import logger

from biomedxpro.core.domain import Individual, MetricName, Population
from biomedxpro.core.interfaces import SelectionStrategy


class RouletteWheelSelector(SelectionStrategy):
    """
    Selects individuals with probability proportional to their fitness.
    This maintains diversity by giving lower-performing individuals a chance
    to be selected as parents.
    """

    def select(
        self, population: Population, k: int, metric: MetricName
    ) -> Sequence[Individual]:
        """
        Performs roulette wheel selection for k parents.
        """
        # 1. Filter for evaluated individuals
        candidates = [ind for ind in population.individuals if ind.is_evaluated]
        if not candidates:
            logger.warning("No evaluated individuals available for selection.")
            return []

        # 2. Extract fitness scores
        scores = [ind.get_fitness(metric) for ind in candidates]
        total_fitness = sum(scores)

        # 3. Handle edge cases (e.g., all scores zero)
        if total_fitness <= 0:
            # Fall back to uniform random selection if no fitness signal exists
            logger.warning(
                "Total fitness is zero or negative; using uniform random selection."
            )
            return random.choices(candidates, k=k)

        # 4. Stochastic selection using fitness as weights
        # random.choices implements the roulette wheel logic internally
        return random.choices(candidates, weights=scores, k=k)


class ElitismSelector(SelectionStrategy):
    """
    Always selects the top-k best performing individuals.
    This leads to faster convergence (exploitation) but higher risk of mode collapse.
    """

    def select(
        self, population: Population, k: int, metric: MetricName
    ) -> Sequence[Individual]:
        """
        Returns the top-k individuals from the population.
        """
        # Population.get_elite_individuals already implements sorted top-k logic
        return population.get_elite_individuals(k=k, metric=metric)


class RankSelector(SelectionStrategy):
    """
    Selects individuals based on their rank rather than raw fitness.
    - Solves the 'Negative Fitness' problem (Margin Score).
    - Prevents 'Super Individuals' from dominating (Roulette problem).
    - Maintains constant selection pressure.
    """

    def select(
        self, population: Population, k: int, metric: MetricName
    ) -> Sequence[Individual]:
        # 1. Filter valid candidates
        candidates = [ind for ind in population.individuals if ind.is_evaluated]
        if not candidates:
            logger.warning("No evaluated individuals available for selection.")
            return []

        # 2. Sort by Fitness (Ascending: Worst -> Best)
        # We sort ascending so the index corresponds to 'weight'
        # (index 0 = worst = weight 1, index N = best = weight N+1)
        sorted_candidates = sorted(candidates, key=lambda ind: ind.get_fitness(metric))

        n = len(sorted_candidates)

        # 3. Assign Rank Weights (Linear Ranking)
        # Simplest approach: Weight = Rank Index (1 to N)
        # Worst gets weight 1, Best gets weight N.
        # This ensures positive probabilities regardless of negative fitness.
        rank_weights = [i + 1 for i in range(n)]

        # 4. Select k parents
        # random.choices uses the weights to build the cumulative distribution
        return random.choices(sorted_candidates, weights=rank_weights, k=k)
