"""
Analysis metrics and calculators for evolutionary runs.
"""

from dataclasses import dataclass
from typing import Optional

from biomedxpro.analysis.loader import EvolutionHistory


@dataclass
class FitnessStats:
    """Statistics for fitness progression."""

    generation: int
    max_fitness: float
    mean_fitness: float
    min_fitness: float
    std_fitness: float
    num_evaluated: int


@dataclass
class ChampionInfo:
    """Information about a champion individual."""

    individual_id: str
    concept: str
    generation_born: int
    final_generation: int
    final_fitness: float
    prompts: list[str]
    lineage: list[str]
    lineage_depth: int
    survival_time: int  # How many generations it survived


@dataclass
class ConvergenceInfo:
    """Convergence detection results."""

    island: str
    converged: bool
    convergence_generation: Optional[int]
    plateau_length: int
    final_fitness: float
    improvement_rate: float  # Average fitness improvement per generation


class FitnessProgressionAnalyzer:
    """
    Analyzes fitness progression across generations and islands.
    """

    def __init__(self, history: EvolutionHistory) -> None:
        self.history = history

    def compute_island_progression(self, concept: str) -> list[FitnessStats]:
        """
        Compute fitness statistics for each generation in an island.

        Args:
            concept: Island name

        Returns:
            List of FitnessStats, one per generation
        """
        stats_list = []
        snapshots = self.history.iter_island_snapshots(concept)

        for snapshot in snapshots:
            values = snapshot.fitness_values
            if not values:
                continue

            # Compute statistics
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            std_val = variance**0.5

            stats_list.append(
                FitnessStats(
                    generation=snapshot.generation,
                    max_fitness=max(values),
                    mean_fitness=mean_val,
                    min_fitness=min(values),
                    std_fitness=std_val,
                    num_evaluated=len(values),
                )
            )

        return stats_list

    def compute_global_progression(self) -> dict[str, list[FitnessStats]]:
        """
        Compute fitness progression for all islands.

        Returns:
            Dictionary mapping island name -> list of FitnessStats
        """
        return {
            island: self.compute_island_progression(island)
            for island in self.history.islands
        }

    def get_best_generation(self, concept: str) -> tuple[int, float]:
        """
        Find the generation with the highest fitness for an island.

        Returns:
            Tuple of (generation_number, max_fitness)
        """
        progression = self.compute_island_progression(concept)
        if not progression:
            return (0, 0.0)

        best = max(progression, key=lambda x: x.max_fitness)
        return (best.generation, best.max_fitness)


class ChampionAnalyzer:
    """
    Analyzes champion individuals: when they were born, their lineage, survival.
    """

    def __init__(self, history: EvolutionHistory) -> None:
        self.history = history

    def get_champion_info(self, individual_id: str) -> Optional[ChampionInfo]:
        """
        Get detailed information about a champion individual.

        Args:
            individual_id: UUID of the individual

        Returns:
            ChampionInfo or None if not found
        """
        appearances = self.history.get_individual_history(individual_id)
        if not appearances:
            return None

        # Get first and last appearance
        first = appearances[0]
        last = appearances[-1]

        # Get lineage (all unique ancestors)
        lineage = self.history.get_lineage(individual_id)

        return ChampionInfo(
            individual_id=individual_id,
            concept=first.concept,
            generation_born=first.generation_born,
            final_generation=last.generation,
            final_fitness=last.fitness or 0.0,
            prompts=last.prompts,
            lineage=lineage,
            lineage_depth=len(lineage),
            survival_time=last.generation - first.generation_born + 1,
        )

    def get_all_champions_info(self) -> dict[str, ChampionInfo]:
        """
        Get info for all final champions (one per island).

        Returns:
            Dictionary mapping island -> ChampionInfo
        """
        champions = self.history.get_champions()
        champion_info = {}

        for island, record in champions.items():
            info = self.get_champion_info(record.id)
            if info:
                champion_info[island] = info

        return champion_info

    def find_earliest_champion(self) -> Optional[ChampionInfo]:
        """
        Find the champion that was born earliest.

        Returns:
            ChampionInfo of the earliest-born champion
        """
        all_champions = self.get_all_champions_info()
        if not all_champions:
            return None

        return min(all_champions.values(), key=lambda x: x.generation_born)

    def find_deepest_lineage(self) -> Optional[ChampionInfo]:
        """
        Find the champion with the deepest ancestry tree.

        Returns:
            ChampionInfo with most ancestors
        """
        all_champions = self.get_all_champions_info()
        if not all_champions:
            return None

        return max(all_champions.values(), key=lambda x: x.lineage_depth)


class ConvergenceDetector:
    """
    Detects when islands have converged (plateaued without improvement).
    """

    def __init__(self, history: EvolutionHistory, plateau_threshold: int = 5) -> None:
        """
        Args:
            history: Evolution history
            plateau_threshold: Number of generations without improvement to consider converged
        """
        self.history = history
        self.plateau_threshold = plateau_threshold

    def detect_convergence(
        self, concept: str, min_improvement: float = 0.001
    ) -> ConvergenceInfo:
        """
        Detect if an island has converged.

        Args:
            concept: Island name
            min_improvement: Minimum fitness improvement to not be considered a plateau

        Returns:
            ConvergenceInfo with convergence status
        """
        snapshots = self.history.iter_island_snapshots(concept)
        if not snapshots:
            return ConvergenceInfo(
                island=concept,
                converged=False,
                convergence_generation=None,
                plateau_length=0,
                final_fitness=0.0,
                improvement_rate=0.0,
            )

        # Track max fitness seen and plateau length
        max_fitness_seen = 0.0
        plateau_start = None
        plateau_length = 0
        convergence_gen = None

        for snapshot in snapshots:
            current_max = snapshot.max_fitness
            if current_max is None:
                continue

            # Check if we improved
            if current_max > max_fitness_seen + min_improvement:
                max_fitness_seen = current_max
                plateau_start = None
                plateau_length = 0
            else:
                # No improvement
                if plateau_start is None:
                    plateau_start = snapshot.generation
                plateau_length = snapshot.generation - plateau_start + 1

                # Check if converged
                if plateau_length >= self.plateau_threshold and convergence_gen is None:
                    convergence_gen = plateau_start

        # Calculate improvement rate
        first_fitness = snapshots[0].max_fitness or 0.0
        final_fitness = snapshots[-1].max_fitness or 0.0
        num_gens = len(snapshots)
        improvement_rate = (
            (final_fitness - first_fitness) / num_gens if num_gens > 0 else 0.0
        )

        return ConvergenceInfo(
            island=concept,
            converged=convergence_gen is not None,
            convergence_generation=convergence_gen,
            plateau_length=plateau_length,
            final_fitness=final_fitness,
            improvement_rate=improvement_rate,
        )

    def detect_all_convergence(self) -> dict[str, ConvergenceInfo]:
        """
        Detect convergence for all islands.

        Returns:
            Dictionary mapping island -> ConvergenceInfo
        """
        return {
            island: self.detect_convergence(island) for island in self.history.islands
        }
