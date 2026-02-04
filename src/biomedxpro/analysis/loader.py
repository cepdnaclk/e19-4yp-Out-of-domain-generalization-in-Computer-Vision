"""
Loader for parsing evolutionary history from JSONL logs.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


@dataclass
class HistoryRecord:
    """
    Single individual record from history log.
    Represents one individual at one point in time.
    """

    generation: int
    concept: str
    id: str
    parents: list[str]
    prompts: list[str]
    metrics: Optional[dict[str, float]]
    metadata: dict[str, str]
    generation_born: int

    @property
    def fitness(self) -> Optional[float]:
        """Primary fitness value (inverted_bce)."""
        return self.metrics.get("inverted_bce") if self.metrics else None

    @property
    def accuracy(self) -> Optional[float]:
        """Accuracy metric."""
        return self.metrics.get("accuracy") if self.metrics else None

    @property
    def auc(self) -> Optional[float]:
        """AUC metric."""
        return self.metrics.get("auc") if self.metrics else None

    @property
    def f1_macro(self) -> Optional[float]:
        """F1 macro metric."""
        return self.metrics.get("f1_macro") if self.metrics else None


@dataclass
class IslandSnapshot:
    """
    All individuals from a single island at a specific generation.
    """

    concept: str
    generation: int
    individuals: list[HistoryRecord] = field(default_factory=list)

    @property
    def fitness_values(self) -> list[float]:
        """Extract all valid fitness values."""
        return [ind.fitness for ind in self.individuals if ind.fitness is not None]

    @property
    def max_fitness(self) -> Optional[float]:
        """Best fitness in this snapshot."""
        values = self.fitness_values
        return max(values) if values else None

    @property
    def mean_fitness(self) -> Optional[float]:
        """Average fitness in this snapshot."""
        values = self.fitness_values
        return sum(values) / len(values) if values else None

    @property
    def min_fitness(self) -> Optional[float]:
        """Worst fitness in this snapshot."""
        values = self.fitness_values
        return min(values) if values else None

    @property
    def champion(self) -> Optional[HistoryRecord]:
        """Best individual in this snapshot."""
        if not self.individuals:
            return None
        valid_individuals = [ind for ind in self.individuals if ind.fitness is not None]
        if not valid_individuals:
            return None
        return max(valid_individuals, key=lambda x: x.fitness)  # type: ignore[return-value, arg-type]


class EvolutionHistory:
    """
    Complete evolutionary history loaded from JSONL log file.
    Provides indexing and querying capabilities for analysis.
    """

    def __init__(self, log_path: str | Path) -> None:
        """
        Load and parse evolutionary history from JSONL file.

        Args:
            log_path: Path to *_history.jsonl file
        """
        self.log_path = Path(log_path)
        if not self.log_path.exists():
            raise FileNotFoundError(f"History log not found: {self.log_path}")

        logger.info(f"Loading evolutionary history from {self.log_path}")

        # Raw records storage
        self.records: list[HistoryRecord] = []

        # Indexed access structures
        self._by_generation: dict[int, list[HistoryRecord]] = {}
        self._by_island: dict[str, list[HistoryRecord]] = {}
        self._by_id: dict[str, list[HistoryRecord]] = {}
        self._islands: set[str] = set()
        self._max_generation: int = 0

        self._load()

    def _load(self) -> None:
        """Parse JSONL file and build indices."""
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    record = HistoryRecord(
                        generation=data["generation"],
                        concept=data["concept"],
                        id=data["id"],
                        parents=data["parents"],
                        prompts=data["prompts"],
                        metrics=data.get("metrics"),
                        metadata=data.get("metadata", {}),
                        generation_born=data["generation_born"],
                    )

                    self.records.append(record)

                    # Build indices
                    self._by_generation.setdefault(record.generation, []).append(record)
                    self._by_island.setdefault(record.concept, []).append(record)
                    self._by_id.setdefault(record.id, []).append(record)
                    self._islands.add(record.concept)
                    self._max_generation = max(self._max_generation, record.generation)

                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping malformed line {line_num}: {e}")

        logger.success(
            f"Loaded {len(self.records)} records across {self._max_generation + 1} generations"
        )
        logger.info(f"Islands: {', '.join(sorted(self._islands))}")

    @property
    def islands(self) -> list[str]:
        """List of all island concepts."""
        return sorted(self._islands)

    @property
    def num_generations(self) -> int:
        """Total number of generations."""
        return self._max_generation + 1

    @property
    def num_islands(self) -> int:
        """Total number of islands."""
        return len(self._islands)

    def get_generation(self, gen: int) -> list[HistoryRecord]:
        """Get all individuals from a specific generation."""
        return self._by_generation.get(gen, [])

    def get_island(self, concept: str) -> list[HistoryRecord]:
        """Get all records from a specific island (across all generations)."""
        return self._by_island.get(concept, [])

    def get_island_at_generation(self, concept: str, gen: int) -> IslandSnapshot:
        """Get snapshot of a specific island at a specific generation."""
        individuals = [
            rec for rec in self._by_generation.get(gen, []) if rec.concept == concept
        ]
        return IslandSnapshot(concept=concept, generation=gen, individuals=individuals)

    def get_individual_history(self, individual_id: str) -> list[HistoryRecord]:
        """
        Get all appearances of an individual across generations.
        (Same ID may appear in multiple generations due to survival/elitism)
        """
        return self._by_id.get(individual_id, [])

    def get_champions(self) -> dict[str, HistoryRecord]:
        """
        Get the best individual from each island across ALL generations.

        CRITICAL: We find the absolute best individual that ever existed on each island,
        not just the final generation (which might not have the champion due to
        population dynamics, extinction, or recording issues).

        Returns:
            Dictionary mapping island concept -> best individual ever
        """
        champions = {}
        for island in self.islands:
            island_records = self.get_island(island)

            # Filter to only records with valid fitness
            valid_records = [rec for rec in island_records if rec.fitness is not None]

            if not valid_records:
                logger.warning(
                    f"No individuals with fitness found for island '{island}' "
                    f"across all {self.num_generations} generations!"
                )
                continue

            # Find the absolute best
            champion = max(valid_records, key=lambda rec: rec.fitness)  # type: ignore[arg-type, return-value]
            champions[island] = champion

        return champions

    def get_lineage(self, individual_id: str) -> list[str]:
        """
        Trace ancestry back to generation 0.

        Returns all unique ancestors (parents, grandparents, great-grandparents, etc.).
        With crossover, an individual can have 2+ parents per generation, so the total
        ancestor count can be much larger than the number of generations.

        Args:
            individual_id: ID of individual to trace

        Returns:
            List of unique ancestor IDs
        """
        # Get the record (take first appearance)
        records = self.get_individual_history(individual_id)
        if not records:
            return []

        record = records[0]

        # Use BFS to collect all ancestors without duplicates
        ancestors = set()
        to_visit = list(record.parents)

        while to_visit:
            parent_id = to_visit.pop(0)
            if parent_id in ancestors:
                continue

            ancestors.add(parent_id)

            # Get this parent's parents
            parent_records = self.get_individual_history(parent_id)
            if parent_records:
                to_visit.extend(parent_records[0].parents)

        return list(ancestors)

    def iter_island_snapshots(self, concept: str) -> list[IslandSnapshot]:
        """
        Get chronological snapshots of an island across all generations.

        Args:
            concept: Island name

        Returns:
            List of IslandSnapshot objects, one per generation
        """
        return [
            self.get_island_at_generation(concept, gen)
            for gen in range(self.num_generations)
        ]

    def __repr__(self) -> str:
        return (
            f"EvolutionHistory(log={self.log_path.name}, "
            f"generations={self.num_generations}, "
            f"islands={self.num_islands}, "
            f"records={len(self.records)})"
        )
