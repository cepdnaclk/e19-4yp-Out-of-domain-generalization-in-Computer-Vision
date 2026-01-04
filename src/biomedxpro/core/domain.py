# src/biomedxpro/core/domain.py
import statistics
import uuid
from dataclasses import dataclass, field, fields
from enum import StrEnum, auto
from typing import Any, Iterator, Literal, NotRequired, Sequence, TypedDict

import torch

# --- Enums & Value Objects ---


class CreationOperation(StrEnum):
    INITIALIZATION = auto()
    LLM_MUTATION = auto()


class DataSplit(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


@dataclass(slots=True, frozen=True)
class StandardSample:
    """
    The standardized representation of a single data sample.
    Adapters return lists of these; the DataLoader consumes them.
    
    This allows adapters to work with any dataset schema while providing
    a unified interface to the data loading pipeline.
    """

    image_path: str
    label: int


@dataclass(slots=True, frozen=True)
class PromptGenotype:
    """
    Immutable DNA.
    Using dataclass ensures accidental mutation is impossible.
    """

    negative_prompt: str
    positive_prompt: str

    def to_dict(self) -> dict[str, str]:
        return {
            "negative_prompt": self.negative_prompt,
            "positive_prompt": self.positive_prompt,
        }


class EvaluationMetrics(TypedDict):
    """The Report Card."""

    inverted_bce: float
    f1_macro: float
    accuracy: float
    auc: float
    f1_weighted: float
    confusion_matrix: NotRequired[list[list[int]]]


MetricName = Literal["inverted_bce", "f1_macro", "accuracy", "auc", "f1_weighted"]


@dataclass(slots=True, frozen=True)
class EvolutionParams:
    """
    The Control Plane configuration.
    Defines all hyperparameters for the evolutionary run.
    """

    # Global Settings
    generations: int = 20
    target_metric: MetricName = "f1_macro"

    # Island Settings
    island_capacity: int = 100  # Max size of an island
    initial_pop_size: int = 50  # How many prompts to start with

    # Evolutionary Operator Settings
    num_parents: int = 10  # How many winners to select
    offspring_per_gen: int = 10  # How many children to create per gen

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "EvolutionParams":
        """
        Helper to load from a raw dictionary (e.g., from YAML),
        filtering out unknown keys to prevent crashes.
        """
        # Get the field names of this dataclass
        valid_keys = {f.name for f in fields(cls)}
        # Filter the input dict
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)


@dataclass(slots=True, frozen=True)
class TaskDefinition:
    """
    Defines the semantic domain of the problem.
    These fields are injected into the Prompt Template.
    """

    task_name: str  # e.g. "Melanoma Classification"
    image_modality: str  # e.g. "Dermoscopy images"
    positive_class: str  # e.g. "Malignant Melanoma"
    negative_class: str  # e.g. "Benign Nevus"

    # NEW: The Persona for the LLM
    role: str  # e.g. "Expert Dermatologist"

    # Optional: Any extra specific instructions for this dataset
    description: str = ""

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TaskDefinition":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)


# --- Core Entities ---


@dataclass(slots=True, frozen=True)
class EncodedDataset:
    name: str
    features: torch.Tensor
    labels: torch.Tensor
    class_names: list[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]

    def to(self, device: torch.device) -> "EncodedDataset":
        return EncodedDataset(
            name=self.name,
            features=self.features.to(device),
            labels=self.labels.to(device),
            class_names=self.class_names,
        )


@dataclass(slots=True, kw_only=True)
class Individual:
    """
    Mutable entity (metrics update allowed once).
    """

    id: uuid.UUID | str = field(default_factory=uuid.uuid4)
    genotype: PromptGenotype
    generation_born: int
    parents: list[uuid.UUID | str] = field(default_factory=list)
    operation: CreationOperation
    concept: str  # The concept this individual belongs to
    metadata: dict[str, Any] = field(default_factory=dict)

    # Initially None
    metrics: EvaluationMetrics | None = field(default=None, init=False)

    @property
    def is_evaluated(self) -> bool:
        return self.metrics is not None

    @property
    def signature(self) -> tuple[str, str]:
        """
        Returns a hashable representation of the Genotype.
        Used for deduplication.
        """
        return (self.genotype.negative_prompt, self.genotype.positive_prompt)

    def update_metrics(self, metrics: EvaluationMetrics) -> None:
        if self.metrics is not None:
            raise RuntimeError(f"individual {self.id} is already evaluated.")
        self.metrics = metrics

    def get_fitness(self, metric: MetricName) -> float:
        if self.metrics is None:
            raise RuntimeError(f"individual {self.id} has no metrics.")
        return self.metrics[metric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "genotype": self.genotype,
            "metrics": self.metrics,
            "generation": self.generation_born,
            "parents": [str(p) for p in self.parents],
            "operation": self.operation,
            "metadata": self.metadata,
            "concept": self.concept,
        }


@dataclass(slots=True)
class Population:
    """
    Represents an Island of Evolution.
    It maintains a collection of individuals focused on a specific medical concept.
    """

    # The "Concept" this island isolates (e.g., "Texture", "Shape")
    concept: str

    # Unique Island ID
    id: uuid.UUID | str = field(default_factory=uuid.uuid4)

    # Capacity limit (PromptBreeder/EvoPrompt usually limit this to 50-100)
    capacity: int = 50

    # Internal storage
    _individuals: list[Individual] = field(default_factory=list)

    # Stores signatures of individuals for deduplication, we will not remove signatures upon deletion
    _signatures: set[tuple[str, str]] = field(default_factory=set)

    # track the generation
    _generation: int = 0

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self) -> Iterator[Individual]:
        return iter(self._individuals)

    @property
    def generation(self) -> int:
        """Returns the current generation of the population."""
        return self._generation

    def increment_generation(self) -> None:
        """Increments the generation counter by one."""
        self._generation += 1

    def is_all_evaluated(self) -> bool:
        """Checks if all individuals in the population have been evaluated."""
        return all(individual.is_evaluated for individual in self._individuals)

    def _sort_by_metric(self, metric: MetricName) -> None:
        """Sorts individuals in descending order based on the specified metric."""
        if not self.is_all_evaluated():
            raise RuntimeError(
                "Cannot sort population: not all individuals are evaluated."
            )
        self._individuals.sort(key=lambda ind: ind.get_fitness(metric), reverse=True)

    @property
    def individuals(self) -> list[Individual]:
        """Returns a read-only view of individuals."""
        return list(self._individuals)

    def add_individual(self, individual: Individual) -> bool:
        """
        Adds an individual if it is evaluated and unique.
        Returns True if added, False if it was a duplicate.
        """
        if not individual.is_evaluated:
            raise ValueError("Cannot add unevaluated individual to population.")

        # 2. Deduplication Check
        if individual.signature in self._signatures:
            return False

        # 3. Add
        self._signatures.add(individual.signature)
        self._individuals.append(individual)
        return True

    def add_individuals(self, individuals: Sequence[Individual]) -> int:
        """
        Adds a batch of individuals. Returns count of actually added (non-duplicate) items.
        """
        count = 0
        for ind in individuals:
            if self.add_individual(ind):
                count += 1
        return count

    def keep_elites(self, metric: MetricName) -> None:
        """
        The 'Survival of the Fittest' mechanism.
        Sorts population descending by metric and cuts to capacity.
        """
        # Sort descending (Higher is better)
        self._sort_by_metric(metric)

        # Trim to capacity (Elitism)
        self._individuals = self._individuals[: self.capacity]

    def get_elite_individuals(self, k: int, metric: MetricName) -> list[Individual]:
        """
        Selects the top-K individuals based on the specified metric.
        """
        # Ensure population is sorted
        self._sort_by_metric(metric)
        return self._individuals[:k]

    def get_stats(self, metric: MetricName) -> dict[str, float]:
        """
        Returns high-level stats for the engine logs.
        """
        if not self._individuals:
            return {"max": 0.0, "mean": 0.0, "min": 0.0}

        scores = [c.get_fitness(metric) for c in self._individuals if c.is_evaluated]
        if not scores:
            return {"max": 0.0, "mean": 0.0, "min": 0.0}

        return {"max": max(scores), "mean": statistics.mean(scores), "min": min(scores)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_id": str(self.id),
            "concept": self.concept,
            "size": len(self),
            "individuals": [c.to_dict() for c in self._individuals],
        }
