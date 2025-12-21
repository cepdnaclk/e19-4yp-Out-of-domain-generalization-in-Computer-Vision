import statistics
import uuid
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import Any, Iterator, Literal, NotRequired, TypedDict

import torch

# --- Enums & Value Objects ---


class CreationOperator(StrEnum):
    INITIALIZATION = auto()
    LLM_MUTATION = auto()


class DataSplit(StrEnum):
    TRAIN = auto()
    VAL = auto()
    TEST = auto()


class PromptGenotype(TypedDict):
    """Immutable DNA."""

    negative_prompt: str
    positive_prompt: str


class EvaluationMetrics(TypedDict):
    """The Report Card."""

    inverted_bce: float
    f1_macro: float
    accuracy: float
    auc: float
    f1_weighted: float
    confusion_matrix: NotRequired[list[list[int]]]


MetricName = Literal["inverted_bce", "f1_macro", "accuracy", "auc", "f1_weighted"]

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
class PromptCandidate:
    """
    Mutable entity (metrics update allowed once).
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)
    genotype: PromptGenotype
    generation_born: int
    parents: list[uuid.UUID] = field(default_factory=list)
    operator: CreationOperator

    # New: Track which medical concept this candidate was born to address
    # e.g. "Cell Texture" vs "Nucleus Shape"
    concept: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    # Initially None
    metrics: EvaluationMetrics | None = field(default=None, init=False)

    @property
    def is_evaluated(self) -> bool:
        return self.metrics is not None

    def update_metrics(self, metrics: EvaluationMetrics) -> None:
        if self.metrics is not None:
            raise RuntimeError(f"Candidate {self.id} is already evaluated.")
        self.metrics = metrics

    def get_fitness(self, metric: MetricName) -> float:
        if self.metrics is None:
            raise RuntimeError(f"Candidate {self.id} has no metrics.")
        return self.metrics[metric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "concept": self.concept,
            "genotype": self.genotype,
            "metrics": self.metrics,
            "generation": self.generation_born,
            "parents": [str(p) for p in self.parents],
            "operator": self.operator,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Population:
    """
    Represents an Island of Evolution.
    It maintains a collection of candidates focused on a specific medical concept.
    """

    id: uuid.UUID = field(default_factory=uuid.uuid4)

    # The "Concept" this island isolates (e.g., "Texture", "Shape")
    concept: str

    # Capacity limit (PromptBreeder/EvoPrompt usually limit this to 50-100)
    capacity: int = 50

    # Internal storage
    _candidates: list[PromptCandidate] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._candidates)

    def __iter__(self) -> Iterator[PromptCandidate]:
        return iter(self._candidates)

    @property
    def candidates(self) -> list[PromptCandidate]:
        """Returns a read-only view of candidates."""
        return list(self._candidates)

    def add(self, candidate: PromptCandidate) -> None:
        """
        Adds a candidate. If island is full, we DO NOT evict immediately.
        We usually evict after sorting.
        """
        self._candidates.append(candidate)

    def sort_and_trim(self, metric: MetricName) -> None:
        """
        The 'Survival of the Fittest' mechanism.
        Sorts population descending by metric and cuts to capacity.
        """
        # Sort descending (Higher is better)
        # Note: We filter out unevaluated candidates to prevent crashes
        evaluated = [c for c in self._candidates if c.is_evaluated]

        evaluated.sort(key=lambda c: c.get_fitness(metric), reverse=True)

        # Trim to capacity (Elitism)
        self._candidates = evaluated[: self.capacity]

    def get_stats(self, metric: MetricName) -> dict[str, float]:
        """
        Returns high-level stats for the engine logs.
        """
        if not self._candidates:
            return {"max": 0.0, "mean": 0.0, "min": 0.0}

        scores = [c.get_fitness(metric) for c in self._candidates if c.is_evaluated]
        if not scores:
            return {"max": 0.0, "mean": 0.0, "min": 0.0}

        return {"max": max(scores), "mean": statistics.mean(scores), "min": min(scores)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "island_id": str(self.id),
            "concept": self.concept,
            "size": len(self),
            "candidates": [c.to_dict() for c in self._candidates],
        }
