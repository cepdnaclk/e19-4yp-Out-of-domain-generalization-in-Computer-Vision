from typing import Protocol, Sequence

from biomedxpro.core.domain import (
    EncodedDataset,
    MetricName,
    Population,
    PromptCandidate,
)


class IFitnessEvaluator(Protocol):
    """
    Evaluates candidates against the ground truth.
    This is usually universal across all islands.
    """

    def evaluate(
        self, candidates: Sequence[PromptCandidate], dataset: EncodedDataset
    ) -> None:
        """
        Calculates scores (e.g., F1, BCE) and calls candidate.update_metrics().
        """
        ...


class IOperator(Protocol):
    """
    The LLM Interface.
    Crucially, it must be 'Concept-Aware'.
    """

    def reproduce(
        self, parents: Sequence[PromptCandidate], concept: str, num_offsprings: int
    ) -> Sequence[PromptCandidate]:
        """
        Generates offspring.
        The LLM prompt must explicitly include instructions like:
        'Focus strictly on visual features related to {concept}.'
        """
        ...

    def initialize_population(
        self, num_offsprings: int, concept: str
    ) -> Sequence[PromptCandidate]:
        """
        Creates the Adam & Eve for a specific island (e.g., 'Generate 10 prompts about Irregular Shape').
        """
        ...


class SelectionStrategy(Protocol):
    """
    Decides which candidates become parents for the next generation.
    Operates locally within one island.
    """

    def select(
        self, population: Population, k: int, metric: MetricName
    ) -> Sequence[PromptCandidate]: ...
