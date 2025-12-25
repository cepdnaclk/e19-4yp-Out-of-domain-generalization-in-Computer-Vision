# src/biomedxpro/core/interfaces.py
from typing import Protocol, Sequence

from biomedxpro.core.domain import EncodedDataset, Individual, MetricName, Population


class IFitnessEvaluator(Protocol):
    """
    Evaluates candidates against the ground truth.
    This is usually universal across all islands.
    """

    def evaluate(
        self, candidates: Sequence[Individual], dataset: EncodedDataset
    ) -> None:
        """
        Calculates scores (e.g., F1, BCE) and calls candidate.update_metrics().
        """
        ...


class ILLMClient(Protocol):
    """
    Abstract contract for any Text-to-Text model provider.
    This allows us to swap Gemini for Ollama, OpenAI, or a Mock without breaking code.
    """

    def generate(self, prompt: str) -> str:
        """
        Takes a raw string prompt and returns the raw string response.
        Implementations should handle retries, timeouts, and API errors internally.
        """
        ...


class IOperator(Protocol):
    """
    The Operator Interface utilizing the ILLMClient.
    Crucially, it must be 'Concept-Aware'.
    """

    def discover_concepts(self) -> list[str]:
        """
        Generates a list of medical concepts to form islands around.
        """

    def reproduce(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        target_metric: MetricName,
    ) -> Sequence[Individual]:
        """
        Generates offspring.
        The LLM prompt must explicitly include instructions like:
        'Focus strictly on visual features related to {concept}.'
        """
        ...

    def initialize_population(
        self, num_offsprings: int, concept: str
    ) -> Sequence[Individual]:
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
    ) -> Sequence[Individual]: ...


class IHistoryRecorder(Protocol):
    """
    Contract for persistence layers.
    Allows swapping between JSONL files, SQL Databases, or WandB/MLflow.
    """

    def record_generation(self, islands: Sequence[Population]) -> None:
        """
        Persists the state of the entire archipelago for the current generation.
        """
        ...
