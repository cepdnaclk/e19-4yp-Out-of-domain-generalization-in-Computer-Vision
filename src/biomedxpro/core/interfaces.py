# src/biomedxpro/core/interfaces.py
from typing import Protocol, Sequence

from biomedxpro.core.domain import (
    DataSplit,
    EncodedDataset,
    Individual,
    MetricName,
    Population,
    StandardSample,
)


class IDatasetAdapter(Protocol):
    """
    Strategy for standardizing a specific dataset format.
    
    Different datasets have different folder structures, CSV formats, and label mappings.
    An adapter knows how to parse the "messy" raw data and convert it into a clean,
    standardized list of (filepath, label) pairs.
    
    Each adapter is registered by a unique string key (e.g., "derm7pt", "camelyon17")
    so it can be selected via configuration.
    """

    def load_samples(
        self, root: str, split: DataSplit
    ) -> list[StandardSample]:
        """
        Parse the dataset at `root` for the given split and return a list of samples.
        
        Args:
            root: The root directory containing the dataset files.
            split: The split to load (TRAIN, VAL, or TEST).
        
        Returns:
            A list of StandardSample objects, each containing an image_path and label.
            This is the "Common Currency" that the DataLoader understands.
        """
        ...


class IFitnessEvaluator(Protocol):
    """
    Evaluates individuals against the ground truth.
    This is usually universal across all islands.
    """

    def evaluate(
        self, individuals: Sequence[Individual], dataset: EncodedDataset
    ) -> None:
        """
        Calculates scores (e.g., F1, BCE) and calls individual.update_metrics().
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
        current_generation: int,
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
