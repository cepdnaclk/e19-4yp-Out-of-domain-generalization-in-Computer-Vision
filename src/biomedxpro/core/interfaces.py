# src/biomedxpro/core/interfaces.py
from typing import Any, Protocol, Sequence

from biomedxpro.core.domain import (
    DataSplit,
    DecisionNode,
    EncodedDataset,
    EvaluationMetrics,
    Individual,
    MetricName,
    Population,
    PromptEnsemble,
    StandardSample,
    TaskDefinition,
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

    def load_samples(self, split: DataSplit) -> list[StandardSample]:
        """
        Parse the dataset for the given split and return a list of samples.
        Note: The dataset root path should be injected via __init__.

        Args:
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

    def evaluate_ensemble(
        self,
        ensemble: PromptEnsemble,
        dataset: EncodedDataset,
    ) -> EvaluationMetrics:
        """
        Evaluates a Prompt Ensemble as a single unit.
        Returns the ensemble-level metric (e.g., F1 score).
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

    def mutate(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        current_generation: int,
        target_metric: MetricName,
        cross_concept_exemplars: Sequence[Individual] | None = None,
    ) -> Sequence[Individual]:
        """
        Generates offspring via LLM-based mutation.
        The LLM prompt must explicitly include instructions like:
        'Focus strictly on visual features related to {concept}.'

        Args:
            cross_concept_exemplars: Optional top individuals from OTHER islands to provide
                      cross-concept context and formatting examples.
        """
        ...

    def crossover(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        current_generation: int,
        target_metric: MetricName,
    ) -> Sequence[Individual]:
        """
        Generates offspring via crossover (genetic recombination).
        Pure combinatorial logic - no LLM needed.
        Mixes prompts from different parents to create new combinations.
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


class ITaxonomyBuilder(Protocol):
    """
    Constructs hierarchical taxonomies from flat class lists.

    The Architect: This component leverages LLM semantic reasoning to organize
    classes into a binary decision tree based on visual and pathological similarity.

    Implementations:
    - LLMTaxonomyBuilder: Uses an ILLMClient to discover structure
    - ManualTaxonomyBuilder: Loads predefined JSON tree (clinician-specified)
    - HeuristicTaxonomyBuilder: Uses rule-based clustering

    Critical Requirement: The builder MUST enforce set-theory validation.
    The union of all leaf nodes must equal the input class set (bijective mapping).
    """

    def build_taxonomy(
        self,
        task_definition: "TaskDefinition",
    ) -> "DecisionNode":
        """
        Constructs a binary decision tree from the task's class list.

        Args:
            task_definition: The complete task specification containing class_names,
                           task_name, modality, and other semantic context

        Returns:
            The root DecisionNode of the constructed taxonomy.

        Raises:
            ValueError: If the tree fails set-theory validation (hallucinations, omissions, overlaps)

        Example:
            Input: TaskDefinition with class_names=["BCC", "SCC", "Melanoma", "Nevus"]
            Output Tree:
                Root [All Classes]
                ├─ Left: Carcinomas ["BCC", "SCC"]
                │  ├─ Left: ["BCC"]
                │  └─ Right: ["SCC"]
                └─ Right: Melanocytic ["Melanoma", "Nevus"]
                   ├─ Left: ["Melanoma"]
                   └─ Right: ["Nevus"]
        """
        ...


class IArtifactStore(Protocol):
    """
    Persists and retrieves node-specific trained artifacts.

    The Archive: In a production taxonomy with 50+ nodes, we cannot keep all
    PromptEnsembles in RAM. This interface provides standardized save/load
    operations for node artifacts.

    Each node's artifact includes:
    - The trained PromptEnsemble (top individuals)
    - Node metadata (label mappings, binary metrics, parent lineage)
    - Evaluation metrics from validation

    Storage Strategy:
    - File-based: artifacts/{experiment_id}/{node_id}/
    - Database: SQL table with BLOB columns
    - Cloud: S3/Azure Blob with node_id as key
    """

    def save_node_artifacts(
        self,
        node_id: str,
        ensemble: "PromptEnsemble",
        metadata: dict[str, Any],
    ) -> str:
        """
        Persists the trained artifacts for a specific node.

        Args:
            node_id: Unique identifier for this decision node
            ensemble: The trained PromptEnsemble (collection of top Individuals)
            metadata: Node-specific metadata including:
                - label_mapping: {class_name: binary_index} for this node
                - metrics: Binary classification metrics (accuracy, F1, AUC)
                - parent_node_id: For provenance tracking
                - group_name: Semantic label for this decision

        Returns:
            artifact_id: A unique identifier or path for retrieval (e.g., file path or UUID)

        Example:
            artifact_id = store.save_node_artifacts(
                node_id="node_1_left",
                ensemble=trained_ensemble,
                metadata={
                    "label_mapping": {"BCC": 0, "SCC": 1},
                    "metrics": {"f1_macro": 0.92, "auc": 0.95},
                    "parent_node_id": "root",
                    "group_name": "Carcinomas"
                }
            )
        """
        ...

    def load_node_artifacts(
        self,
        artifact_id: str,
    ) -> tuple["PromptEnsemble", dict[str, Any]]:
        """
        Retrieves the trained artifacts for a specific node.

        Args:
            artifact_id: The identifier returned by save_node_artifacts()

        Returns:
            (ensemble, metadata) tuple:
                - ensemble: The reconstructed PromptEnsemble
                - metadata: The original metadata dictionary

        Raises:
            FileNotFoundError: If the artifact does not exist
            ValueError: If the artifact is corrupted or incompatible

        Example:
            ensemble, metadata = store.load_node_artifacts("artifacts/exp1/node_1_left")
            print(metadata["group_name"])  # "Carcinomas"
            predictions = ensemble.apply(expert_probs)
        """
        ...


class ITaxonomicPredictor(Protocol):
    """
    Abstract contract for any system that can perform taxonomic inference.

    This decouples the API/Consumer layer from the specific implementation
    (Tree Traversal, Flat Ensemble, Remote Service, etc.).
    """

    def predict(self, input_text: str) -> dict[str, Any]:
        """
        Classifies the input and returns the decision path.

        Args:
            input_text: The content to classify.

        Returns:
            Dictionary containing 'final_class', 'path', and 'status'.
        """
        pass
