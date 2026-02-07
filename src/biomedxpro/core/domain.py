# src/biomedxpro/core/domain.py
import statistics
import uuid
from dataclasses import dataclass, field, fields
from enum import StrEnum, auto
from typing import Any, Iterator, Literal, NotRequired, Sequence, TypedDict

import torch
import torch.nn.functional as F

# --- Enums & Value Objects ---


class CreationOperation(StrEnum):
    INITIALIZATION = auto()
    LLM_MUTATION = auto()
    CROSSOVER = auto()


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
    DNA as a pure ordered sequence.

    Mathematical invariant: prompts[i] corresponds to task_def.class_names[i]
    and aligns with label tensor indices.

    Immutable structure ensures accidental mutation is impossible.
    """

    # Ordered prompts: Index i = class i from TaskDefinition.class_names
    prompts: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize as list (order preserved)."""
        return {"prompts": list(self.prompts)}

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in this genotype."""
        return len(self.prompts)

    def __len__(self) -> int:
        """Support len() for convenience."""
        return len(self.prompts)

    def validate_against_task(self, task_def: "TaskDefinition") -> None:
        """
        Runtime contract enforcement: Ensures this genotype is compatible with the task.

        Raises:
            ValueError: If the number of prompts doesn't match the task's class count.

        This prevents bugs where a binary-trained genotype is accidentally used
        for a multiclass task, or vice versa.
        """
        if len(self.prompts) != task_def.num_classes:
            raise ValueError(
                f"Genotype validation failed: Expected {task_def.num_classes} prompts "
                f"for task '{task_def.task_name}' with classes {task_def.class_names}, "
                f"but genotype has {len(self.prompts)} prompts."
            )


class EvaluationMetrics(TypedDict):
    """The Report Card."""

    inverted_bce: float
    f1_macro: float
    accuracy: float
    auc: float
    f1_weighted: float
    soft_f1_macro: float
    confusion_matrix: NotRequired[list[list[int]]]


MetricName = Literal[
    "inverted_bce", "f1_macro", "accuracy", "auc", "f1_weighted", "soft_f1_macro"
]


@dataclass(slots=True, frozen=True)
class EvolutionParams:
    """
    The Control Plane configuration.
    Defines all hyperparameters for the evolutionary run.
    """

    # Global Settings
    generations: int = 20
    target_metric: MetricName = "soft_f1_macro"

    # Island Settings
    island_capacity: int = 100  # Max size of an island
    initial_pop_size: int = 50  # How many prompts to start with

    # Evolutionary Operator Settings
    num_parents: int = 10  # How many winners to select
    offspring_mutated: int = 5  # How many children to create via mutation
    offspring_crossover: int = 5  # How many children to create via crossover

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
    # Replaces positive/negative_class - supports N classes
    class_names: list[str]  # e.g. ["Benign", "Malignant", "In-Situ"]
    concepts: list[str] | None  # e.g. ["Texture", "Color", "Shape"]
    role: str  # e.g. "Expert Dermatologist"

    # Optional: Any extra specific instructions for this dataset
    description: str = ""

    @property
    def num_classes(self) -> int:
        """Returns the number of classes in this task."""
        return len(self.class_names)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "TaskDefinition":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)


@dataclass(slots=True, frozen=True)
class DecisionNode:
    """
    Represents a binary decision point in the taxonomic hierarchy.

    The Composite Pattern: Each node is either:
    - A Leaf (terminal classification)
    - A Branch (splits data into left_child and right_child)

    This is the skeleton of the Taxonomic Evolutionary Solver.
    Each node will have its own trained PromptEnsemble artifact.
    """

    # Unique identifier for this node (e.g., "root", "node_1_left", "melanocytic_vs_nonmelanocytic")
    node_id: str

    # Semantic label for this decision (e.g., "Melanocytic Lesions", "Solid Renal Neoplasms")
    # This is CRITICAL: guides the LLM during prompt evolution by providing biological context
    group_name: str

    # The partition of the universe at this node
    # These are class names (strings), NOT indices
    left_classes: list[str]  # Maps to binary target 0 during training
    right_classes: list[str]  # Maps to binary target 1 during training

    # Tree structure (None for leaf nodes)
    left_child: "DecisionNode | None" = None
    right_child: "DecisionNode | None" = None

    # Reference to the trained artifact (ensemble ID or file path)
    # None until this node has been trained
    ensemble_artifact_id: str | None = None

    @property
    def is_leaf(self) -> bool:
        """Returns True if this is a terminal node (no children)."""
        return self.left_child is None and self.right_child is None

    @property
    def is_binary(self) -> bool:
        """Returns True if this node represents a binary decision (has children)."""
        return not self.is_leaf

    def get_all_classes(self) -> list[str]:
        """
        Returns the union of left_classes and right_classes.
        Used for validation during tree construction.
        """
        return self.left_classes + self.right_classes

    def get_binary_class_names(self) -> tuple[str, str]:
        """
        Generates semantic binary class names for this node's decision.

        Instead of returning generic "Left" vs "Right", this method creates
        biologically meaningful labels by:
        1. Using group_name if this is a leaf-vs-group split
        2. Extracting common prefixes/suffixes from class lists
        3. Falling back to concatenated class lists

        Returns:
            (left_label, right_label) tuple for TaskDefinition.class_names

        Example:
            Node splitting ["BCC", "SCC"] vs ["Melanoma", "Nevus"]
            Returns: ("Carcinomas", "Melanocytic")
        """
        # Simple heuristic: use first class name as representative
        # A more sophisticated implementation could use an LLM to generate semantic names
        left_label = (
            self.left_classes[0]
            if len(self.left_classes) == 1
            else f"{self.left_classes[0]}_group"
        )
        right_label = (
            self.right_classes[0]
            if len(self.right_classes) == 1
            else f"{self.right_classes[0]}_group"
        )
        return (left_label, right_label)

    def __post_init__(self) -> None:
        """Validation: Ensure node is self-consistent."""

        # IF LEAF: We are done. No need to validate class splits.
        if self.is_leaf:
            return

        # IF BRANCH: Validate that left and right classes are valid split
        left_set = set(self.left_classes)
        right_set = set(self.right_classes)

        if left_set & right_set:
            raise ValueError(
                f"Node {self.node_id}: left_classes and right_classes must be disjoint. "
                f"Found overlap: {left_set & right_set}"
            )

        if not self.left_classes or not self.right_classes:
            raise ValueError(
                f"Node {self.node_id}: Both left_classes and right_classes must be non-empty."
            )


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
    def signature(self) -> tuple[str, ...]:
        """
        Returns a hashable representation of the Genotype.
        Used for deduplication.

        The genotype.prompts is already a tuple, so we return it directly.
        """
        return self.genotype.prompts

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
    # Variable-length tuples for multi-class support
    _signatures: set[tuple[str, ...]] = field(default_factory=set)

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


@dataclass(slots=True)
class PromptEnsemble:
    """
    A Pure Domain Entity.
    It holds the state (experts + weights) and the business logic
    for combining them. It relies on the caller to provide the raw probabilities.
    """

    experts: list[Individual]
    weights: torch.Tensor
    metric: MetricName

    @property
    def prompts(self) -> list[list[str]]:
        """
        Helper to extract the raw text prompts for all experts.
        Returns list of prompt lists, one per expert.
        Each inner list contains prompts in canonical order (index = class).
        """
        return [list(ind.genotype.prompts) for ind in self.experts]

    @classmethod
    def from_individuals(
        cls,
        individuals: list[Individual],
        metric: MetricName,
        temperature: float = 0.1,
    ) -> "PromptEnsemble":
        """Factory: Pure logic to calculate weights based on fitness."""
        if not individuals:
            raise ValueError("Cannot create ensemble from empty list.")

        scores = torch.tensor(
            [ind.get_fitness(metric) for ind in individuals], dtype=torch.float32
        )
        # Softmax Weighting (Pure Math)
        weights = torch.softmax(scores / temperature, dim=0)

        return cls(experts=individuals, weights=weights, metric=metric)

    def apply(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Applies the ensemble weights using Logarithmic Opinion Pool (LogOP).

        Args:
            expert_probs: Tensor of shape (N_Samples, N_Experts, N_Classes).
                          This data comes from the outside world.

        Returns:
            Tensor of shape (N_Samples, N_Classes). The final weighted class probabilities.
        """
        if expert_probs.shape[1] != len(self.experts):
            raise ValueError(
                f"Dimension Mismatch: Ensemble has {len(self.experts)} experts, "
                f"but input tensor has {expert_probs.shape[1]} experts in dim 1."
            )

        return self._apply_logarithmic(expert_probs)

    def _apply_logarithmic(self, expert_probs: torch.Tensor) -> torch.Tensor:
        """
        Logarithmic Opinion Pool (LogOP) - Geometric Mean.

        Good for: Consensus enforcement, veto power
        Bad for: Can be overly conservative if one expert is wrong

        Math: P_final = Softmax(Sum(w_i * Log(P_i)))

        Why: If one expert says prob=0, the ensemble respects that veto.
        All experts must agree for high confidence.
        """
        # Numerical stability: avoid log(0)
        eps = 1e-9
        log_probs = torch.log(expert_probs + eps)

        w = self.weights.to(expert_probs.device).view(1, -1, 1)

        # Weighted sum of LOG probabilities
        weighted_log_sum = (log_probs * w).sum(dim=1)

        # Re-normalize back to probability distribution
        return F.softmax(weighted_log_sum, dim=1)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the ensemble to a pure Python dictionary.
        Converts the Tensor weights to a list for JSON/pickle compatibility.

        Returns:
            Dictionary with experts, weights (as list), and metric name.
        """
        return {
            "experts": [ind.to_dict() for ind in self.experts],
            # CRITICAL: Tensor -> List for JSON serialization
            "weights": self.weights.tolist(),
            "metric": self.metric,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptEnsemble":
        """
        Reconstructs the ensemble from a dictionary.

        Args:
            data: Dictionary containing serialized ensemble data

        Returns:
            Reconstructed PromptEnsemble instance

        Raises:
            KeyError: If required keys are missing from data
            ValueError: If data is malformed
        """
        # Reconstruct Individuals from serialized data
        experts = []
        for ind_data in data["experts"]:
            # Reconstruct genotype from prompts list
            genotype = PromptGenotype(prompts=tuple(ind_data["genotype"]["prompts"]))

            # Reconstruct individual
            ind = Individual(
                id=ind_data["id"],
                genotype=genotype,
                generation_born=ind_data["generation"],
                parents=ind_data["parents"],
                operation=CreationOperation(ind_data["operation"]),
                concept=ind_data["concept"],
                metadata=ind_data.get("metadata", {}),
            )

            # Re-attach metrics if they exist
            if ind_data.get("metrics"):
                ind.update_metrics(ind_data["metrics"])

            experts.append(ind)

        # CRITICAL: List -> Tensor reconstruction
        weights = torch.tensor(data["weights"], dtype=torch.float32)

        return cls(
            experts=experts,
            weights=weights,
            metric=data["metric"],
        )
