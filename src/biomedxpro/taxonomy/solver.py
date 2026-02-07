"""
Taxonomic Solver - Hierarchical Decision Tree Trainer.

This module implements the recursive orchestrator that trains a binary decision tree
by instantiating fresh evolutionary engines at each node, converting complex multiclass
problems into focused binary decisions.
"""

from datetime import datetime
from typing import Any, Callable

from loguru import logger

from biomedxpro.core.domain import (
    DecisionNode,
    EncodedDataset,
    EvolutionParams,
    Individual,
    PromptEnsemble,
    TaskDefinition,
)
from biomedxpro.core.interfaces import IArtifactStore
from biomedxpro.engine.orchestrator import Orchestrator
from biomedxpro.taxonomy.dataset_slicer import DatasetSlicer

# Type alias for the factory that creates the inner Orchestrator
# It takes a Dataset and a TaskDefinition, and returns a runnable Orchestrator instance
OrchestratorFactory = Callable[[EncodedDataset, TaskDefinition], Orchestrator]


class TaxonomicSolver:
    """
    The High-Level Recursive Controller.

    Traverses a DecisionNode tree depth-first, invoking the Evolutionary Engine
    at each binary node. Coordinates dataset slicing, artifact persistence, and
    checkpoint recovery.

    Architecture:
    - Uses Dependency Injection via OrchestratorFactory (no coupling to engine internals)
    - Performs zero-copy dataset slicing per node via DatasetSlicer
    - Persists trained ensembles via IArtifactStore
    - Supports checkpoint recovery (skip already-trained nodes)

    Example:
        >>> def factory(ds, task):
        ...     return build_orchestrator(ds, task, config)
        >>> solver = TaxonomicSolver(root, store, train_data, factory, base_config)
        >>> solver.run()
    """

    def __init__(
        self,
        root_node: DecisionNode,
        artifact_store: IArtifactStore,
        train_dataset: EncodedDataset,
        orchestrator_factory: OrchestratorFactory,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
    ):
        """
        Initialize the Taxonomic Solver.

        Args:
            root_node: The root of the decision tree to train
            artifact_store: Persistence layer for trained ensembles
            train_dataset: Full multiclass training dataset (will be sliced per node)
            orchestrator_factory: Factory function to create Orchestrator instances
            base_task_def: Base task configuration (inherited by all nodes)
            evolution_params: Evolution hyperparameters (shared across nodes)
        """
        self.root_node = root_node
        self.store = artifact_store
        self.train_dataset = train_dataset
        self.orchestrator_factory = orchestrator_factory
        self.base_task_def = base_task_def
        self.evolution_params = evolution_params

        # Track training statistics
        self.nodes_trained = 0
        self.nodes_skipped = 0
        self.nodes_failed = 0

    def run(self) -> dict[str, str]:
        """
        Entry point to solve the entire tree.

        Initiates recursive DFS traversal starting from the root node.
        Returns a manifest of trained artifacts for verification.

        Returns:
            Dictionary mapping node_id -> artifact_id for all trained nodes

        Raises:
            RuntimeError: If critical errors occur during training
        """
        logger.info("=" * 80)
        logger.info(
            f"Starting Taxonomic Optimization for Root: {self.root_node.node_id}"
        )
        logger.info(f"Root Group: {self.root_node.group_name}")
        logger.info(f"Total Classes: {len(self.root_node.get_all_classes())}")
        logger.info(f"Training Samples: {self.train_dataset.num_samples}")
        logger.info("=" * 80)

        try:
            self._solve_recursive(self.root_node, depth=0, parent_id=None)
        except Exception as e:
            logger.critical(f"Taxonomic training failed: {e}")
            raise RuntimeError("Training failed at node level") from e

        logger.info("=" * 80)
        logger.success("Taxonomic Optimization Complete.")
        logger.info(
            f"Summary: {self.nodes_trained} trained, {self.nodes_skipped} skipped, {self.nodes_failed} failed"
        )
        logger.info("=" * 80)

        return self._collect_artifact_manifest()

    def _solve_recursive(
        self, node: DecisionNode, depth: int, parent_id: str | None
    ) -> None:
        """
        DFS Traversal to train each node recursively.

        Args:
            node: Current node to process
            depth: Current depth in tree (for logging)
            parent_id: Parent node ID (for provenance tracking)
        """
        indent = "  " * depth
        logger.info(f"{indent}→ Processing Node: {node.node_id} (depth={depth})")

        # 1. Check if Leaf (Base Case)
        if node.is_leaf:
            logger.info(f"{indent}  ✓ LEAF node (no training needed)")
            return

        logger.info(f"{indent}  Binary Split: {node.group_name}")
        logger.info(f"{indent}  Left: {node.left_classes}")
        logger.info(f"{indent}  Right: {node.right_classes}")

        # 2. Check Cache (Checkpoint Recovery)
        # If we already trained this node (e.g. previous run crashed), skip it
        if node.ensemble_artifact_id:
            logger.info(
                f"{indent}  ⚡ Cached artifact found: {node.ensemble_artifact_id}"
            )
            self.nodes_skipped += 1
        else:
            # Train this node
            try:
                self._train_node(node, depth, parent_id)
                self.nodes_trained += 1
            except Exception as e:
                logger.error(f"{indent}  ✗ Training failed: {e}")
                self.nodes_failed += 1
                # Don't propagate - allow sibling/parent recovery
                return

        # 3. Recurse to Children
        if node.left_child:
            self._solve_recursive(node.left_child, depth + 1, node.node_id)
        if node.right_child:
            self._solve_recursive(node.right_child, depth + 1, node.node_id)

    def _train_node(
        self, node: DecisionNode, depth: int, parent_id: str | None
    ) -> None:
        """
        Prepares data and runs evolution for a single binary node.

        Steps:
        1. Slice dataset (multiclass → binary)
        2. Create node-specific TaskDefinition
        3. Instantiate Orchestrator via factory
        4. Run evolution
        5. Create PromptEnsemble
        6. Persist artifacts

        Args:
            node: Node to train
            depth: Tree depth (for logging)
            parent_id: Parent node ID (for metadata)

        Raises:
            ValueError: If node has no data after slicing
            RuntimeError: If evolution produces no individuals
        """
        indent = "  " * depth

        # A. Slice the Data (Multiclass -> Binary)
        logger.debug(
            f"{indent}  Slicing dataset: {node.left_classes} vs {node.right_classes}"
        )

        node_dataset = DatasetSlicer.create_binary_view(
            self.train_dataset,
            node.left_classes,
            node.right_classes,
        )

        if node_dataset.num_samples == 0:
            error_msg = f"Node {node.node_id} has NO data after slicing"
            logger.error(f"{indent}  ✗ {error_msg}")
            raise ValueError(error_msg)

        logger.info(
            f"{indent}  Dataset: {node_dataset.num_samples} samples "
            f"({(node_dataset.labels == 0).sum().item()} left, "
            f"{(node_dataset.labels == 1).sum().item()} right)"
        )

        # B. Define the Task (Binary Semantics)
        task_def = self._create_node_task_definition(node)
        logger.info(
            f"{indent}  Task: '{task_def.task_name}' "
            f"[{task_def.class_names[0]} vs {task_def.class_names[1]}]"
        )

        # C. Instantiate the Engine (The Inner Loop)
        logger.info(f"{indent}  🔧 Initializing Orchestrator...")
        try:
            orchestrator = self.orchestrator_factory(node_dataset, task_def)
        except Exception as e:
            error_msg = f"Failed to create Orchestrator for {node.node_id}: {e}"
            logger.error(f"{indent}  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        # D. Evolve!
        logger.info(
            f"{indent}  🧬 Starting evolution ({self.evolution_params.generations} generations)..."
        )
        try:
            best_individuals = orchestrator.run()
        except Exception as e:
            error_msg = f"Evolution failed for {node.node_id}: {e}"
            logger.error(f"{indent}  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        if not best_individuals:
            error_msg = f"Evolution produced no individuals for {node.node_id}"
            logger.error(f"{indent}  ✗ {error_msg}")
            raise RuntimeError(error_msg)

        logger.success(
            f"{indent}  ✓ Evolution complete: {len(best_individuals)} champions found"
        )

        # E. Create Ensemble
        ensemble = PromptEnsemble.from_individuals(
            best_individuals,
            metric=self.evolution_params.target_metric,
            temperature=0.1,  # TODO: Make configurable
        )

        # F. Persist Artifacts
        metadata = self._build_node_metadata(
            node, task_def, best_individuals, node_dataset, depth, parent_id
        )

        logger.info(f"{indent}  💾 Saving artifacts...")
        try:
            artifact_id = self.store.save_node_artifacts(
                node.node_id, ensemble, metadata
            )
        except Exception as e:
            error_msg = f"Failed to persist artifacts for {node.node_id}: {e}"
            logger.critical(f"{indent}  ✗ {error_msg}")
            raise RuntimeError(error_msg) from e

        # G. Update Node (In-Memory Mutation of Frozen Dataclass)
        # We use object.__setattr__ to bypass the frozen=True restriction
        object.__setattr__(node, "ensemble_artifact_id", artifact_id)

        logger.success(f"{indent}  ✓ Node {node.node_id} trained successfully")
        logger.info(f"{indent}  Artifact ID: {artifact_id}")

    def _create_node_task_definition(self, node: DecisionNode) -> TaskDefinition:
        """
        Creates a node-specific binary TaskDefinition.

        Inherits base configuration but overrides class names and task name
        to reflect the binary split at this node.

        Args:
            node: DecisionNode to create task for

        Returns:
            TaskDefinition configured for binary classification at this node
        """
        left_name, right_name = node.get_binary_class_names()

        return TaskDefinition(
            task_name=f"{self.base_task_def.task_name} - {node.group_name}",
            image_modality=self.base_task_def.image_modality,
            class_names=[left_name, right_name],  # Binary override
            role=self.base_task_def.role,
            concepts=None,  # Let LLM discover concepts per node
            description=(
                f"Binary classification at node '{node.node_id}': {node.group_name}. "
                f"Distinguish {left_name} from {right_name}."
            ),
        )

    def _build_node_metadata(
        self,
        node: DecisionNode,
        task_def: TaskDefinition,
        best_individuals: list[Individual],
        node_dataset: EncodedDataset,
        depth: int,
        parent_id: str | None,
    ) -> dict[str, Any]:
        """
        Constructs comprehensive metadata for artifact storage.

        Metadata enables:
        - Provenance tracking (parent relationships)
        - Performance analysis (metrics per node)
        - Debugging (sample counts, class distributions)
        - Inference (label mappings for predictions)

        Args:
            node: Current decision node
            task_def: Binary task definition
            best_individuals: Evolved champions
            node_dataset: Sliced binary dataset
            depth: Tree depth
            parent_id: Parent node ID

        Returns:
            Dictionary with comprehensive node metadata
        """
        # Get best individual's metrics
        best_metrics = best_individuals[0].metrics

        return {
            # Node Identity
            "node_id": node.node_id,
            "group_name": node.group_name,
            "parent_node_id": parent_id,
            "depth": depth,
            # Class Information
            "left_classes": node.left_classes,
            "right_classes": node.right_classes,
            "label_mapping": {
                task_def.class_names[0]: 0,  # Left
                task_def.class_names[1]: 1,  # Right
            },
            # Training Statistics
            "num_champions": len(best_individuals),
            "training_samples": node_dataset.num_samples,
            "left_samples": int((node_dataset.labels == 0).sum().item()),
            "right_samples": int((node_dataset.labels == 1).sum().item()),
            # Performance Metrics (from best individual)
            "best_metrics": best_metrics,
            "target_metric": self.evolution_params.target_metric,
            "best_score": best_individuals[0].get_fitness(
                self.evolution_params.target_metric
            )
            if best_individuals[0].metrics
            else None,
            # Configuration
            "generations": self.evolution_params.generations,
            "task_name": task_def.task_name,
            # Provenance
            "timestamp": datetime.now().isoformat(),
            "concepts": [ind.concept for ind in best_individuals],
        }

    def _collect_artifact_manifest(self) -> dict[str, str]:
        """
        Collects artifact IDs from all trained nodes in the tree.

        Performs DFS traversal to gather all ensemble_artifact_id values.
        Useful for verification and batch loading during inference.

        Returns:
            Dictionary mapping node_id -> artifact_id
        """
        manifest: dict[str, str] = {}

        def _traverse(node: DecisionNode) -> None:
            if node.ensemble_artifact_id:
                manifest[node.node_id] = node.ensemble_artifact_id
            if node.left_child:
                _traverse(node.left_child)
            if node.right_child:
                _traverse(node.right_child)

        _traverse(self.root_node)
        return manifest
