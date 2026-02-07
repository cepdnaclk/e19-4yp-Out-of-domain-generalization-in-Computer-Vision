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
        self, node: DecisionNode, depth: int = 0, parent_id: str | None = None
    ) -> None:
        """
        DFS Traversal to train each node.
        """
        # Visual indentation for logs
        indent = "  " * depth

        # 1. Base Case: Semantic Purity Check
        # We only skip if the node represents a single final concept.
        # If a node has multiple classes (e.g. "Cyst" vs "Tumor") it MUST be trained,
        # even if it has no children objects (Implicit Leaf).
        all_classes = node.get_all_classes()
        if len(all_classes) <= 1:
            logger.info(
                f"{indent}Node {node.node_id} is a PURE LEAF {all_classes}. Skipping."
            )
            return

        logger.info(f"{indent}=== Solving Node: {node.node_id} ({node.group_name}) ===")

        # 2. Idempotency Check (Cache Layer)
        if node.ensemble_artifact_id:
            logger.info(f"{indent}Node {node.node_id} already has artifact. Skipping.")
            self.nodes_skipped += 1
        else:
            # Attempt to train
            success = self._train_node(node, depth, parent_id)
            if success:
                self.nodes_trained += 1
            else:
                self.nodes_failed += 1

        # 3. Recursive Step
        # If children exist, we traverse them.
        # If children do NOT exist (Terminal Decision Node), recursion naturally stops here.
        if node.left_child:
            self._solve_recursive(
                node.left_child, depth=depth + 1, parent_id=node.node_id
            )

        if node.right_child:
            self._solve_recursive(
                node.right_child, depth=depth + 1, parent_id=node.node_id
            )

    def _train_node(
        self, node: DecisionNode, depth: int, parent_id: str | None
    ) -> bool:
        """
        Returns:
            bool: True if training succeeded, False if skipped/failed gracefully.
        """
        indent = "  " * depth

        # A. Slice the Data
        # We wrap this in a try-block to handle data starvation gracefully
        try:
            node_dataset = DatasetSlicer.create_binary_view(
                self.train_dataset,
                node.left_classes,
                node.right_classes,
            )
        except Exception as e:
            # Catch slicing errors (e.g. alignment issues)
            logger.warning(f"{indent}  ! Data slicing failed for {node.node_id}: {e}")
            return False

        if node_dataset.num_samples == 0:
            logger.warning(f"{indent}  ! Node {node.node_id} has NO data. Skipping.")
            return False

        logger.info(
            f"{indent}  Dataset: {node_dataset.num_samples} samples "
            f"({(node_dataset.labels == 0).sum().item()} left, "
            f"{(node_dataset.labels == 1).sum().item()} right)"
        )

        # B. Define Task & C. Instantiate Engine
        try:
            task_def = self._create_node_task_definition(node)
            logger.info(f"{indent}  🔧 Initializing Orchestrator...")
            orchestrator = self.orchestrator_factory(node_dataset, task_def)

            # D. Evolve
            logger.info(f"{indent}  🧬 Starting evolution...")
            best_individuals = orchestrator.run()

            if not best_individuals:
                logger.error(f"{indent}  ✗ Evolution produced no individuals.")
                return False

            # E. Ensemble & F. Persist
            ensemble = PromptEnsemble.from_individuals(
                best_individuals,
                metric=self.evolution_params.target_metric,
                temperature=0.1,
            )

            metadata = self._build_node_metadata(
                node, task_def, best_individuals, node_dataset, depth, parent_id
            )

            artifact_id = self.store.save_node_artifacts(
                node.node_id, ensemble, metadata
            )

            # G. Update Node
            object.__setattr__(node, "ensemble_artifact_id", artifact_id)

            logger.success(
                f"{indent}  ✓ Node {node.node_id} trained. ID: {artifact_id}"
            )
            return True

        except Exception as e:
            logger.error(f"{indent}  ✗ Training failed for {node.node_id}: {e}")
            # We return False (Soft Failure) so the tree traversal can continue
            # to other branches that might be healthy.
            return False

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
