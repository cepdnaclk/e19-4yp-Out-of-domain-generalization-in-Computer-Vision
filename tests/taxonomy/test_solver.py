"""
Test cases for TaxonomicSolver.

Tests the recursive orchestrator that trains hierarchical decision trees
by coordinating dataset slicing, orchestrator instantiation, and artifact persistence.
"""

from typing import Callable
from unittest.mock import Mock

import pytest
import torch

from biomedxpro.core.domain import (
    CreationOperation,
    DecisionNode,
    EncodedDataset,
    EvolutionParams,
    Individual,
    PromptGenotype,
    TaskDefinition,
)
from biomedxpro.taxonomy.mocks import MemoryArtifactStore, MockTaxonomyBuilder
from biomedxpro.taxonomy.solver import TaxonomicSolver


class TestTaxonomicSolver:
    """Test suite for TaxonomicSolver recursive training."""

    @pytest.fixture
    def sample_dataset(self) -> EncodedDataset:
        """
        Creates a toy multiclass dataset for testing.

        4 classes with 25 samples each (100 total).
        """
        num_samples = 100
        features = torch.randn(num_samples, 512)
        labels = torch.tensor([i // 25 for i in range(num_samples)])

        return EncodedDataset(
            name="TestDataset",
            features=features,
            labels=labels,
            class_names=["ClassA", "ClassB", "ClassC", "ClassD"],
        )

    @pytest.fixture
    def base_task_def(self) -> TaskDefinition:
        """Base task configuration."""
        return TaskDefinition(
            task_name="Test Classification",
            image_modality="Test Images",
            class_names=["ClassA", "ClassB", "ClassC", "ClassD"],
            role="Test Expert",
            concepts=None,
        )

    @pytest.fixture
    def evolution_params(self) -> EvolutionParams:
        """Minimal evolution parameters for testing."""
        return EvolutionParams(
            generations=2,
            target_metric="soft_f1_macro",
            initial_pop_size=5,
            num_parents=2,
            offspring_mutated=2,
            offspring_crossover=2,
        )

    @pytest.fixture
    def mock_orchestrator_factory(self) -> Callable[..., Mock]:
        """
        Factory that returns mock orchestrators producing fake champions.
        """

        def factory(dataset: EncodedDataset, task: TaskDefinition) -> Mock:
            mock_orch = Mock()
            # Create fake champions for this node
            champions = [
                Individual(
                    genotype=PromptGenotype(prompts=("prompt_left", "prompt_right")),
                    generation_born=1,
                    operation=CreationOperation.INITIALIZATION,
                    concept="MockConcept",
                    metadata={"source": "mock"},
                )
            ]
            # Attach fake metrics
            champions[0].update_metrics(
                {
                    "soft_f1_macro": 0.95,
                    "accuracy": 0.93,
                    "f1_macro": 0.94,
                    "auc": 0.96,
                    "f1_weighted": 0.94,
                    "inverted_bce": 0.92,
                }
            )
            mock_orch.run.return_value = champions
            return mock_orch

        return factory

    def test_simple_tree_training(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test training a tree with trainable nodes."""
        # Complex tree has nodes with children (trainable)
        builder = MockTaxonomyBuilder(use_complex_tree=True)
        root = builder.build_taxonomy(base_task_def)

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=mock_orchestrator_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        # Run training
        manifest = solver.run()

        # Verify orchestrators were called for non-leaf nodes
        assert solver.nodes_trained >= 1
        assert solver.nodes_skipped == 0
        assert solver.nodes_failed == 0

        # Verify artifacts were saved
        assert len(store.list_artifacts()) >= 1

        # Verify manifest contains trained nodes
        assert len(manifest) >= 1

    def test_complex_tree_training(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test training a multi-level tree with internal nodes."""
        # Create complex tree with depth 2
        builder = MockTaxonomyBuilder(use_complex_tree=True)
        root = builder.build_taxonomy(base_task_def)

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=mock_orchestrator_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        manifest = solver.run()

        # Complex tree should have root + 2 intermediate nodes (left and right subtrees)
        # Count non-leaf nodes
        def count_non_leaves(node: DecisionNode) -> int:
            if node.is_leaf:
                return 0
            count = 1  # This node
            if node.left_child:
                count += count_non_leaves(node.left_child)
            if node.right_child:
                count += count_non_leaves(node.right_child)
            return count

        expected_trained = count_non_leaves(root)
        assert solver.nodes_trained == expected_trained
        assert len(store.list_artifacts()) == expected_trained
        assert len(manifest) == expected_trained

    def test_checkpoint_recovery(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test that solver skips nodes that already have artifacts."""
        builder = MockTaxonomyBuilder(use_complex_tree=True)
        root = builder.build_taxonomy(base_task_def)

        # Simulate pre-existing artifact on root
        object.__setattr__(root, "ensemble_artifact_id", "existing_artifact_123")

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=mock_orchestrator_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        manifest = solver.run()

        # Root skipped due to checkpoint, but children should still be trained
        assert solver.nodes_skipped >= 1  # At least root was skipped
        assert (
            solver.nodes_trained >= 0
        )  # Children may or may not be trained depending on structure

        # No new artifacts for root
        assert "existing_artifact_123" not in [aid for aid in store.list_artifacts()]

        # Manifest should contain the existing artifact for root
        assert root.node_id in manifest
        assert manifest[root.node_id] == "existing_artifact_123"

    def test_leaf_nodes_skipped(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test that leaf nodes are correctly identified and skipped."""
        # Create a leaf node
        leaf = DecisionNode(
            node_id="leaf",
            group_name="Leaf Node",
            left_classes=["ClassA"],
            right_classes=["ClassB"],
            # No children = leaf
        )

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=leaf,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=mock_orchestrator_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        manifest = solver.run()

        # Leaf should not be trained
        assert solver.nodes_trained == 0
        assert len(store.list_artifacts()) == 0
        assert len(manifest) == 0

    def test_empty_dataset_handling(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test that solver handles nodes with no data gracefully."""
        # Create tree with children so it's trainable, but dataset has no matching classes
        builder = MockTaxonomyBuilder(use_complex_tree=True)

        # Create a task with classes not in the dataset
        empty_task = TaskDefinition(
            task_name="Empty Test",
            image_modality="Test Images",
            class_names=["ClassC", "ClassD", "ClassE", "ClassF"],
            concepts=None,
            role="Test Expert",
        )
        root = builder.build_taxonomy(empty_task)

        # Dataset has different classes (0=ClassA, 1=ClassB)
        empty_dataset = EncodedDataset(
            name="EmptyTest",
            features=torch.randn(10, 512),
            labels=torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),  # Only classes 0 and 1
            class_names=["ClassA", "ClassB", "ClassC", "ClassD", "ClassE", "ClassF"],
        )

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=empty_dataset,
            orchestrator_factory=mock_orchestrator_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        _ = solver.run()

        # Should fail when trying to slice - at least one node fails
        assert solver.nodes_failed >= 1
        assert solver.nodes_trained == 0
        assert len(store.list_artifacts()) == 0

    def test_task_definition_creation(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test that node-specific task definitions are created correctly."""
        captured_tasks = []

        def capturing_factory(dataset: EncodedDataset, task: TaskDefinition) -> Mock:
            captured_tasks.append(task)
            # Return mock orchestrator
            mock_orch = Mock()
            champion = Individual(
                genotype=PromptGenotype(prompts=("p1", "p2")),
                generation_born=1,
                operation=CreationOperation.INITIALIZATION,
                concept="Test",
            )
            champion.update_metrics(
                {
                    "soft_f1_macro": 0.9,
                    "accuracy": 0.9,
                    "f1_macro": 0.9,
                    "auc": 0.9,
                    "f1_weighted": 0.9,
                    "inverted_bce": 0.9,
                }
            )
            mock_orch.run.return_value = [champion]
            return mock_orch

        builder = MockTaxonomyBuilder(use_complex_tree=True)
        root = builder.build_taxonomy(base_task_def)

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=capturing_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        solver.run()

        # Verify tasks were created (complex tree has multiple nodes)
        assert len(captured_tasks) >= 1
        task = captured_tasks[0]  # Check first task (root)
        assert len(task.class_names) == 2
        assert task.task_name.startswith(base_task_def.task_name)
        assert root.group_name in task.task_name
        assert task.image_modality == base_task_def.image_modality
        assert task.role == base_task_def.role

    def test_metadata_completeness(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
        mock_orchestrator_factory: Callable[..., Mock],
    ) -> None:
        """Test that artifact metadata contains all required fields."""
        builder = MockTaxonomyBuilder(use_complex_tree=True)
        root = builder.build_taxonomy(base_task_def)

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=mock_orchestrator_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        solver.run()

        # Load artifact and check metadata (complex tree has multiple)
        artifacts = store.list_artifacts()
        assert len(artifacts) >= 1  # At least root node

        ensemble, metadata = store.load_node_artifacts(artifacts[0])

        # Verify required metadata fields
        required_fields = [
            "node_id",
            "group_name",
            "parent_node_id",
            "depth",
            "left_classes",
            "right_classes",
            "label_mapping",
            "num_champions",
            "training_samples",
            "left_samples",
            "right_samples",
            "best_metrics",
            "target_metric",
            "generations",
            "timestamp",
        ]

        for field in required_fields:
            assert field in metadata, f"Missing metadata field: {field}"

        # Verify metadata values
        assert metadata["node_id"] == root.node_id
        assert metadata["group_name"] == root.group_name
        assert metadata["parent_node_id"] is None  # Root has no parent
        assert metadata["depth"] == 0
        assert metadata["num_champions"] == 1
        assert metadata["training_samples"] > 0

    def test_dfs_traversal_order(
        self,
        sample_dataset: EncodedDataset,
        base_task_def: TaskDefinition,
        evolution_params: EvolutionParams,
    ) -> None:
        """Test that nodes are trained in correct DFS order."""
        training_order = []

        def order_tracking_factory(
            dataset: EncodedDataset, task: TaskDefinition
        ) -> Mock:
            training_order.append(task.task_name)
            mock_orch = Mock()
            champion = Individual(
                genotype=PromptGenotype(prompts=("p1", "p2")),
                generation_born=1,
                operation=CreationOperation.INITIALIZATION,
                concept="Test",
            )
            champion.update_metrics(
                {
                    "soft_f1_macro": 0.9,
                    "accuracy": 0.9,
                    "f1_macro": 0.9,
                    "auc": 0.9,
                    "f1_weighted": 0.9,
                    "inverted_bce": 0.9,
                }
            )
            mock_orch.run.return_value = [champion]
            return mock_orch

        builder = MockTaxonomyBuilder(use_complex_tree=True)
        root = builder.build_taxonomy(base_task_def)

        store = MemoryArtifactStore()

        solver = TaxonomicSolver(
            root_node=root,
            artifact_store=store,
            train_dataset=sample_dataset,
            orchestrator_factory=order_tracking_factory,
            base_task_def=base_task_def,
            evolution_params=evolution_params,
        )

        solver.run()

        # In DFS, root should be trained first
        assert len(training_order) > 0
        assert "Lesion Classification" in training_order[0]  # Root's group_name
