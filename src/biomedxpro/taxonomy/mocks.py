"""
Mock implementations of taxonomy interfaces for testing and development.

These test doubles allow rapid iteration on the hierarchical solver logic
without requiring LLM API calls or persistent storage systems.
"""

import uuid
from typing import Any

from biomedxpro.core.domain import DecisionNode, PromptEnsemble
from biomedxpro.core.interfaces import IArtifactStore, ITaxonomyBuilder


class MockTaxonomyBuilder(ITaxonomyBuilder):
    """
    A controllable builder that returns a pre-defined tree structure.
    Useful for unit testing the Recursive Solver without calling an LLM.

    Provides deterministic tree structures for testing:
    - Simple: Single binary split (depth 1)
    - Complex: Multi-level hierarchy (depth 2+)
    """

    def __init__(self, use_complex_tree: bool = True):
        """
        Args:
            use_complex_tree: If True, returns a multi-level tree.
                            If False, returns a single binary split.
        """
        self.use_complex_tree = use_complex_tree

    def build_taxonomy(
        self,
        class_names: list[str],
        max_depth: int | None = None,
    ) -> DecisionNode:
        """
        Returns a fixed tree structure for testing.

        Ignores max_depth parameter in mock implementation.
        Validates that class_names has sufficient classes for the chosen structure.

        Args:
            class_names: The complete list of classes to organize
            max_depth: Ignored in mock implementation

        Returns:
            DecisionNode root with deterministic structure

        Raises:
            ValueError: If class_names doesn't have enough classes for the structure
        """
        if not self.use_complex_tree:
            # Simple Case: Root with one binary split, terminating in leaf nodes
            # This creates ONE trainable node (the root)
            if len(class_names) < 2:
                raise ValueError("Need at least 2 classes for binary split")

            # With 4 classes [A, B, C, D]:
            # Root: [A] vs [B, C, D]  <- This is the only trainable node
            # No children = both branches are leaves (no further training)

            # For testing with 4 classes, split [A] vs [B, C, D]
            return DecisionNode(
                node_id="root_simple",
                group_name="Global Classification",
                left_classes=[class_names[0]],  # Single class
                right_classes=class_names[1:],  # Rest of classes
                # No children = leaf node (won't be trained, but that's the test's bug)
            )

        # Complex Case: Multi-level hierarchy (Skin Lesion Style)
        # Requires at least 4 classes: [Benign1, Benign2, Malignant1, Malignant2]
        if len(class_names) < 4:
            raise ValueError(
                f"Complex tree requires at least 4 classes, got {len(class_names)}"
            )

        # Distribute classes: First half -> Benign, Second half -> Malignant
        mid = len(class_names) // 2
        benign_classes = class_names[:mid]
        malignant_classes = class_names[mid:]

        # Split each group further
        benign_mid = len(benign_classes) // 2
        malignant_mid = len(malignant_classes) // 2

        benign_left = (
            benign_classes[:benign_mid] if benign_mid > 0 else [benign_classes[0]]
        )
        benign_right = benign_classes[benign_mid:] if benign_mid > 0 else benign_classes

        malignant_left = (
            malignant_classes[:malignant_mid]
            if malignant_mid > 0
            else [malignant_classes[0]]
        )
        malignant_right = (
            malignant_classes[malignant_mid:]
            if malignant_mid > 0
            else malignant_classes
        )

        # Build tree bottom-up
        # Level 2: Benign subtree (only if we can split)
        benign_subtree = None
        if len(benign_classes) > 1:
            benign_subtree = DecisionNode(
                node_id="node_benign",
                group_name="Benign Lesion Subtype",
                left_classes=benign_left,
                right_classes=benign_right,
            )

        # Level 2: Malignant subtree (only if we can split)
        malignant_subtree = None
        if len(malignant_classes) > 1:
            malignant_subtree = DecisionNode(
                node_id="node_malignant",
                group_name="Malignant Lesion Subtype",
                left_classes=malignant_left,
                right_classes=malignant_right,
            )

        # Level 1: Root (Benign vs Malignant)
        root = DecisionNode(
            node_id="root_complex",
            group_name="Lesion Classification",
            left_classes=benign_classes,
            right_classes=malignant_classes,
            left_child=benign_subtree,
            right_child=malignant_subtree,
        )

        return root


class MemoryArtifactStore(IArtifactStore):
    """
    An in-memory implementation of the artifact store.

    Perfect for:
    - Unit tests (no disk I/O, instant operations)
    - Local debugging (inspect stored artifacts in memory)
    - CI/CD pipelines (no cleanup required)

    Limitations:
    - Data lost when program terminates
    - No cross-process sharing
    - RAM-limited capacity
    """

    def __init__(self) -> None:
        """Initialize empty in-memory storage."""
        # Storage: artifact_id -> (Ensemble, Metadata)
        self._store: dict[str, tuple[PromptEnsemble, dict[str, Any]]] = {}

    def save_node_artifacts(
        self,
        node_id: str,
        ensemble: PromptEnsemble,
        metadata: dict[str, Any],
    ) -> str:
        """
        Saves artifacts to in-memory dictionary.

        Args:
            node_id: Unique identifier for this decision node
            ensemble: The trained PromptEnsemble
            metadata: Node-specific metadata

        Returns:
            artifact_id in format "mem://{node_id}/{short_uuid}"
        """
        # Generate a deterministic-ish artifact ID
        artifact_id = f"mem://{node_id}/{uuid.uuid4().hex[:8]}"

        # Deep copy not needed since we control the environment
        self._store[artifact_id] = (ensemble, metadata)

        return artifact_id

    def load_node_artifacts(
        self,
        artifact_id: str,
    ) -> tuple[PromptEnsemble, dict[str, Any]]:
        """
        Retrieves artifacts from in-memory dictionary.

        Args:
            artifact_id: The identifier returned by save_node_artifacts()

        Returns:
            (ensemble, metadata) tuple

        Raises:
            FileNotFoundError: If artifact_id doesn't exist in store
        """
        if artifact_id not in self._store:
            raise FileNotFoundError(
                f"Artifact '{artifact_id}' not found in MemoryStore. "
                f"Available: {list(self._store.keys())}"
            )

        return self._store[artifact_id]

    def list_artifacts(self) -> list[str]:
        """
        Helper for debugging to see what's saved.

        Returns:
            List of all artifact IDs in the store
        """
        return list(self._store.keys())

    def clear(self) -> None:
        """
        Clears all stored artifacts.

        Useful for test cleanup or resetting state between experiments.
        """
        self._store.clear()

    def size(self) -> int:
        """
        Returns the number of stored artifacts.

        Returns:
            Count of artifacts in store
        """
        return len(self._store)
