import pytest

from biomedxpro.core.domain import DecisionNode

# --- Happy Paths ---


def test_decision_node_creation_valid_leaf() -> None:
    """Verifies a standard leaf node can be created."""
    node = DecisionNode(
        node_id="leaf_1",
        group_name="Cystic Lesions",
        left_classes=["Simple Cyst"],
        right_classes=["Complex Cyst"],
        left_child=None,
        right_child=None,
    )
    assert node.is_leaf is True
    assert node.is_binary is False
    assert node.get_all_classes() == ["Simple Cyst", "Complex Cyst"]


def test_decision_node_creation_valid_branch() -> None:
    """Verifies a standard branch node connects to children."""
    child_l = DecisionNode("L", "L_Group", ["A"], ["B"])
    child_r = DecisionNode("R", "R_Group", ["C"], ["D"])

    node = DecisionNode(
        node_id="root",
        group_name="Root Group",
        left_classes=["A", "B"],
        right_classes=["C", "D"],
        left_child=child_l,
        right_child=child_r,
    )
    assert node.is_leaf is False
    assert node.left_child == child_l


# --- Invariant Enforcement (The Critical Validation Tests) ---


def test_validation_rejects_overlap() -> None:
    """
    CRITICAL: Ensures the system crashes immediately if classes overlap.
    This prevents 'impossible' classification tasks.
    """
    with pytest.raises(ValueError, match="must be disjoint"):
        DecisionNode(
            node_id="bad_node",
            group_name="Overlap",
            left_classes=["Tumor", "Cyst"],
            right_classes=["Tumor", "Stone"],  # 'Tumor' overlaps
        )


def test_validation_rejects_empty_sides() -> None:
    """Ensures we don't create degenerate nodes with nothing to classify."""
    with pytest.raises(ValueError, match="must be non-empty"):
        DecisionNode(
            node_id="empty_node",
            group_name="Empty",
            left_classes=[],  # Empty
            right_classes=["Stone"],
        )


def test_validation_rejects_asymmetric_children() -> None:
    """
    Ensures the tree structure is valid (either both children or neither).
    """
    child = DecisionNode("L", "G", ["A"], ["B"])

    with pytest.raises(
        ValueError, match="Must have either both children or no children"
    ):
        DecisionNode(
            node_id="broken_tree",
            group_name="Broken",
            left_classes=["A", "B"],
            right_classes=["C"],
            left_child=child,
            right_child=None,  # Missing sibling
        )


# --- Logic Tests ---


def test_binary_class_names_formatting() -> None:
    """
    Verifies the heuristic for naming the binary tasks.
    This ensures the LLM gets readable class names.
    """
    # Case 1: Single class names -> Use the name directly
    node_simple = DecisionNode("n1", "g", ["Cyst"], ["Tumor"])
    assert node_simple.get_binary_class_names() == ("Cyst", "Tumor")

    # Case 2: Multiple classes -> Use the first class + "_group" suffix
    node_complex = DecisionNode("n2", "g", ["Cyst", "Polyp"], ["Tumor", "Cancer"])
    assert node_complex.get_binary_class_names() == ("Cyst_group", "Tumor_group")
