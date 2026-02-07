from typing import Any, Dict, List, cast

import pytest

from biomedxpro.core.domain import DecisionNode
from biomedxpro.core.interfaces import IArtifactStore
from biomedxpro.taxonomy.predictor import TaxonomicPredictor

# --- MOCKS ---


class RuleBasedEnsemble:
    """
    A deterministic mock that behaves like PromptEnsemble for testing.
    We rely on duck typing here, but for strict mypy in the predictor,
    we may need to cast it or treat the return from store as Any.
    """

    def __init__(self, rules: Dict[str, str], default: str) -> None:
        self.rules = rules
        self.default = default

    def predict(self, text: str) -> str:
        text_lower = text.lower()
        for keyword, label in self.rules.items():
            if keyword in text_lower:
                return label
        return self.default


class MockArtifactStore(IArtifactStore):
    """
    Simulates loading artifacts from disk.
    Strictly implements IArtifactStore interface.
    """

    def __init__(self) -> None:
        # We store the mocks as Any because they are test doubles (RuleBasedEnsemble)
        # and not actual PromptEnsemble objects.
        self.artifacts: Dict[str, Any] = {}
        self.load_count: int = 0

    def save_node_artifacts(
        self, node_id: str, artifact: Any, metadata: Dict[str, Any]
    ) -> str:
        # No-op for predictor tests
        return f"mock://{node_id}"

    def list_artifacts(self) -> List[str]:
        return list(self.artifacts.keys())

    def load_node_artifact(self, node_id: str, artifact_type: Any) -> Any:
        """
        Mocks the loading process.
        """
        self.load_count += 1
        if node_id in self.artifacts:
            return self.artifacts[node_id]
        raise FileNotFoundError(f"Artifact for {node_id} not found")

    def register_artifact(self, node_id: str, ensemble: RuleBasedEnsemble) -> None:
        """Helper to populate the store for tests."""
        self.artifacts[node_id] = ensemble


# --- FIXTURES ---


@pytest.fixture
def kidney_tree_structure() -> DecisionNode:
    """
    Constructs a valid asymmetric tree for testing.
    """
    # Level 1 Node: Lesion Classifier (Splits Cyst vs Tumor)
    lesion_node = DecisionNode(
        node_id="node_pathology",
        group_name="Pathology Type",
        left_classes=["Cyst"],
        right_classes=["Tumor"],
        ensemble_artifact_id="art_pathology",
    )

    # Root Node: Global Classifier (Splits Normal vs Lesion)
    root = DecisionNode(
        node_id="node_root",
        group_name="Global Check",
        left_classes=["Normal"],
        right_classes=["Cyst", "Tumor"],
        right_child=lesion_node,
        ensemble_artifact_id="art_root",
    )

    return root


@pytest.fixture
def populated_store() -> MockArtifactStore:
    store = MockArtifactStore()

    # Root Logic
    root_ensemble = RuleBasedEnsemble(
        rules={"sick": "Tumor", "fluid": "Cyst", "bad": "Tumor"}, default="Normal"
    )

    # Pathology Logic
    pathology_ensemble = RuleBasedEnsemble(
        rules={"fluid": "Cyst", "solid": "Tumor"}, default="Tumor"
    )

    store.register_artifact("node_root", root_ensemble)
    store.register_artifact("node_pathology", pathology_ensemble)
    return store


# --- TESTS ---


def test_inference_happy_path_leaf(
    kidney_tree_structure: DecisionNode, populated_store: MockArtifactStore
) -> None:
    """
    Scenario: Input is 'healthy'.
    Expectation: Root predicts 'Normal' (Left Branch).
    """
    predictor = TaxonomicPredictor(kidney_tree_structure, populated_store)

    result = predictor.predict("Patient is healthy")

    assert result["status"] == "success"
    assert result["final_class"] == "Normal"
    # Ensure 'path' exists and is a list
    path = cast(List[Dict[str, str]], result.get("path", []))
    assert len(path) == 1
    assert path[0]["node_id"] == "node_root"
    assert path[0]["decision"] == "Normal"


def test_inference_happy_path_deep(
    kidney_tree_structure: DecisionNode, populated_store: MockArtifactStore
) -> None:
    """
    Scenario: Input is 'fluid filled sac'.
    Expectation: Root -> Cyst (Right) -> Pathology -> Cyst (Left).
    """
    predictor = TaxonomicPredictor(kidney_tree_structure, populated_store)

    result = predictor.predict("Mass is a fluid filled sac")

    assert result["status"] == "success"
    assert result["final_class"] == "Cyst"

    path = cast(List[Dict[str, str]], result.get("path", []))
    assert len(path) == 2
    assert path[0]["node_id"] == "node_root"
    assert path[1]["node_id"] == "node_pathology"
    assert path[1]["decision"] == "Cyst"


def test_lazy_loading_efficiency(
    kidney_tree_structure: DecisionNode, populated_store: MockArtifactStore
) -> None:
    """
    Scenario: Run inference on the Left Branch (Normal).
    Expectation: Only the Root artifact is loaded.
    """
    predictor = TaxonomicPredictor(kidney_tree_structure, populated_store)

    # 1. First Run (Normal)
    predictor.predict("healthy patient")
    assert populated_store.load_count == 1

    # 2. Second Run (Normal again)
    predictor.predict("very healthy patient")
    assert populated_store.load_count == 1

    # 3. Third Run (Deep - Cyst)
    predictor.predict("fluid mass")
    assert populated_store.load_count == 2


def test_hallucination_handling(
    kidney_tree_structure: DecisionNode, populated_store: MockArtifactStore
) -> None:
    """
    Scenario: Model predicts a label that is NOT in the node's definition.
    """
    broken_ensemble = RuleBasedEnsemble(
        rules={"weird": "AlienSpecies"}, default="Normal"
    )
    populated_store.register_artifact("node_root", broken_ensemble)

    predictor = TaxonomicPredictor(kidney_tree_structure, populated_store)

    result = predictor.predict("weird scan")

    assert result["status"] == "error"
    assert "Invalid prediction 'AlienSpecies'" in str(result.get("error", ""))
    assert result["final_class"] == "Unknown"


def test_missing_artifact_handling(
    kidney_tree_structure: DecisionNode, populated_store: MockArtifactStore
) -> None:
    """
    Scenario: Tree structure exists, but artifact file is missing.
    """
    # Remove pathology artifact from store
    del populated_store.artifacts["node_pathology"]

    predictor = TaxonomicPredictor(kidney_tree_structure, populated_store)

    # Trigger deep traversal
    result = predictor.predict("fluid mass")

    assert result["status"] == "error"
    assert "Artifact for node_pathology not found" in str(result.get("error", ""))
    # Path should contain the successful root step
    path = cast(List[Dict[str, str]], result.get("path", []))
    assert len(path) == 1
