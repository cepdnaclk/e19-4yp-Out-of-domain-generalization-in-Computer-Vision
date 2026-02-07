import json
from pathlib import Path
from typing import Any

import pytest

from biomedxpro.core.domain import TaskDefinition
from biomedxpro.core.interfaces import ILLMClient
from biomedxpro.taxonomy.builder import LLMTaxonomyBuilder


class MockLLMClient(ILLMClient):
    def __init__(self) -> None:
        self.response: str = ""

    def generate(self, prompt: str) -> str:
        return self.response


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def temp_template(tmp_path: Path) -> str:
    template_file = tmp_path / "test_taxonomy.jinja2"
    template_file.write_text("Mock Prompt")
    return str(template_file)


@pytest.fixture
def valid_task() -> TaskDefinition:
    return TaskDefinition(
        task_name="Kidney",
        image_modality="CT",
        class_names=["Normal", "Cyst", "Tumor", "Stone"],
        role="Rad",
        concepts=None,
    )


@pytest.fixture
def builder(mock_llm: MockLLMClient, temp_template: str) -> LLMTaxonomyBuilder:
    return LLMTaxonomyBuilder(mock_llm, prompt_template_path=temp_template)


# --- TESTS ---


def test_builder_constructs_valid_tree(
    builder: LLMTaxonomyBuilder, mock_llm: MockLLMClient, valid_task: TaskDefinition
) -> None:
    """
    Happy Path: Valid tree where every leaf is a singleton.
    """
    valid_json: dict[str, Any] = {
        "node_id": "root",
        "group_name": "Global",
        "left_classes": ["Normal"],  # Singleton Leaf (Implicit)
        "right_classes": ["Cyst", "Tumor", "Stone"],  # <--- Added Stone here
        "children": {
            "left": None,
            "right": {
                "node_id": "pathology",
                "group_name": "Path",
                "left_classes": ["Cyst"],  # Singleton Leaf
                "right_classes": ["Tumor", "Stone"],  # <--- Stone flows down
                "children": {
                    "left": None,
                    "right": {
                        "node_id": "solid_vs_stone",
                        "group_name": "Solid Split",
                        "left_classes": ["Tumor"],
                        "right_classes": ["Stone"],  # <--- Stone reaches leaf
                        "children": None,
                    },
                },
            },
        },
    }

    mock_llm.response = json.dumps(valid_json)
    root = builder.build_taxonomy(valid_task)

    assert root.node_id == "root"
    assert root.left_child is None  # "Normal" is a terminal leaf at this split
    # Verify deep structure
    assert root.right_child is not None
    assert root.right_child.node_id == "pathology"
    assert root.right_child.right_child is not None
    assert root.right_child.right_child.node_id == "solid_vs_stone"


def test_validation_rejects_multiclass_leaf(
    builder: LLMTaxonomyBuilder, mock_llm: MockLLMClient, valid_task: TaskDefinition
) -> None:
    """
    CRITICAL TEST: Ensures we reject "Lazy Leaves" with > 1 class.
    """
    bad_json = {
        "node_id": "root",
        "group_name": "Global",
        "left_classes": ["Normal"],
        "right_classes": ["Cyst", "Tumor", "Stone"],
        "children": {
            "left": None,
            "right": {
                "node_id": "pathology",
                "group_name": "Path",
                "left_classes": ["Cyst"],
                "right_classes": ["Tumor", "Stone"],  # <--- 2 classes here!
                "children": None,  # <--- But no children to split them!
            },
        },
    }
    mock_llm.response = json.dumps(bad_json)

    with pytest.raises(ValueError, match="A leaf must contain exactly 1 class"):
        builder.build_taxonomy(valid_task)


def test_validation_catches_child_mismatch(
    builder: LLMTaxonomyBuilder, mock_llm: MockLLMClient, valid_task: TaskDefinition
) -> None:
    bad_json = {
        "node_id": "root",
        "group_name": "R",
        "left_classes": ["Normal"],
        "right_classes": ["Cyst", "Tumor", "Stone"],
        "children": {
            "left": None,
            "right": {
                "node_id": "child",
                "group_name": "C",
                "left_classes": ["Cyst"],
                "right_classes": [
                    "Stone"
                ],  # Mismatch: Parent said ["Tumor", "Stone"], Child says ["Stone"] (Tumor missing)
                "children": None,
            },
        },
    }
    mock_llm.response = json.dumps(bad_json)

    with pytest.raises(ValueError, match="Taxonomy Integrity Error"):
        builder.build_taxonomy(valid_task)
