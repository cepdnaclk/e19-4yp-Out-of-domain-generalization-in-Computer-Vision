"""
Test cases for JSONArtifactStore.

Validates JSON serialization, file I/O, and artifact reconstruction.
"""

import json
from pathlib import Path

import pytest
import torch

from biomedxpro.core.domain import (
    CreationOperation,
    Individual,
    PromptEnsemble,
    PromptGenotype,
)
from biomedxpro.taxonomy.artifact_store import JSONArtifactStore


class TestJSONArtifactStore:
    """Test suite for file-based artifact persistence."""

    @pytest.fixture
    def temp_store(self, tmp_path: Path) -> JSONArtifactStore:
        """Creates a temporary artifact store."""
        return JSONArtifactStore(str(tmp_path / "artifacts"))

    @pytest.fixture
    def sample_ensemble(self) -> PromptEnsemble:
        """Creates a sample PromptEnsemble for testing."""
        # Create individuals
        ind1 = Individual(
            genotype=PromptGenotype(prompts=("prompt_left", "prompt_right")),
            generation_born=1,
            operation=CreationOperation.INITIALIZATION,
            concept="TestConcept",
        )
        ind1.update_metrics(
            {
                "soft_f1_macro": 0.95,
                "accuracy": 0.93,
                "f1_macro": 0.94,
                "auc": 0.96,
                "f1_weighted": 0.94,
                "inverted_bce": 0.92,
            }
        )

        ind2 = Individual(
            genotype=PromptGenotype(prompts=("prompt_a", "prompt_b")),
            generation_born=2,
            operation=CreationOperation.LLM_MUTATION,
            concept="TestConcept",
            parents=[ind1.id],
        )
        ind2.update_metrics(
            {
                "soft_f1_macro": 0.97,
                "accuracy": 0.95,
                "f1_macro": 0.96,
                "auc": 0.98,
                "f1_weighted": 0.96,
                "inverted_bce": 0.94,
            }
        )

        # Create ensemble
        return PromptEnsemble(
            experts=[ind1, ind2],
            weights=torch.tensor([0.4, 0.6]),
            metric="soft_f1_macro",
        )

    @pytest.fixture
    def sample_metadata(self) -> dict:
        """Sample node metadata."""
        return {
            "node_id": "test_node",
            "group_name": "Test Group",
            "left_classes": ["ClassA"],
            "right_classes": ["ClassB"],
            "best_metrics": {"f1_macro": 0.95},
        }

    def test_save_and_load_ensemble(
        self,
        temp_store: JSONArtifactStore,
        sample_ensemble: PromptEnsemble,
        sample_metadata: dict,
    ) -> None:
        """Test basic save and load cycle."""
        # Save
        artifact_id = temp_store.save_node_artifacts(
            node_id="test_node", ensemble=sample_ensemble, metadata=sample_metadata
        )

        # Verify file exists
        assert Path(artifact_id).exists()

        # Load
        loaded_ensemble, loaded_metadata = temp_store.load_node_artifacts(artifact_id)

        # Verify metadata
        assert loaded_metadata == sample_metadata

        # Verify ensemble structure
        assert len(loaded_ensemble.experts) == 2
        assert loaded_ensemble.weights.shape == torch.Size([2])
        assert torch.allclose(loaded_ensemble.weights, sample_ensemble.weights)

    def test_individual_serialization(self, sample_ensemble: PromptEnsemble) -> None:
        """Test that Individual.to_dict produces valid JSON."""
        ind = sample_ensemble.experts[0]
        serialized = ind.to_dict()

        # Verify no raw objects remain
        assert isinstance(serialized["genotype"], dict)
        assert isinstance(serialized["operation"], str)
        assert serialized["operation"] == "initialization"

        # Verify JSON-serializable
        json_str = json.dumps(serialized)
        assert len(json_str) > 0

    def test_individual_round_trip(self, sample_ensemble: PromptEnsemble) -> None:
        """Test Individual serialization and deserialization."""
        original = sample_ensemble.experts[1]  # The mutated one

        # Serialize
        serialized = original.to_dict()

        # Deserialize
        reconstructed = Individual.from_dict(serialized)

        # Verify fields
        assert str(reconstructed.id) == str(original.id)
        assert reconstructed.genotype.prompts == original.genotype.prompts
        assert reconstructed.generation_born == original.generation_born
        assert reconstructed.operation == original.operation
        assert reconstructed.concept == original.concept
        assert [str(p) for p in reconstructed.parents] == [
            str(p) for p in original.parents
        ]

        # Verify metrics were restored
        assert reconstructed.is_evaluated
        assert reconstructed.metrics == original.metrics

    def test_ensemble_round_trip(self, sample_ensemble: PromptEnsemble) -> None:
        """Test PromptEnsemble serialization and deserialization."""
        # Serialize
        serialized = sample_ensemble.to_dict()

        # Verify JSON-serializable
        json_str = json.dumps(serialized)
        assert len(json_str) > 0

        # Deserialize
        reconstructed = PromptEnsemble.from_dict(serialized)

        # Verify structure
        assert len(reconstructed.experts) == len(sample_ensemble.experts)
        assert torch.allclose(reconstructed.weights, sample_ensemble.weights)

        # Verify individuals
        for orig, recon in zip(
            sample_ensemble.experts, reconstructed.experts, strict=False
        ):
            assert recon.genotype.prompts == orig.genotype.prompts
            assert recon.metrics == orig.metrics

    def test_list_artifacts(
        self,
        temp_store: JSONArtifactStore,
        sample_ensemble: PromptEnsemble,
        sample_metadata: dict,
    ) -> None:
        """Test artifact listing functionality."""
        # Initially empty
        assert len(temp_store.list_artifacts()) == 0

        # Save multiple nodes
        temp_store.save_node_artifacts("node_1", sample_ensemble, sample_metadata)
        temp_store.save_node_artifacts("node_2", sample_ensemble, sample_metadata)
        temp_store.save_node_artifacts("node_3", sample_ensemble, sample_metadata)

        # Verify list
        artifacts = temp_store.list_artifacts()
        assert len(artifacts) == 3

        # All should be loadable
        for artifact_id in artifacts:
            ensemble, metadata = temp_store.load_node_artifacts(artifact_id)
            assert len(ensemble.experts) == 2

    def test_load_nonexistent_artifact(self, temp_store: JSONArtifactStore) -> None:
        """Test error handling for missing artifacts."""
        with pytest.raises(FileNotFoundError):
            temp_store.load_node_artifacts("nonexistent/path/ensemble.json")

    def test_corrupted_artifact_handling(self, temp_store: JSONArtifactStore) -> None:
        """Test error handling for corrupted JSON files."""
        # Create a corrupted file
        node_dir = temp_store.base_path / "corrupt_node"
        node_dir.mkdir(exist_ok=True)
        corrupt_file = node_dir / "ensemble.json"

        # Write invalid JSON
        corrupt_file.write_text("{ invalid json", encoding="utf-8")

        # Attempt to load should raise ValueError
        with pytest.raises(ValueError, match="Invalid artifact format"):
            temp_store.load_node_artifacts(str(corrupt_file))

    def test_nested_enum_serialization(self) -> None:
        """Test that CreationOperation enum values serialize correctly."""
        for op in CreationOperation:
            ind = Individual(
                genotype=PromptGenotype(prompts=("p1", "p2")),
                generation_born=1,
                operation=op,
                concept="Test",
            )

            # Serialize
            serialized = ind.to_dict()
            assert isinstance(serialized["operation"], str)
            assert serialized["operation"] == op.value

            # Deserialize
            reconstructed = Individual.from_dict(serialized)
            assert reconstructed.operation == op

    def test_file_structure(
        self,
        temp_store: JSONArtifactStore,
        sample_ensemble: PromptEnsemble,
        sample_metadata: dict,
    ) -> None:
        """Test the directory structure created by the store."""
        artifact_id = temp_store.save_node_artifacts(
            node_id="my_node", ensemble=sample_ensemble, metadata=sample_metadata
        )

        # Verify structure: base_path/my_node/ensemble.json
        path = Path(artifact_id)
        assert path.name == "ensemble.json"
        assert path.parent.name == "my_node"
        assert path.parent.parent == temp_store.base_path

    def test_overwrite_artifact(
        self,
        temp_store: JSONArtifactStore,
        sample_ensemble: PromptEnsemble,
        sample_metadata: dict,
    ) -> None:
        """Test that saving to the same node_id overwrites the previous artifact."""
        # Save first version
        artifact_id_1 = temp_store.save_node_artifacts(
            "node_x", sample_ensemble, {"version": 1}
        )

        # Save second version (should overwrite)
        artifact_id_2 = temp_store.save_node_artifacts(
            "node_x", sample_ensemble, {"version": 2}
        )

        # Same path
        assert artifact_id_1 == artifact_id_2

        # Load should return version 2
        _, metadata = temp_store.load_node_artifacts(artifact_id_2)
        assert metadata["version"] == 2

    def test_empty_ensemble(self, temp_store: JSONArtifactStore) -> None:
        """Test handling of edge case: empty ensemble."""
        empty_ensemble = PromptEnsemble(
            experts=[], weights=torch.tensor([]), metric="soft_f1_macro"
        )

        artifact_id = temp_store.save_node_artifacts(
            "empty_node", empty_ensemble, {"empty": True}
        )

        # Should be able to load
        loaded, metadata = temp_store.load_node_artifacts(artifact_id)
        assert len(loaded.experts) == 0
        assert loaded.weights.numel() == 0
        assert metadata["empty"] is True
