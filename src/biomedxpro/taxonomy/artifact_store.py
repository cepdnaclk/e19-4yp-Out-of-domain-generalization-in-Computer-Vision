"""
File-system based artifact store for trained prompt ensembles.

Provides transparent, human-readable JSON persistence for hierarchical
taxonomy training results.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger

from biomedxpro.core.domain import PromptEnsemble
from biomedxpro.core.interfaces import IArtifactStore


class JSONArtifactStore(IArtifactStore):
    """
    A transparent, file-system based artifact store.
    Saves trained ensembles as human-readable JSON files.

    Directory Structure:
        base_path/
        ├── node_root/
        │   └── ensemble.json
        ├── node_left_benign/
        │   └── ensemble.json
        └── node_right_malignant/
            └── ensemble.json

    Each ensemble.json contains:
    {
        "node_id": "node_root",
        "metadata": {...},
        "ensemble": {
            "individuals": [...],
            "weights": [...]
        }
    }
    """

    def __init__(self, base_path: str):
        """
        Args:
            base_path: The root directory where artifacts will be saved.
                       e.g., "experiments/run_001/artifacts"
        """
        self.base_path = Path(base_path)
        # Ensure the directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Initialized JSONArtifactStore at {self.base_path}")

    def save_node_artifacts(
        self,
        node_id: str,
        ensemble: PromptEnsemble,
        metadata: dict[str, Any],
    ) -> str:
        """
        Saves the ensemble and metadata for a specific node.

        Structure:
            base_path/
            └── {node_id}/
                └── ensemble.json

        Args:
            node_id: Unique identifier for the decision node
            ensemble: The trained PromptEnsemble to persist
            metadata: Node-specific metadata (metrics, label mappings, etc.)

        Returns:
            The absolute path to the saved artifact (for use as artifact_id)

        Raises:
            IOError: If file write fails
            TypeError: If data cannot be serialized to JSON
        """
        # 1. Create Node Directory
        node_dir = self.base_path / node_id
        node_dir.mkdir(exist_ok=True)

        file_path = node_dir / "ensemble.json"

        # 2. Prepare Payload
        # We wrap the ensemble data and metadata into one clean JSON object
        payload = {
            "node_id": node_id,
            "metadata": metadata,
            # ensemble.to_dict() now produces pure JSON primitives
            "ensemble": ensemble.to_dict(),
        }

        # 3. Write to Disk
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)  # indent=2 for readability

            logger.debug(f"Saved artifact for node {node_id} at {file_path}")
            return str(file_path)

        except (IOError, TypeError) as e:
            logger.error(f"Failed to save artifact for {node_id}: {e}")
            raise

    def load_node_artifacts(
        self,
        artifact_id: str,
    ) -> tuple[PromptEnsemble, dict[str, Any]]:
        """
        Loads an ensemble from disk.

        Args:
            artifact_id: The full file path (or relative path inside base).
                        This is the string returned by save_node_artifacts().

        Returns:
            (ensemble, metadata) tuple

        Raises:
            FileNotFoundError: If the artifact does not exist
            ValueError: If the artifact file is corrupted or incompatible
        """
        file_path = Path(artifact_id)

        # Handle relative vs absolute paths
        if not file_path.is_absolute():
            # If it's just "experiments/...", resolve it relative to cwd
            file_path = Path.cwd() / file_path

        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

            # Reconstruct Domain Object
            # This relies on the updated from_dict methods
            ensemble = PromptEnsemble.from_dict(payload["ensemble"])
            metadata = payload["metadata"]

            logger.debug(
                f"Loaded artifact for node {payload['node_id']} from {file_path}"
            )
            return ensemble, metadata

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.error(f"Corrupted artifact file at {file_path}: {e}")
            raise ValueError(f"Invalid artifact format: {e}")

    def list_artifacts(self) -> list[str]:
        """
        Helper method to see what is currently stored.
        Returns a list of artifact_ids (file paths) for all saved ensembles.

        Returns:
            List of artifact file paths that can be passed to load_node_artifacts()
        """
        # Scans subdirectories for ensemble.json files
        artifacts = []
        # Use glob to find all ensemble.json files one level deep
        for path in self.base_path.glob("*/ensemble.json"):
            # Return the full path so it can be used directly with load_node_artifacts
            artifacts.append(str(path))
        return artifacts
