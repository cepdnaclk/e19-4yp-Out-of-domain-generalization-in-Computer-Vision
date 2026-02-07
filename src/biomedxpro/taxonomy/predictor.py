"""
Taxonomic Predictor - Hierarchical Inference Engine.

This module implements the inference logic for a trained decision tree.
It traverses the tree from root to leaf, lazily loading node artifacts
only as needed for the specific path a sample takes.
"""

from typing import Any, Dict, List

from loguru import logger

from biomedxpro.core.domain import DecisionNode, PromptEnsemble
from biomedxpro.core.interfaces import IArtifactStore


class TaxonomicPredictor:
    """
    Routes input samples through a trained DecisionNode tree.

    Features:
    - Lazy Loading: Only loads artifacts for nodes visited during inference.
    - Path Tracing: Returns the full decision path, not just the final label.
    - Robust Routing: Validates model predictions against the tree structure.
    """

    def __init__(self, root_node: DecisionNode, artifact_store: IArtifactStore):
        """
        Args:
            root_node: The root of the trained taxonomy.
            artifact_store: The storage backend to load node artifacts from.
        """
        self.root = root_node
        self.store = artifact_store
        # Cache loaded ensembles to avoid disk I/O on repeated inference
        # Key: node_id, Value: PromptEnsemble
        self._ensemble_cache: Dict[str, PromptEnsemble] = {}

    def predict(self, input_text: str) -> Dict[str, Any]:
        """
        Classifies a single text input by traversing the tree.

        Args:
            input_text: The clinical text or prompt to classify.

        Returns:
            Dictionary containing:
            - final_class: The leaf label reached.
            - path: List of decision steps taken.
            - status: "success" or "error".
        """
        current_node = self.root
        path_trace: List[Dict[str, str]] = []

        logger.info(f"Starting inference traversal from root: {self.root.node_id}")

        while True:
            # 1. Base Case: Are we at a Semantic Leaf?
            # If the node represents a single class (e.g. "Normal"), we are done.
            # This handles "Implicit Leaves" where the tree structure ends but
            # the node still has a label.
            all_classes = current_node.get_all_classes()
            if len(all_classes) <= 1:
                final_label = all_classes[0] if all_classes else "Unknown"
                logger.info(f"🏁 Reached Leaf: {final_label}")
                return {
                    "final_class": final_label,
                    "path": path_trace,
                    "status": "success",
                }

            # 2. Lazy Load the Model
            # If we haven't loaded this node's ensemble yet, fetch it.
            if current_node.node_id not in self._ensemble_cache:
                if not current_node.ensemble_artifact_id:
                    error_msg = f"Node {current_node.node_id} has no artifact! Tree is untrained."
                    logger.error(error_msg)
                    return {"status": "error", "error": error_msg, "path": path_trace}

                logger.debug(f"📂 Loading artifact for {current_node.node_id}...")
                try:
                    ensemble = self.store.load_node_artifact(
                        current_node.node_id, PromptEnsemble
                    )
                    self._ensemble_cache[current_node.node_id] = ensemble
                except Exception as e:
                    error_msg = (
                        f"Failed to load artifact for {current_node.node_id}: {e}"
                    )
                    logger.critical(error_msg)
                    return {"status": "error", "error": error_msg, "path": path_trace}

            ensemble = self._ensemble_cache[current_node.node_id]

            # 3. Execute Prediction
            # The ensemble returns a string label (e.g. "Cyst" or "Tumor")
            try:
                prediction = ensemble.predict(input_text)
            except Exception as e:
                return {
                    "status": "error",
                    "error": f"Model inference failed at {current_node.node_id}: {e}",
                    "path": path_trace,
                }

            logger.info(f"⚡ Node {current_node.node_id} predicted: {prediction}")

            path_trace.append(
                {
                    "node_id": current_node.node_id,
                    "group": current_node.group_name,
                    "decision": prediction,
                }
            )

            # 4. Route to Next Step
            # We must determine if the prediction directs us to the Left or Right branch.
            if prediction in current_node.left_classes:
                # Go Left
                if current_node.left_child:
                    current_node = current_node.left_child
                else:
                    # Implicit Leaf: Logic says "Left", but no child object exists.
                    # This means the "Left" concept is a terminal leaf (e.g. "Normal").
                    return {
                        "final_class": prediction,
                        "path": path_trace,
                        "status": "success",
                    }

            elif prediction in current_node.right_classes:
                # Go Right
                if current_node.right_child:
                    current_node = current_node.right_child
                else:
                    # Implicit Leaf: Logic says "Right", but no child object exists.
                    return {
                        "final_class": prediction,
                        "path": path_trace,
                        "status": "success",
                    }

            else:
                # Hallucination / Schema Violation
                # The model predicted something that isn't in this node's left or right definition.
                error_msg = (
                    f"Invalid prediction '{prediction}' at node {current_node.node_id}. "
                    f"Expected one of: {current_node.left_classes + current_node.right_classes}"
                )
                logger.warning(error_msg)
                return {
                    "final_class": "Unknown",
                    "status": "error",
                    "error": error_msg,
                    "path": path_trace,
                }
