import json
import re
from pathlib import Path
from typing import Any, Set

from jinja2 import Template
from loguru import logger

from biomedxpro.core.domain import DecisionNode, TaskDefinition
from biomedxpro.core.interfaces import ILLMClient, ITaxonomyBuilder


class LLMTaxonomyBuilder(ITaxonomyBuilder):
    """
    Concrete implementation of the Architect.
    Uses an LLM to discover the optimal hierarchical structure for a given Task.
    """

    def __init__(
        self,
        llm_client: ILLMClient,
        prompt_template_path: str = "src/biomedxpro/prompts/taxonomy_v1.j2",
    ):
        self.llm = llm_client
        self.template_path = Path(prompt_template_path)

        if not self.template_path.exists():
            raise FileNotFoundError(
                f"Taxonomy prompt template not found at: {self.template_path.absolute()}"
            )

        with open(self.template_path, "r", encoding="utf-8") as f:
            self.template = Template(f.read())

    def build_taxonomy(
        self,
        task_definition: TaskDefinition,
    ) -> DecisionNode:
        """
        Orchestrates the creation of the decision tree.
        """
        logger.info(f"Architecting taxonomy for task: '{task_definition.task_name}'")

        # 1. Render Prompt with Task Context
        prompt = self.template.render(task_definition=task_definition)

        # 2. Query LLM
        logger.info("Querying LLM for structural analysis...")
        raw_response = self.llm.generate(prompt)

        # 3. Clean & Parse JSON
        try:
            cleaned_json = self._clean_json_markdown(raw_response)
            tree_dict = json.loads(cleaned_json)
        except json.JSONDecodeError as e:
            logger.error("LLM returned malformed JSON.")
            logger.debug(f"Raw Response: {raw_response}")
            raise ValueError(f"Taxonomy Generation Failed: {e}")

        # 4. Deserialize & Construct Domain Objects
        try:
            root_node = self._deserialize_node(tree_dict)
        except Exception as e:
            logger.error(f"Failed to construct DecisionNode tree: {e}")
            raise

        # 5. Validation
        logger.info("Validating taxonomy integrity...")
        self._verify_bijective_integrity(root_node, set(task_definition.class_names))

        # 6. CRITICAL: Verify Leafs are Pure (Singleton)
        self._verify_pure_leaves(root_node)

        logger.success(
            f"Taxonomy constructed successfully. Root: {root_node.group_name}"
        )
        return root_node

    def _clean_json_markdown(self, text: str) -> str:
        """Strips markdown code blocks (```json ... ```) if present."""
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def _deserialize_node(self, data: dict[str, Any]) -> DecisionNode:
        """
        Recursive factory to convert raw dictionary -> DecisionNode object.
        """
        node_id = data.get("node_id")
        group_name = data.get("group_name")
        left_classes = data.get("left_classes", [])
        right_classes = data.get("right_classes", [])

        if not node_id or not group_name:
            raise ValueError("Node missing required fields 'node_id' or 'group_name'")

        children_data = data.get("children")
        left_child = None
        right_child = None

        if children_data:
            # If JSON has "left": {...}, recurse. If "left": null, keep it None.
            if children_data.get("left"):
                left_child = self._deserialize_node(children_data["left"])

            # If JSON has "right": {...}, recurse. If "right": null, keep it None.
            if children_data.get("right"):
                right_child = self._deserialize_node(children_data["right"])

        return DecisionNode(
            node_id=node_id,
            group_name=group_name,
            left_classes=left_classes,
            right_classes=right_classes,
            left_child=left_child,
            right_child=right_child,
        )

    def _verify_bijective_integrity(
        self, node: DecisionNode, expected_universe: Set[str]
    ) -> None:
        """
        Verifies that the tree perfectly covers the input classes.
        """
        current_coverage = set(node.left_classes) | set(node.right_classes)

        if current_coverage != expected_universe:
            missing = expected_universe - current_coverage
            extra = current_coverage - expected_universe
            raise ValueError(
                f"Taxonomy Integrity Error at Node '{node.node_id}':\n"
                f" - Expected: {expected_universe}\n"
                f" - Got: {current_coverage}\n"
                f" - Missing: {missing}\n"
                f" - Extra: {extra}"
            )

        if node.left_child:
            self._verify_bijective_integrity(node.left_child, set(node.left_classes))

        if node.right_child:
            self._verify_bijective_integrity(node.right_child, set(node.right_classes))

    def _verify_pure_leaves(self, node: DecisionNode) -> None:
        """
        Ensures that every branch ends in a SINGLE class.
        If a branch stops (child is None) but has > 1 class, the taxonomy is incomplete.
        """
        # Check Left Branch
        if node.left_child is None:
            # If no child, we must have exactly 1 class
            if len(node.left_classes) != 1:
                raise ValueError(
                    f"Taxonomy Error at Node '{node.node_id}': "
                    f"Left branch is a leaf but contains {len(node.left_classes)} classes "
                    f"{node.left_classes}. A leaf must contain exactly 1 class."
                )
        else:
            # Recurse
            self._verify_pure_leaves(node.left_child)

        # Check Right Branch
        if node.right_child is None:
            # If no child, we must have exactly 1 class
            if len(node.right_classes) != 1:
                raise ValueError(
                    f"Taxonomy Error at Node '{node.node_id}': "
                    f"Right branch is a leaf but contains {len(node.right_classes)} classes "
                    f"{node.right_classes}. A leaf must contain exactly 1 class."
                )
        else:
            # Recurse
            self._verify_pure_leaves(node.right_child)
