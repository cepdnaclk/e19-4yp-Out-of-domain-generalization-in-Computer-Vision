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
        task_definition: TaskDefinition,  # UPDATED: Accepts full TaskDefinition
    ) -> DecisionNode:
        """
        Orchestrates the creation of the decision tree.
        """
        logger.info(f"Architecting taxonomy for task: '{task_definition.task_name}'")
        logger.debug(
            f"Input Classes ({len(task_definition.class_names)}): {task_definition.class_names}"
        )

        # 1. Render Prompt with Task Context
        prompt = self.template.render(task=task_definition)

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

        # 5. The "Senior Architect" Check (Set Theory Validation)
        # We ensure the tree perfectly covers the Task's class list
        logger.info("Validating taxonomy integrity...")
        self._verify_bijective_integrity(root_node, set(task_definition.class_names))

        logger.success(
            f"Taxonomy constructed successfully. Root: {root_node.group_name}"
        )
        return root_node

    def _clean_json_markdown(self, text: str) -> str:
        """
        Strips markdown code blocks (```json ... ```) if present.
        """
        pattern = r"```(?:json)?\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return text.strip()

    def _deserialize_node(self, data: dict[str, Any]) -> DecisionNode:
        """
        Recursive factory to convert raw dictionary -> DecisionNode object.
        """
        # 1. Extract Fields
        node_id = data.get("node_id")
        group_name = data.get("group_name")
        left_classes = data.get("left_classes", [])
        right_classes = data.get("right_classes", [])

        if not node_id or not group_name:
            raise ValueError("Node missing required fields 'node_id' or 'group_name'")

        # 2. Recursively Build Children
        children_data = data.get("children")
        left_child = None
        right_child = None

        if children_data:
            # If "left" is present and not null, recurse
            if children_data.get("left"):
                left_child = self._deserialize_node(children_data["left"])

            # If "right" is present and not null, recurse
            if children_data.get("right"):
                right_child = self._deserialize_node(children_data["right"])

        # 3. Construct Node
        # Note: __post_init__ in DecisionNode handles overlap/empty checks
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
        Recursively verifies that:
        1. The node covers exactly the expected universe of classes.
        2. The children (if any) cover exactly the classes assigned to their branch.
        """
        # 1. Check current node's coverage
        current_coverage = set(node.left_classes) | set(node.right_classes)

        if current_coverage != expected_universe:
            missing = expected_universe - current_coverage
            extra = current_coverage - expected_universe
            raise ValueError(
                f"Taxonomy Integrity Error at Node '{node.node_id}':\n"
                f" - Expected: {expected_universe}\n"
                f" - Got: {current_coverage}\n"
                f" - Missing: {missing}\n"
                f" - Extra (Hallucinated): {extra}"
            )

        # 2. Check Left Child Consistency
        if node.left_child:
            self._verify_bijective_integrity(node.left_child, set(node.left_classes))
        elif len(node.left_classes) > 0 and node.is_binary:
            # Logic Check: If it's a binary node, but left child is missing
            # despite having classes assigned, that's ambiguous structure.
            # However, DecisionNode logic allows leaves.
            pass

        # 3. Check Right Child Consistency
        if node.right_child:
            self._verify_bijective_integrity(node.right_child, set(node.right_classes))
