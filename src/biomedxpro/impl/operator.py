import json
import math
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List, Sequence, TypedDict

from jinja2 import Template
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from biomedxpro.core.domain import (
    CreationOperation,
    Individual,
    MetricName,
    PromptGenotype,
    TaskDefinition,
)
from biomedxpro.core.interfaces import ILLMClient, IOperator
from biomedxpro.utils.config import PromptStrategy
from biomedxpro.utils.logging import loguru_before_sleep


class ParentViewModel(TypedDict):
    score: int
    genotype: PromptGenotype


class LLMOperator(IOperator):
    """
    An LLM-driven operator that evolves prompts.
    It implements 'View Models' to normalize fitness scores for the LLM
    and strictly adheres to the JSON output format defined in the templates.
    """

    def __init__(
        self, llm: ILLMClient, strategy: PromptStrategy, task_def: TaskDefinition
    ) -> None:
        self.llm = llm
        self.strategy = strategy
        self.task_def = task_def

        self._validate_paths()

    def _validate_paths(self) -> None:
        """Fail fast if templates are missing."""
        paths = [
            self.strategy.discover_concepts_template_path,
            self.strategy.init_template_path,
            self.strategy.mutation_template_path,
        ]
        for p in paths:
            if not Path(p).exists():
                logger.error(f"Template missing at: {p}")
                raise FileNotFoundError(f"Template not found: {p}")

    def _render(self, path: str, context: Dict[str, Any] = {}) -> str:
        """Loads and renders a Jinja2 template."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                template = Template(f.read())

            # Inject the task definition into every template context
            full_context = {**context, "task_definition": self.task_def}
            return template.render(**full_context)
        except Exception as e:
            logger.error(f"Template rendering failed for {path}: {e}")
            raise e

    def _parse_llm_json(self, raw_response: str) -> Any:
        """
        Extracts JSON from LLM response and loads it.
        Handles markdown code blocks with ```json fences using regex.
        Falls back to parsing raw JSON if no code block is found.
        """

        text = raw_response.strip()

        # Try to extract JSON from markdown code blocks using regex
        # Matches ```json ... ``` or ``` ... ```
        pattern = r"```(?:json)?\s*\n?(.*?)\n?```"
        match = re.search(pattern, text, re.DOTALL)

        if match:
            text = match.group(1).strip()
        # If no code block, use the raw text

        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"JSON Parse Error: {e}. Raw output: {raw_response[:200]}...")
            raise e

    def _normalize_scores_to_int(
        self, parents: Sequence[Individual], metric: MetricName
    ) -> List[int]:
        """
        Normalizes raw float fitness scores to Integers between 60 and 90.
        This helps the LLM perceive differences more clearly (human-readable scale).
        """
        if not parents:
            return []

        raw_scores = [p.get_fitness(metric) for p in parents]
        min_s, max_s = min(raw_scores), max(raw_scores)

        # Avoid division by zero if all scores are identical
        if math.isclose(min_s, max_s, abs_tol=1e-9):
            return [85] * len(parents)  # Return a generic 'good' score

        # Linear Mapping: 60 + (score - min) * (30 / range)
        normalized = []
        for s in raw_scores:
            norm = 60 + (s - min_s) * (30 / (max_s - min_s))
            normalized.append(int(round(norm)))

        return normalized

    def _create_parent_view_models(
        self, parents: Sequence[Individual], metric: MetricName
    ) -> List[ParentViewModel]:
        """
        Transforms Domain Entities (Individuals) into View Models for the Template.
        """
        norm_scores: List[int] = self._normalize_scores_to_int(parents, metric)
        view_models: List[ParentViewModel] = []

        for parent, score in zip(parents, norm_scores):
            genotype_data = parent.genotype

            vm: ParentViewModel = {
                "score": score,
                "genotype": genotype_data,
            }
            view_models.append(vm)

        # Sort by score ascending (lowest to highest)
        view_models.sort(key=lambda x: x["score"])

        return view_models

    _retry_policy = retry(
        wait=wait_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
        before_sleep=loguru_before_sleep,
        reraise=True,
    )

    @_retry_policy
    def discover_concepts(self) -> list[str]:
        prompt = self._render(self.strategy.discover_concepts_template_path)

        logger.info("Discovering concepts...")
        response = self.llm.generate(prompt)
        concepts = self._parse_llm_json(response)

        if not isinstance(concepts, list):
            logger.error(
                f"Concept discovery returned non-list response.\nResponse:\n{response}"
            )
            raise ValueError("Concept discovery did not return a list.")

        # Sanitize strings
        return [str(c).strip() for c in concepts if isinstance(c, str)]

    @_retry_policy
    def initialize_population(
        self, num_offsprings: int, concept: str
    ) -> Sequence[Individual]:
        prompt = self._render(
            self.strategy.init_template_path,
            context={"concept": concept, "num_offsprings": num_offsprings},
        )

        response = self.llm.generate(prompt)
        data_list: list[list[str]] = self._parse_llm_json(response)

        offspring = []
        if isinstance(data_list, list):
            for data in data_list:
                if (not isinstance(data, list)) or len(data) != 2:
                    logger.warning(
                        f"Skipping invalid individual data during initialization for concept '{concept}': {data}"
                    )
                    continue

                ind = Individual(
                    id=uuid.uuid4(),
                    # We store the structured pair in genotype
                    genotype=PromptGenotype(
                        negative_prompt=data[0],
                        positive_prompt=data[1],
                    ),
                    generation_born=0,
                    parents=[],
                    concept=concept,
                    operation=CreationOperation.INITIALIZATION,
                )
                offspring.append(ind)

        if not offspring:
            logger.warning(
                f"Initialization for '{concept}' produced 0 valid individuals. Retrying..."
            )
            raise RuntimeError("Initialization failed to produce valid individuals.")

        logger.debug(f"Initialized {len(offspring)} individuals for '{concept}'")
        return offspring

    @_retry_policy
    def reproduce(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        current_generation: int,
        target_metric: MetricName,
    ) -> Sequence[Individual]:
        # 1. Prepare View Models (Normalized Scores)
        parent_view_models = self._create_parent_view_models(parents, target_metric)

        # 2. Render Template
        prompt = self._render(
            self.strategy.mutation_template_path,
            context={
                "parents": parent_view_models,
                "concept": concept,
                "num_offsprings": num_offsprings,
            },
        )

        # 3. LLM Generation
        response = self.llm.generate(prompt)

        # 4. Parse & Create Offspring
        data_list: list[list[str]] = self._parse_llm_json(response)
        parent_ids = [p.id for p in parents]

        offspring: list[Individual] = []
        if isinstance(data_list, list):
            for data in data_list:
                if (not isinstance(data, list)) or len(data) != 2:
                    logger.warning(
                        f"Skipping invalid individual data during reproduction for concept '{concept}': {data}"
                    )
                    continue

                ind = Individual(
                    id=uuid.uuid4(),
                    genotype=PromptGenotype(
                        negative_prompt=data[0],
                        positive_prompt=data[1],
                    ),
                    generation_born=current_generation,
                    parents=parent_ids,
                    operation=CreationOperation.LLM_MUTATION,
                    concept=concept,
                )
                offspring.append(ind)

        if len(offspring) < num_offsprings:
            logger.warning(
                f"Requested {num_offsprings} offsprings but only got {len(offspring)} from LLM for concept '{concept}'."
            )

        if not offspring:
            logger.error(
                f"No valid offsprings generated for concept '{concept}'. LLM Response: {response[:200]}..."
            )
            raise RuntimeError("Reproduction failed to produce any valid individuals.")

        if len(offspring) > num_offsprings:
            offspring = offspring[:num_offsprings]  # Trim excess

        return offspring
