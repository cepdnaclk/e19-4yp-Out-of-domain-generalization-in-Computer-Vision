# src/biomedxpro/impl/llm_operator.py

import json
import math
import random
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
from biomedxpro.impl.config import PromptStrategy
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

        # Cleanup: LLMs often include trailing commas which break standard json.loads
        # Regex to remove trailing commas before closing brackets or braces
        text = re.sub(r",\s*([\]\}])", r"\1", text)

        # Parse JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Provide more diagnostic info on failure
            logger.error(f"JSON Parse Error at line {e.lineno} col {e.colno}: {e.msg}")
            # Identify the problematic line
            lines = text.splitlines()
            if 0 < e.lineno <= len(lines):
                logger.debug(f"Offending line {e.lineno}: {lines[e.lineno - 1]}")

            logger.error(f"Raw output snippet: {raw_response[:200]}...")
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

        # Linear Mapping: configurable range for normalized scores
        lower_bound = 30
        upper_bound = 70

        score_range = upper_bound - lower_bound
        normalized = []
        for s in raw_scores:
            norm = lower_bound + (s - min_s) * (score_range / (max_s - min_s))
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

        logger.debug(f"Discovered concepts response: {response}")

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

        logger.debug(f"Initialization prompt ({concept}):\n{prompt}")

        response = self.llm.generate(prompt)
        logger.debug(f"Initialization response ({concept}):\n{response}")

        data_list: list[list[str]] = self._parse_llm_json(response)

        offspring = []
        if isinstance(data_list, list):
            for data in data_list:
                # Validate: LLM should return list with length = num_classes
                expected_len = len(self.task_def.class_names)
                if (not isinstance(data, list)) or len(data) != expected_len:
                    logger.warning(
                        f"Skipping invalid individual data during initialization for concept '{concept}': "
                        f"Expected {expected_len} prompts, got {data}"
                    )
                    continue

                ind = Individual(
                    id=uuid.uuid4(),
                    # Store as ordered tuple matching task_def.class_names order
                    genotype=PromptGenotype(prompts=tuple(data)),
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
    def mutate(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        current_generation: int,
        target_metric: MetricName,
        cross_concept_exemplars: Sequence[Individual] | None = None,
    ) -> Sequence[Individual]:
        # 1. Prepare View Models (Normalized Scores)
        parent_view_models = self._create_parent_view_models(parents, target_metric)

        # Prepare exemplar view models from other islands
        cross_concept_exemplar_view_models = None
        if cross_concept_exemplars:
            cross_concept_exemplar_view_models = self._create_parent_view_models(
                cross_concept_exemplars, target_metric
            )

        # 2. Render Template
        prompt = self._render(
            self.strategy.mutation_template_path,
            context={
                "parents": parent_view_models,
                "concept": concept,
                "num_offsprings": num_offsprings,
                "cross_concept_exemplars": cross_concept_exemplar_view_models,
            },
        )

        logger.debug(f"Reproduction prompt ({concept}):\n{prompt}")
        # 3. LLM Generation
        response = self.llm.generate(prompt)

        logger.debug(f"Reproduction response ({concept}):\n{response}")
        # 4. Parse & Create Offspring
        data_list: list[list[str]] = self._parse_llm_json(response)
        parent_ids = [p.id for p in parents]

        offspring: list[Individual] = []
        if isinstance(data_list, list):
            for data in data_list:
                if not isinstance(data, list):
                    logger.warning(
                        f"Skipping invalid individual data during reproduction for concept '{concept}': {data}"
                    )
                    continue

                expected_len = len(self.task_def.class_names)
                if len(data) != expected_len:
                    logger.warning(
                        f"Individual data does not contain exactly {expected_len} prompts. "
                        f"Expected {expected_len}, got {len(data)}. Skipping."
                    )
                    continue

                ind = Individual(
                    id=uuid.uuid4(),
                    genotype=PromptGenotype(prompts=tuple(data)),
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
            raise RuntimeError("Mutation failed to produce any valid individuals.")

        if len(offspring) > num_offsprings:
            offspring = offspring[:num_offsprings]  # Trim excess

        return offspring

    def crossover(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        current_generation: int,
        target_metric: MetricName,
    ) -> Sequence[Individual]:
        """
        Generates offspring via crossover (genetic recombination).

        Pure combinatorial logic - no LLM needed.
        Creates new prompt combinations by randomly mixing class-specific
        prompts from the parent population.

        Args:
            parents: Source individuals to recombine
            concept: The concept this crossover is focused on
            num_offsprings: Target number of offspring to generate
            current_generation: Current generation number
            target_metric: Unused (kept for interface compatibility)

        Returns:
            List of new individuals created via crossover
        """
        # Need at least 2 individuals to mix useful traits
        if len(parents) < 2:
            logger.warning(f"Crossover needs at least 2 parents, got {len(parents)}")
            return []

        num_classes = len(self.task_def.class_names)

        # 1. Extract Gene Pools
        # pool[0] = {all descriptions for class 0}
        gene_pools: List[List[str]] = [[] for _ in range(num_classes)]
        for ind in parents:
            if len(ind.genotype.prompts) == num_classes:
                for i, prompt in enumerate(ind.genotype.prompts):
                    gene_pools[i].append(prompt)

        # Validate we have genetic material
        if any(len(pool) == 0 for pool in gene_pools):
            logger.warning(f"Crossover for '{concept}' has empty gene pools")
            return []

        # 2. Track existing genotypes to ensure uniqueness
        # Start with the input individuals so we don't duplicate them
        existing_genotypes = {ind.genotype.prompts for ind in parents}
        new_offspring: List[Individual] = []

        # 3. Generate Combinations
        # Safety break: 5x attempts to find num_offsprings unique children
        attempts = 0
        max_attempts = num_offsprings * 5

        while len(new_offspring) < num_offsprings and attempts < max_attempts:
            attempts += 1

            # Construct a random genotype from the pools
            new_prompts_list = []
            for i in range(num_classes):
                new_prompts_list.append(random.choice(gene_pools[i]))

            new_genotype = tuple(new_prompts_list)

            # Skip if we already have this specific combination
            if new_genotype in existing_genotypes:
                continue

            # Found a unique child!
            existing_genotypes.add(new_genotype)

            # Map all source individuals as parents (Pool Parentage)
            parent_ids = [ind.id for ind in parents]

            ind = Individual(
                id=uuid.uuid4(),
                genotype=PromptGenotype(prompts=new_genotype),
                generation_born=current_generation,
                parents=parent_ids,
                operation=CreationOperation.CROSSOVER,
                concept=concept,
            )
            new_offspring.append(ind)

        logger.debug(
            f"Crossover generated {len(new_offspring)}/{num_offsprings} unique recombinations "
            f"for concept '{concept}' (attempts: {attempts})"
        )

        return new_offspring
