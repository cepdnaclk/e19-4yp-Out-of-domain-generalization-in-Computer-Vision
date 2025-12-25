import random
from typing import Sequence

import torch

from biomedxpro.core.domain import (
    CreationOperator,
    EncodedDataset,
    Individual,
    MetricName,
    Population,
    PromptGenotype,
    TaskDefinition,
)
from biomedxpro.core.interfaces import IFitnessEvaluator, IOperator, SelectionStrategy


# --- Mock Operator ---
class MockOperator(IOperator):
    """
    A purely synthetic operator for smoke-testing the engine.
    It generates random strings directly, bypassing Jinja2 templates and LLM Clients.
    """

    def __init__(self, task_def: TaskDefinition) -> None:
        self.task_def = task_def

    def discover_concepts(self) -> list[str]:
        # Return a fixed list so the loop has something to work on
        return ["MockTexture", "MockShape", "MockColor"]

    def initialize_population(
        self, num_offsprings: int, concept: str
    ) -> Sequence[Individual]:
        """
        Generates random 'Adam & Eve' individuals.
        """
        offspring = []
        for _ in range(num_offsprings):
            suffix = random.randint(1000, 9999)
            ind = Individual(
                genotype=PromptGenotype(
                    negative_prompt=f"init neg {concept} {suffix}",
                    positive_prompt=f"init pos {concept} {suffix}",
                ),
                generation_born=0,
                operator=CreationOperator.INITIALIZATION,
                metadata={"source": "mock_init"},
                concept=concept,
            )
            offspring.append(ind)
        return offspring

    def reproduce(
        self,
        parents: Sequence[Individual],
        concept: str,
        num_offsprings: int,
        current_generation: int,
        target_metric: MetricName,
    ) -> Sequence[Individual]:
        """
        Generates random mutations.
        """
        if not parents:
            return []

        # Inherit generation info
        gen_born = current_generation + 1
        parent_ids = [p.id for p in parents]

        offspring = []
        for _ in range(num_offsprings):
            suffix = random.randint(1000, 9999)
            # Simulate inheritance by picking a random parent to mention in debug text
            parent = random.choice(parents)

            ind = Individual(
                genotype=PromptGenotype(
                    negative_prompt=f"mutated neg {concept} {suffix} (from {parent.genotype['negative_prompt'][:10]}...)",
                    positive_prompt=f"mutated pos {concept} {suffix}",
                ),
                generation_born=gen_born,
                parents=parent_ids,
                operator=CreationOperator.LLM_MUTATION,
                metadata={"source": "mock_reproduce"},
                concept=concept,
            )
            offspring.append(ind)
        return offspring


# --- Mock Evaluator ---
class MockEvaluator(IFitnessEvaluator):
    """
    Pretends to be BiomedCLIP.
    """

    def evaluate(
        self, candidates: Sequence[Individual], dataset: EncodedDataset
    ) -> None:
        for cand in candidates:
            if cand.is_evaluated:
                continue

            # Generate a random score skewed slightly high to simulate "learning"
            score = random.uniform(0.5, 0.95)

            cand.update_metrics(
                {
                    "inverted_bce": score,
                    "f1_macro": score,
                    "accuracy": score,
                    "auc": score,
                    "f1_weighted": score,
                }
            )


# --- Mock Selector ---
class RandomSelector(SelectionStrategy):
    """
    Just picks k random parents.
    """

    def select(
        self, population: Population, k: int, metric: MetricName
    ) -> Sequence[Individual]:
        candidates = [ind for ind in population.individuals if ind.is_evaluated]
        if not candidates:
            return []

        # Safe sample
        k = min(k, len(candidates))
        return random.sample(candidates, k)


# --- Helper for Mock Data ---
def create_dummy_dataset() -> EncodedDataset:
    """
    Creates fake tensors so the Orchestrator doesn't crash on init.
    """
    return EncodedDataset(
        name="MockDataset",
        features=torch.randn(10, 512),
        labels=torch.randint(0, 2, (10,)),
        class_names=["Benign", "Malignant"],
    )
