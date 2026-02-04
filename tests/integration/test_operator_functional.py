import os

import pytest
from dotenv import load_dotenv

from biomedxpro.core.domain import (
    CreationOperation,
    Individual,
    PromptGenotype,
    TaskDefinition,
)
from biomedxpro.impl.config import LLMSettings, PromptStrategy
from biomedxpro.impl.llm_client import create_llm_client
from biomedxpro.impl.llm_operator import LLMOperator

# Load secrets
load_dotenv()

# --- Fixtures ---


@pytest.fixture
def functional_task_def() -> TaskDefinition:
    return TaskDefinition(
        task_name="Pneumonia Detection",
        image_modality="Chest X-Ray",
        class_names=["Normal", "Pneumonia"],
        role="Radiologist",
        concepts=None,
    )


@pytest.fixture
def real_operator(functional_task_def: TaskDefinition) -> LLMOperator:
    """
    Creates an operator connected to a REAL, fast, cheap LLM.
    We prefer Groq (Llama 3) or Gemini Flash for speed/cost.
    """
    # Prefer Groq if available, fallback to Gemini
    if os.getenv("GROQ_API_KEYS"):
        settings = LLMSettings(
            provider="groq",
            model_name="openai/gpt-oss-20b",  # Cheap, fast model for testing
            base_url="https://api.groq.com/openai/v1",
            llm_params={"temperature": 0.0},
        )

    elif os.getenv("GEMINI_API_KEYS"):
        settings = LLMSettings(
            provider="gemini",
            model_name="gemma-3-27b-it",
            llm_params={"temperature": 0.5},
        )
    else:
        pytest.skip("No API keys available for functional integration test.")

    client = create_llm_client(settings)

    # Ensure strategy paths are correct relative to where pytest is run
    strategy = PromptStrategy(
        discover_concepts_template_path="src/biomedxpro/prompts/discover_concepts_v1.j2",
        init_template_path="src/biomedxpro/prompts/init_v1.j2",
        mutation_template_path="src/biomedxpro/prompts/mutation_v1.j2",
    )

    return LLMOperator(client, strategy, functional_task_def)


# --- The Functional Test Suite ---


@pytest.mark.integration
class TestOperatorFunctional:
    def test_full_evolutionary_cycle(self, real_operator: LLMOperator) -> None:
        """
        Executes the 'Circle of Life':
        Discover -> Initialize -> (Simulate Eval) -> Reproduce.
        """
        print(
            f"\n--- Starting Functional Test with Provider: {real_operator.llm.__class__.__name__} ---"
        )

        # 1. DISCOVERY
        # Does the LLM understand the domain and return a valid JSON list?
        concepts = real_operator.discover_concepts()
        print(f"Discovered Concepts: {concepts}")
        assert len(concepts) > 0
        target_concept = concepts[0]

        # 2. INITIALIZATION (Genesis)
        # Does the LLM follow the "List of Lists" format?
        # Does the Operator correctly map [neg, pos] -> PromptGenotype?
        pop_size = 3
        population = real_operator.initialize_population(
            num_offsprings=pop_size, concept=target_concept
        )

        print(f"Initialized {len(population)} individuals.")
        assert len(population) > 0  # It might filter invalid ones, but should get some

        first_born = population[0]
        assert isinstance(first_born, Individual)
        assert isinstance(first_born.genotype, PromptGenotype)  # Dataclass check
        assert len(first_born.genotype.prompts) == 2  # Should have 2 class prompts
        assert all(prompt for prompt in first_born.genotype.prompts)  # All non-empty

        # 3. SIMULATE EVALUATION (The "Mock" Step)
        # We can't integrate the GPU here easily, so we manually assign fitness
        # to simulate that Phase 3 (Batch Eval) happened.
        for i, ind in enumerate(population):
            # Assign fake scores: 0.8, 0.85, 0.9
            score = 0.8 + (i * 0.05)
            ind.metrics = {
                "inverted_bce": score,
                "accuracy": score,
                "f1_macro": score,
                "auc": score,
                "f1_weighted": score,
                "margin_score": score,
            }

        # 4. REPRODUCTION (Mutation)
        # Does the Operator correctly normalize those scores (0.8 -> 60)?
        # Does the LLM understand the mutation task?
        offspring_count = 2
        offspring = real_operator.reproduce(
            parents=population,
            concept=target_concept,
            num_offsprings=offspring_count,
            current_generation=1,
            target_metric="inverted_bce",
        )

        print(f"Produced {len(offspring)} offspring.")
        assert len(offspring) > 0

        child = offspring[0]
        assert child.generation_born == 1
        assert child.parents  # Should track lineage
        assert child.operation == CreationOperation.LLM_MUTATION

        print("--- Functional Test Complete: Cycle Verified ---")
        print("--- Functional Test Complete: Cycle Verified ---")
