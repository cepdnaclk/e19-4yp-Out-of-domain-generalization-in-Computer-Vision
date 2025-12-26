# tests/integration/test_providers.py
import os

import pytest
from dotenv import load_dotenv

from biomedxpro.core.domain import TaskDefinition
from biomedxpro.impl.llm_client import create_llm_client
from biomedxpro.impl.operator import LLMOperator
from biomedxpro.utils.config import LLMSettings, PromptStrategy

# 1. Load Real Secrets
load_dotenv()

# --- Shared Fixtures ---


@pytest.fixture
def basic_task_def() -> TaskDefinition:
    return TaskDefinition(
        task_name="Melanoma Classification",
        image_modality="Dermoscopic Images",
        positive_class="Melanoma",
        negative_class="Benign",
        role="Dermatology Expert",
    )


@pytest.fixture
def basic_strategy() -> PromptStrategy:
    # Ensure these paths exist in your project or point to dummy local files for the test
    return PromptStrategy(
        discover_concepts_template_path="src/biomedxpro/prompts/discover_concepts_v1.j2",
        init_template_path="src/biomedxpro/prompts/init_v1.j2",
        mutation_template_path="src/biomedxpro/prompts/mutation_v1.j2",
    )


# --- Integration Tests ---


@pytest.mark.integration
def test_groq_integration(
    basic_strategy: PromptStrategy, basic_task_def: TaskDefinition
) -> None:
    """
    Verifies that we can talk to Groq, get a response, and parse it.
    """
    if not os.getenv("GROQ_API_KEYS"):
        pytest.skip("Skipping Groq test: GROQ_API_KEYS not found in .env")

    settings = LLMSettings(
        provider="groq",
        model_name="openai/gpt-oss-20b",  # Cheap, fast model for testing
        base_url="https://api.groq.com/openai/v1",
        llm_params={"temperature": 0.0},
    )

    # 1. Create Real Client (loads keys from env)
    client = create_llm_client(settings)

    # 2. Create Real Operator
    operator = LLMOperator(client, basic_strategy, basic_task_def)

    # 3. Execute Real Call
    concepts = operator.discover_concepts()

    # 4. Verify
    assert isinstance(concepts, list)
    assert len(concepts) > 0
    assert isinstance(concepts[0], str)
    print(f"\n[Groq] Successfully discovered: {concepts}")


@pytest.mark.integration
def test_gemini_integration(
    basic_strategy: PromptStrategy, basic_task_def: TaskDefinition
) -> None:
    """
    Verifies that we can talk to Google Gemini.
    """
    if not os.getenv("GEMINI_API_KEYS"):
        pytest.skip("Skipping Gemini test: GEMINI_API_KEYS not found in .env")

    settings = LLMSettings(
        provider="gemini",
        model_name="gemma-3-27b-it",
        llm_params={"temperature": 0.0},
    )

    client = create_llm_client(settings)
    operator = LLMOperator(client, basic_strategy, basic_task_def)

    concepts = operator.discover_concepts()

    assert isinstance(concepts, list)
    assert len(concepts) > 0
    print(f"\n[Gemini] Successfully discovered: {concepts}")


@pytest.mark.integration
def test_openai_integration() -> None:
    """
    Placeholder for OpenAI Integration.
    """
    pytest.skip("OpenAI keys pending. Skipping test.")
