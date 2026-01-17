import json
from unittest.mock import MagicMock, patch

import pytest

from biomedxpro.core.domain import (
    CreationOperation,
    Individual,
    PromptGenotype,
    TaskDefinition,
)
from biomedxpro.core.interfaces import ILLMClient
from biomedxpro.impl.config import PromptStrategy
from biomedxpro.impl.llm_operator import LLMOperator

# --- Fixtures ---


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_strategy() -> PromptStrategy:
    return PromptStrategy(
        discover_concepts_template_path="templates/discover.j2",
        init_template_path="templates/init.j2",
        mutation_template_path="templates/mutation.j2",
    )


@pytest.fixture
def mock_task_def() -> TaskDefinition:
    return TaskDefinition(
        task_name="Test Task",
        image_modality="Test Modality",
        positive_class="Pos",
        negative_class="Neg",
        role="Tester",
        concepts=None,
    )


@pytest.fixture
def operator_with_mocks(
    mock_llm: MagicMock, mock_strategy: PromptStrategy, mock_task_def: TaskDefinition
) -> LLMOperator:
    # We must patch Path.exists and open() during __init__ to pass validation
    with patch("pathlib.Path.exists", return_value=True):
        op = LLMOperator(mock_llm, mock_strategy, mock_task_def)

        # Override retry policy to be fast (no sleep) for tests
        # We replace the wait strategy with a fixed 0 seconds
        op.discover_concepts.retry.wait = lambda *args, **kwargs: 0  # type: ignore[attr-defined]
        op.initialize_population.retry.wait = lambda *args, **kwargs: 0  # type: ignore[attr-defined]
        op.reproduce.retry.wait = lambda *args, **kwargs: 0  # type: ignore[attr-defined]

        return op


# --- Test Cases ---


class TestInitialization:
    def test_raises_if_template_missing(
        self,
        mock_llm: ILLMClient,
        mock_strategy: PromptStrategy,
        mock_task_def: TaskDefinition,
    ) -> None:
        # Simulate check failing
        with patch("pathlib.Path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                LLMOperator(mock_llm, mock_strategy, mock_task_def)


class TestParsingLogic:
    """Tests the _parse_llm_json helper directly."""

    def test_parses_raw_json(self, operator_with_mocks: LLMOperator) -> None:
        raw = '["concept1", "concept2"]'
        result = operator_with_mocks._parse_llm_json(raw)
        assert result == ["concept1", "concept2"]

    def test_parses_markdown_json(self, operator_with_mocks: LLMOperator) -> None:
        raw = """
        Here is the output:
        ```json
        [
            ["neg_foo", "pos_foo"]
        ]
        ```
        """
        result = operator_with_mocks._parse_llm_json(raw)
        assert result == [["neg_foo", "pos_foo"]]

    def test_raises_on_bad_json(self, operator_with_mocks: LLMOperator) -> None:
        raw = '["incomplete...'
        with pytest.raises(json.JSONDecodeError):
            operator_with_mocks._parse_llm_json(raw)


class TestDiscoverConcepts:
    def test_success(
        self, operator_with_mocks: LLMOperator, mock_llm: MagicMock
    ) -> None:
        # Setup LLM to return valid list
        mock_llm.generate.return_value = '["Shape", "Texture"]'

        # Patch render to avoid needing real template files
        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            concepts = operator_with_mocks.discover_concepts()

        assert concepts == ["Shape", "Texture"]
        mock_llm.generate.assert_called_once()

    def test_retries_on_wrong_type(
        self, operator_with_mocks: LLMOperator, mock_llm: MagicMock
    ) -> None:
        """Test scenario where LLM returns a Dict instead of List."""
        # 1st call: Dict (Bad), 2nd call: List (Good)
        mock_llm.generate.side_effect = ['{"concepts": ["Bad"]}', '["Good"]']

        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            concepts = operator_with_mocks.discover_concepts()

        assert concepts == ["Good"]
        assert mock_llm.generate.call_count == 2  # Proves retry happened

    def test_fails_after_retries(
        self, operator_with_mocks: LLMOperator, mock_llm: MagicMock
    ) -> None:
        # Always return garbage
        mock_llm.generate.return_value = "NOT JSON"

        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            with pytest.raises(
                json.JSONDecodeError
            ):  # After retries, the JSON error is raised
                operator_with_mocks.discover_concepts()


class TestInitializePopulation:
    def test_success(
        self, operator_with_mocks: LLMOperator, mock_llm: MagicMock
    ) -> None:
        response_json = json.dumps([["n1", "p1"], ["n2", "p2"]])
        mock_llm.generate.return_value = response_json

        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            offspring = operator_with_mocks.initialize_population(2, "Shape")

        assert len(offspring) == 2
        assert offspring[0].genotype.positive_prompt == "p1"
        assert offspring[0].concept == "Shape"

    def test_filters_invalid_items(
        self, operator_with_mocks: LLMOperator, mock_llm: MagicMock
    ) -> None:
        # One valid, one invalid (wrong length)
        response = json.dumps(
            [
                ["n1", "p1"],  # Valid
                ["n2"],  # Invalid
            ]
        )
        mock_llm.generate.return_value = response

        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            offspring = operator_with_mocks.initialize_population(2, "Shape")

        assert len(offspring) == 1  # Only 1 survived

    def test_retries_if_all_filtered_out(
        self, operator_with_mocks: LLMOperator, mock_llm: MagicMock
    ) -> None:
        # 1st attempt: All invalid. 2nd attempt: Valid.
        bad_response = json.dumps([["wrong"]])
        good_response = json.dumps([["n", "p"]])

        mock_llm.generate.side_effect = [bad_response, good_response]

        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            offspring = operator_with_mocks.initialize_population(1, "Shape")

        assert len(offspring) == 1
        assert mock_llm.generate.call_count == 2  # Retry triggered


class TestReproduce:
    @pytest.fixture
    def parents(self) -> list[Individual]:
        # Create parents with specific scores to test normalization
        # Scores: 0.8, 0.9 (Range = 0.1)
        # Expected Norm:
        #   0.8 -> Min -> 60
        #   0.9 -> Max -> 90
        p1 = Individual(
            id="1",
            genotype=PromptGenotype(negative_prompt="n", positive_prompt="p"),
            generation_born=0,
            parents=[],
            concept="C",
            operation=CreationOperation.INITIALIZATION,
        )
        p1.metrics = {
            "inverted_bce": 0.8,
            "accuracy": 0.85,
            "auc": 0.88,
            "f1_macro": 0.82,
            "f1_weighted": 0.84,
        }

        p2 = Individual(
            id="2",
            genotype=PromptGenotype(negative_prompt="n", positive_prompt="p"),
            generation_born=0,
            parents=[],
            concept="C",
            operation=CreationOperation.INITIALIZATION,
        )
        p2.metrics = {
            "inverted_bce": 0.9,
            "accuracy": 0.92,
            "auc": 0.93,
            "f1_macro": 0.91,
            "f1_weighted": 0.89,
        }
        return [p1, p2]

    def test_score_normalization_passed_to_template(
        self,
        operator_with_mocks: LLMOperator,
        mock_llm: MagicMock,
        parents: list[Individual],
    ) -> None:
        mock_llm.generate.return_value = "[]"  # Return empty so we don't crash on parse

        # We Mock _render to spy on the context passed to it
        with patch.object(operator_with_mocks, "_render") as mock_render:
            # Force empty return exception to stop execution after render,
            # or just mock parse to return valid empty list and catch the Runtime error
            mock_llm.generate.return_value = json.dumps([["n", "p"]])

            operator_with_mocks.reproduce(parents, "Shape", 1, 1, "inverted_bce")

            # Extract arguments passed to _render
            call_args = mock_render.call_args[1]  # kwargs
            context = call_args["context"]
            view_models = context["parents"]

            # Verify Sorting (Lowest score first) and Normalization
            assert view_models[0]["score"] == 60  # 0.8
            assert view_models[1]["score"] == 90  # 0.9

    def test_truncates_excess_offspring(
        self,
        operator_with_mocks: LLMOperator,
        mock_llm: MagicMock,
        parents: list[Individual],
    ) -> None:
        # Requested 1, LLM gives 2
        response = json.dumps([["n1", "p1"], ["n2", "p2"]])
        mock_llm.generate.return_value = response

        with patch.object(operator_with_mocks, "_render", return_value="prompt"):
            offspring = operator_with_mocks.reproduce(
                parents,
                "Shape",
                num_offsprings=1,
                current_generation=1,
                target_metric="inverted_bce",
            )

        assert len(offspring) == 1
        assert offspring[0].genotype.positive_prompt == "p1"
        assert offspring[0].genotype.positive_prompt == "p1"
