# src/biomedxpro/impl/config.py
from dataclasses import dataclass, field, fields
from typing import Any


@dataclass(slots=True, frozen=True)
class PromptStrategy:
    """
    Paths to the external Jinja2 templates.
    This is an infrastructure detail about how prompts are stored.
    """

    mutation_template_path: str = "src/biomedxpro/prompts/mutation_v1.j2"
    init_template_path: str = "src/biomedxpro/prompts/init_v1.j2"
    discover_concepts_template_path: str = (
        "src/biomedxpro/prompts/discover_concepts_v1.j2"
    )


@dataclass(slots=True, frozen=True)
class LLMSettings:
    """
    Infrastructure settings for the API connection.

    Attributes:
        provider: The service provider (e.g., 'openai', 'groq', 'google').
        model_name: The specific model ID (e.g., 'gpt-4o').
        base_url: Optional override for the API endpoint.
        llm_params: A dictionary of model-specific hyperparameters (e.g., temperature,
                    top_p, max_tokens) passed directly to the provider's API.
    """

    provider: str = "groq"
    model_name: str = "openai/gpt-oss-20b"
    base_url: str | None = None

    # Hyperparameters for the text generation
    llm_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LLMSettings":
        """
        Smart Loader:
        Separates explicit infrastructure fields from implicit model parameters.
        """
        # 1. Identify explicit fields (provider, model_name, etc.)
        known_fields = {f.name for f in fields(cls) if f.name != "llm_params"}

        explicit_args = {}
        implicit_params = {}

        # 2. Sort input keys
        for key, value in config.items():
            if key in known_fields:
                explicit_args[key] = value
            else:
                # e.g., "temperature", "top_p" -> llm_params
                implicit_params[key] = value

        return cls(**explicit_args, llm_params=implicit_params)


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    """
    Configuration for data loading and adaptation.
    Defines the bridge between raw storage and the engine.
    """

    adapter: str
    root: str
    name: str
    class_names: list[str]
    shots: int = 0  # 0 = Full Dataset, >0 = Few-Shot (samples per class)
    cache_dir: str = ".biomedxpro_cache"
    adapter_params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DatasetConfig":
        """
        Separates core adapter fields from extra parameters passed to
        the specific adapter implementation.
        """
        known_fields = {f.name for f in fields(cls) if f.name != "adapter_params"}
        explicit_args = {}
        extra_args = {}

        for key, value in config.items():
            if key in known_fields:
                explicit_args[key] = value
            else:
                extra_args[key] = value

        return cls(**explicit_args, adapter_params=extra_args)
