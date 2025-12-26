from dataclasses import dataclass, fields
from typing import Any


@dataclass(slots=True, frozen=True)
class PromptStrategy:
    # Paths to the external Jinja2 templates
    mutation_template_path: str = "src/biomedxpro/prompts/mutation_v1.j2"
    init_template_path: str = "src/biomedxpro/prompts/init_v1.j2"
    discover_concepts_template_path: str = (
        "src/biomedxpro/prompts/discover_concepts_v1.j2"
    )


@dataclass(slots=True, frozen=True)
class LLMSettings:
    """Infrastructure settings for the API connection."""

    provider: str = "gemini"
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.7

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "LLMSettings":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)
