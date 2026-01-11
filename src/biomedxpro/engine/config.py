from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from biomedxpro.core.domain import EvolutionParams, TaskDefinition
from biomedxpro.impl.config import DatasetConfig, LLMSettings, PromptStrategy


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """Controls the 'Physics of the Computer' (Hardware/Runtime)."""

    max_workers: int = 1
    device: str = "cuda"
    batch_size: int = 32

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ExecutionConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)


@dataclass(slots=True, frozen=True)
class MasterConfig:
    """The Root of the configuration tree."""

    task: TaskDefinition
    dataset: DatasetConfig
    evolution: EvolutionParams
    llm: LLMSettings
    execution: ExecutionConfig
    strategy: PromptStrategy = field(default_factory=PromptStrategy)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MasterConfig":
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)

        return cls(
            task=TaskDefinition.from_dict(raw_config.get("task", {})),
            dataset=DatasetConfig.from_dict(raw_config.get("dataset", {})),
            evolution=EvolutionParams.from_dict(raw_config.get("evolution", {})),
            llm=LLMSettings.from_dict(raw_config.get("llm", {})),
            execution=ExecutionConfig.from_dict(raw_config.get("execution", {})),
            strategy=PromptStrategy(**raw_config.get("strategy", {}))
            if "strategy" in raw_config
            else PromptStrategy(),
        )
