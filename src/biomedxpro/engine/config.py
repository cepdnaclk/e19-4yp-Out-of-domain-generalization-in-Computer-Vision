from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from biomedxpro.core.domain import EvolutionParams, TaskDefinition
from biomedxpro.impl.config import DatasetConfig, LLMSettings, PromptStrategy


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """Controls the 'Physics of the Computer' (Hardware/Runtime)."""

    # Threads for parallel LLM API calls (Network I/O)
    orchestrator_io_workers: int = 10

    # Processes for parallel Image Loading (CPU/Disk I/O)
    dataloader_cpu_workers: int = 4

    device: str = "cuda"
    batch_size: int = 32

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "ExecutionConfig":
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in config.items() if k in valid_keys}
        return cls(**filtered)


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> None:
    """Recursively updates a dictionary."""
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


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

    @classmethod
    def from_composable(
        cls,
        task_path: Path,
        evo_path: Path,
        llm_path: Path,
        exec_path: Path,
        target_metric: str,
        shots: int,
    ) -> "MasterConfig":
        """
        Assembles the MasterConfig from four distinct configuration sources.
        """
        combined_config: dict[str, Any] = {}

        # 1. Load Task & Dataset (The Problem)
        with open(task_path, "r") as f:
            _deep_update(combined_config, yaml.safe_load(f) or {})

        # 2. Load Evolution (The Method)
        with open(evo_path, "r") as f:
            _deep_update(combined_config, yaml.safe_load(f) or {})

        # 3. Load LLM Settings (The Intelligence)
        with open(llm_path, "r") as f:
            _deep_update(combined_config, yaml.safe_load(f) or {})

        # 4. Load Execution Settings (The Hardware)
        with open(exec_path, "r") as f:
            _deep_update(combined_config, yaml.safe_load(f) or {})

        # 5. Inject CLI Overrides
        # Ensure mandatory sections exist before injecting
        combined_config.setdefault("evolution", {})
        combined_config.setdefault("dataset", {})

        combined_config["evolution"]["target_metric"] = target_metric
        combined_config["dataset"]["shots"] = shots

        # 6. Validate & Instantiate
        try:
            strategy_dict = combined_config.get("strategy", {})
            return cls(
                task=TaskDefinition.from_dict(combined_config["task"]),
                dataset=DatasetConfig.from_dict(combined_config["dataset"]),
                evolution=EvolutionParams.from_dict(combined_config["evolution"]),
                llm=LLMSettings.from_dict(combined_config["llm"]),
                execution=ExecutionConfig.from_dict(combined_config["execution"]),
                strategy=PromptStrategy(**strategy_dict)
                if strategy_dict
                else PromptStrategy(),
            )
        except KeyError as e:
            raise ValueError(f"Missing configuration section: {e}")
