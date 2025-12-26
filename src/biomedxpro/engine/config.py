from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    """Controls the 'Physics of the Computer' (Hardware/Runtime)."""

    max_workers: int = 1
    device: str = "cuda"
    batch_size: int = 32
