# src/biomedxpro/utils/usage.py
from dataclasses import dataclass

from loguru import logger


@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    total_calls: int = 0


_STATS = UsageStats()


def record_usage(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int = 0,
) -> None:
    """
    Records LLM usage statistics and emits a usage event to the logging sidecar.

    Args:
        provider: LLM provider name (e.g., "openai", "groq", "gemini")
        model: Model name (e.g., "gpt-4o", "gemini-2.0-flash-exp")
        input_tokens: Number of input/prompt tokens consumed
        output_tokens: Number of output/completion tokens generated
        reasoning_tokens: Number of reasoning tokens (for models like o1), default 0
    """
    # 1. Update in-memory state
    _STATS.input_tokens += input_tokens
    _STATS.output_tokens += output_tokens
    _STATS.reasoning_tokens += reasoning_tokens
    _STATS.total_calls += 1

    # 2. Emit usage event (captured by sidecar sink in logging.py)
    payload = {
        "event": "usage",
        "provider": provider,
        "model": model,
        "input": input_tokens,
        "output": output_tokens,
        "reasoning": reasoning_tokens,
    }
    logger.bind(usage_data=payload).info(f"Tokens: I={input_tokens} O={output_tokens}")


def get_usage_stats() -> UsageStats:
    """Returns the current aggregated usage statistics."""
    return _STATS
