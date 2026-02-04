# src/biomedxpro/utils/token_logging.py

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger


class TokenUsageLogger:
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}_tokens.jsonl"
        self.file_path = Path(log_dir) / filename
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.touch()

        # Running totals
        self.total_input = 0
        self.total_output = 0
        self.total_reasoning = 0
        self.total_calls = 0

        logger.info(f"Initialized token logger: {self.file_path}")

    def log(self, record: dict[str, Any]) -> None:
        # Append the record
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Update totals
        input_tokens = record.get("input_tokens", 0)
        output_tokens = record.get("output_tokens", 0)
        reasoning_tokens = record.get("reasoning_tokens", 0)

        self.total_input += input_tokens
        self.total_output += output_tokens
        self.total_reasoning += reasoning_tokens
        self.total_calls += 1

        logger.debug(
            f"LLM call #{self.total_calls}: "
            f"{record.get('provider')}/{record.get('model')} - "
            f"Input: {input_tokens}, Output: {output_tokens}, Reasoning: {reasoning_tokens}"
        )

    def summary(self) -> dict[str, int]:
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input,
            "total_output_tokens": self.total_output,
            "total_reasoning_tokens": self.total_reasoning,
            "total_tokens": self.total_input + self.total_output + self.total_reasoning,
        }

    def dump_summary(self) -> None:
        # Add a final record at the end of the log
        summary_record = {"event": "token_summary", **self.summary()}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record) + "\n")

        logger.info(
            f"Token usage summary: {self.total_calls} calls, "
            f"{self.total_input + self.total_output + self.total_reasoning} total tokens "
            f"(Input: {self.total_input}, Output: {self.total_output}, Reasoning: {self.total_reasoning})"
        )
