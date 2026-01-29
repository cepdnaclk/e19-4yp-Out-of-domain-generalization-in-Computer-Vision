# src/biomedxpro/utils/token_logging.py

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
class TokenUsageLogger:
    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_{timestamp}_tokens.jsonl"
        self.file_path = Path(log_dir) / filename
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.touch()

        # Running totals
        self.total_prompt = 0
        self.total_completion = 0
        self.total_calls = 0

        self.estimated_prompt = 0
        self.estimated_completion = 0
        self.estimated_calls = 0

    def log(self, record: dict):
        # Append the record
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        # Update totals only if numeric
        if not record.get("prompt_estimated", False):
            self.total_prompt += record.get("prompt_tokens", 0) or 0
        else:
            self.estimated_prompt += record.get("prompt_tokens", 0) or 0

        if not record.get("completion_estimated", False):
            self.total_completion += record.get("completion_tokens", 0) or 0
        else:
            self.estimated_completion += record.get("completion_tokens", 0) or 0

        self.total_calls += 1
        if record.get("prompt_estimated", False) or record.get("completion_estimated", False):
            self.estimated_calls += 1

    def summary(self):
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt,
            "total_completion_tokens": self.total_completion,
            "estimated_prompt_tokens": self.estimated_prompt,
            "estimated_completion_tokens": self.estimated_completion,
            "total_tokens": self.total_prompt + self.total_completion,
            "total_estimated_calls": self.estimated_calls,
        }

    def dump_summary(self):
        # Add a final record at the end of the log
        summary_record = {"event": "token_summary", **self.summary()}
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_record) + "\n")
