# src/biomedxpro/utils/token_logging.py

import json
import time
from pathlib import Path
from typing import Any, Dict

class TokenUsageLogger:
    """
    Append-only JSONL logger for LLM token usage.
    """

    def __init__(self, experiment_name: str, log_dir: str = "logs") -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = Path(log_dir) / f"{experiment_name}_{timestamp}_tokens.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def log(self, record: Dict[str, Any]) -> None:
        record["timestamp"] = time.time()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
