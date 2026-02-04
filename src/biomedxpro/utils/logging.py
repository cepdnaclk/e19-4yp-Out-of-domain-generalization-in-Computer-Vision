# src/biomedxpro/utils/logging.py
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import RetryCallState


def setup_logging(
    experiment_name: str, console_level: str = "INFO", log_dir: str = "logs"
) -> tuple[Path, Path]:
    """
    Configures logging with two sinks:
    1. Main Trace: Captures everything (DEBUG/TRACE) for debugging.
    2. Usage Sidecar: Captures only usage metrics for accounting.

    Returns:
        Tuple of (trace_log_path, usage_log_path)
    """

    # 1. Reset
    logger.remove()

    # 2. Generate filenames with timestamps
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trace_filename = f"{experiment_name}_{timestamp}_trace.jsonl"
    usage_filename = f"{experiment_name}_{timestamp}_usage.jsonl"
    trace_log_path = Path(log_dir) / trace_filename
    usage_log_path = Path(log_dir) / usage_filename

    # 3. Console Formatter (Clean)
    def console_formatter(record: Any) -> str:
        fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        if "generation" in record["extra"]:
            fmt += "Gen <bold>{extra[generation]:02d}</bold> | "
        if "island" in record["extra"]:
            fmt += "Island <cyan>{extra[island]: <12}</cyan> | "
        fmt += "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        fmt += "<level>{message}</level>\n{exception}"
        return fmt

    logger.add(sys.stderr, format=console_formatter, level=console_level, colorize=True)

    # 4. Main Trace Sink (Captures everything including usage events for context)
    logger.add(
        trace_log_path,
        serialize=True,
        level="TRACE",
        rotation="100 MB",
        retention="30 days",
        enqueue=True,  # Thread-safe async writes
    )

    # 5. Usage Sidecar Sink (Filters strictly for usage events, writes pure JSON)
    def usage_filter(record: Any) -> bool:
        return "usage_data" in record["extra"]

    def usage_formatter(record: Any) -> str:
        # Extract ONLY the payload, ignore log levels/timestamps/messages
        return json.dumps(record["extra"]["usage_data"]) + "\n"

    logger.add(
        usage_log_path,
        filter=usage_filter,
        format=usage_formatter,
        level="INFO",
        enqueue=True,  # Critical for async writing
    )

    return trace_log_path, usage_log_path


def loguru_before_sleep(retry_state: RetryCallState) -> None:
    """
    Loguru-compatible function to be called before a retry sleep.
    """
    if retry_state.attempt_number > 0:
        # Determine the log level (e.g., INFO for general retries)
        log_level = "INFO"
        # Get information about the current state
        attempt = retry_state.attempt_number
        # next_action is a RetryAction object which has a sleep attribute (float)
        sleep_time = retry_state.next_action.sleep if retry_state.next_action else 0
        fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown"

        logger.log(
            log_level,
            "Retrying function '{}' in {:.3g} seconds: attempt number {}".format(
                fn_name, sleep_time, attempt
            ),
        )
