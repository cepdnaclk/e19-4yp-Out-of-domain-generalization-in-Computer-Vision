import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger
from tenacity import RetryCallState


def setup_logging(
    experiment_name: str, console_level: str = "INFO", log_dir: str = "logs"
) -> Path:
    """
    Configures logging based on the experiment name.

    Output File: logs/{experiment_name}_{timestamp}.jsonl
    """

    # 1. Reset
    logger.remove()

    # 2. Generate a filename that includes the name + exact time
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.jsonl"
    log_path = Path(log_dir) / filename

    # 3. Console Formatter (Clean)
    def console_formatter(record: Any) -> str:
        fmt = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        if "generation" in record["extra"]:
            fmt += "Gen <bold>{extra[generation]:02d}</bold> | "
        if "island" in record["extra"]:
            fmt += "Island <cyan>{extra[island]: <12}</cyan> | "
        fmt += "<level>{message}</level>\n{exception}"
        return fmt

    logger.add(sys.stderr, format=console_formatter, level=console_level, colorize=True)

    # 4. File Formatter (Machine Readable)
    # We serialize to JSONL so you can parse it later.
    logger.add(
        log_path,
        serialize=True,
        level="TRACE",
        rotation="100 MB",
        retention="30 days",
    )

    # Return the path so we can print it to the user
    return log_path  # Return the path so we can print it to the user


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
