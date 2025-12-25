import sys
from datetime import datetime
from pathlib import Path

from loguru import Record, logger


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
    def console_formatter(record: Record) -> str:
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
