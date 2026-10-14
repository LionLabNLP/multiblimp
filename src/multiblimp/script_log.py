import logging
import os
import sys
from datetime import datetime

from .config import OUTPUT_LOGS_DIR


def script_logger(script_name: str) -> logging.Logger:
    """Logger that prints to stdout and rewrites output/logs/<script_name>.log each run."""
    os.makedirs(OUTPUT_LOGS_DIR, exist_ok=True)
    log_path = os.path.join(OUTPUT_LOGS_DIR, f"{script_name}.log")

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w", encoding="utf-8")):
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.info(f"# {script_name} {datetime.now():%Y-%m-%d %H:%M:%S}")
    return logger


def append_log(log_name: str, message: str, ext: str = ".log") -> None:
    """Append one timestamped line to output/logs/<log_name><ext> (for events raised inside library loops)."""
    os.makedirs(OUTPUT_LOGS_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_LOGS_DIR, f"{log_name}{ext}"), "a", encoding="utf-8") as f:
        f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{message}\n")


def log_cache_error(mode: str, lang: str, reason: str, detail: str = "") -> None:
    """One line per failed or killed language in output/logs/cache_error_log.txt,
    followed by the indented traceback if there is one."""
    message = f"{mode}\t{lang}\t{reason}"
    if detail:
        message += "\n" + "".join(f"    {line}\n" for line in detail.rstrip().splitlines()).rstrip("\n")
    append_log("cache_error_log", message, ext=".txt")
