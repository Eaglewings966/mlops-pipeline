"""Logging utility with Rich formatting."""
import logging
import os
from rich.logging import RichHandler
from rich.console import Console

console = Console()


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with Rich handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        if os.environ.get("TESTING"):
            handler = logging.StreamHandler()
        else:
            handler = RichHandler(
                console=console,
                show_time=True,
                show_path=False,
                markup=True
            )
        handler.setFormatter(
            logging.Formatter("%(message)s", datefmt="[%X]")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
