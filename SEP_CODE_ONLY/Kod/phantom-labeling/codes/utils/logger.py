"""Logging utilities for the pipeline."""

import logging
import os
import sys


def setup_logger(
    name: str = "phantom_pipeline",
    level: int = logging.INFO,
    log_file: str = None,
) -> logging.Logger:
    """Create and configure a logger.

    Args:
        name: Logger name.
        level: Logging level.
        log_file: Optional path to a log file. If provided, a file handler is added.
    """
    logger = logging.getLogger(name)

    # When reconfiguring, update level and optionally add file handler
    logger.setLevel(level)

    if not logger.handlers:
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    else:
        for h in logger.handlers:
            h.setLevel(level)

    if log_file:
        log_abs = os.path.abspath(log_file)
        os.makedirs(os.path.dirname(log_abs), exist_ok=True)
        # Close and remove any stale FileHandlers pointing at a different
        # path — without this, batch_run_siemens.py leaks a handler per
        # series (crosstalk + FD exhaustion).
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and h.baseFilename != log_abs:
                h.close()
                logger.removeHandler(h)
        if not any(
            isinstance(h, logging.FileHandler) and h.baseFilename == log_abs
            for h in logger.handlers
        ):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            ))
            logger.addHandler(fh)

    return logger


log = setup_logger()
