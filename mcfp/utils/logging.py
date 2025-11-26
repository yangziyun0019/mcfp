from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


def setup_logger(name: str, log_dir: Optional[str | Path] = None) -> logging.Logger:
    """Create and configure a logger with a unified format.

    Parameters
    ----------
    name:
        Logger name, e.g. "mcfp.sim.data_gen_single".
    log_dir:
        Optional directory where a log file will be created. If None,
        only console logging is used.

    Returns
    -------
    logger:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # Logger already configured

    logger.setLevel(logging.INFO)

    log_format = "[%(asctime)s][%(levelname)s][%(name)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=log_format, datefmt=datefmt)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Optional file handler
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{name.replace('.', '_')}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
