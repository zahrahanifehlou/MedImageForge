"""One logging setup for the whole platform.

Why this module exists:
    Every component (ingest, curation, QC, training, API) logs through the
    same format and the same level, configured once here. That consistency is
    what later lets us build audit trails and debug pipeline runs.
    If components invented their own logging, the trail would be unusable.

Usage in any module:
    from medimageforge.logging_utils import get_logger
    log = get_logger(__name__)
    log.info("...")
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once, at process start (the CLI calls this)."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Pass __name__ so logs show which module spoke."""
    return logging.getLogger(name)
