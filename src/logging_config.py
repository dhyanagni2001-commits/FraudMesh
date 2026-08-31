"""Structured logging setup for the serving API.

Training/analysis scripts (train_*.py, case_study.py) intentionally keep
plain print() — they're interactive CLI tools where progress-as-you-go
stdout is the right UX. serve.py is the long-running service, where real
log levels/timestamps matter (for grepping, log aggregation, etc.).
"""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level_name = os.environ.get("FRAUDMESH_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
