"""Logging, hashing, jitter helpers."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def configure_logging(level: str, log_file: Path) -> None:
    root = logging.getLogger()
    if root.handlers:
        return
    root.setLevel(level.upper())

    fmt = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    fmt.converter = time.gmtime  # UTC partout

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=2_000_000, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def hash_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
