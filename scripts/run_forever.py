"""Continuous collector — Ctrl+C for a clean stop."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scraper.config import load_settings
from scraper.db import init_engine
from scraper.scheduler import run_forever
from scraper.utils import configure_logging


def main() -> int:
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_file)
    init_engine(settings)
    run_forever(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
