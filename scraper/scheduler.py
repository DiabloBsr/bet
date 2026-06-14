"""Continuous scheduling loop with graceful shutdown (SIGINT/SIGTERM)."""
from __future__ import annotations

import logging
import random
import signal
import threading

from scraper.collector import run_iteration
from scraper.config import Settings

log = logging.getLogger("scraper.scheduler")

_shutdown = threading.Event()


def _request_shutdown(signum, frame):
    log.info("signal=%s received — stopping after current iteration", signum)
    _shutdown.set()


def _install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _request_shutdown)
    # SIGTERM existe sur Windows mais ne se déclenche pas comme sur Unix ;
    # on l'installe quand même si supporté.
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _request_shutdown)
        except (AttributeError, ValueError, OSError):
            pass


def run_forever(settings: Settings) -> None:
    _install_signal_handlers()
    log.info(
        "continuous run start url=%s interval=%ds jitter=±%ds",
        settings.target_url,
        settings.scrape_interval_seconds,
        settings.jitter_seconds,
    )

    while not _shutdown.is_set():
        try:
            run_iteration(settings)
        except Exception:  # noqa: BLE001 — l'erreur est déjà loggée plus bas, on continue
            log.exception("uncaught error in iteration — continuing loop")

        if _shutdown.is_set():
            break

        delay = settings.scrape_interval_seconds + random.uniform(
            -settings.jitter_seconds, settings.jitter_seconds
        )
        delay = max(1.0, delay)
        log.info("sleep=%.1fs", delay)
        if _shutdown.wait(delay):
            break

    log.info("scheduler stopped cleanly")
