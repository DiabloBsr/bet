"""Centralized configuration loaded from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    target_url: str
    extra_urls: tuple[str, ...]
    scrape_interval_seconds: int
    db_url: str
    log_level: str
    log_file: Path
    headless: bool
    user_agent: str
    page_timeout_ms: int
    jitter_seconds: int
    league_ids: tuple[str, ...] = ("8035",)

    @property
    def all_urls(self) -> tuple[str, ...]:
        return (self.target_url, *self.extra_urls)


def _env(key: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(key, default)
    if required and not value:
        raise RuntimeError(
            f"Environment variable {key} is required. Copy .env.example to .env."
        )
    return value or ""


def load_settings() -> Settings:
    log_file = Path(_env("LOG_FILE", "./logs/scraper.log"))
    log_file.parent.mkdir(parents=True, exist_ok=True)

    extra_raw = _env("EXTRA_URLS", "")
    extra_urls = tuple(u.strip() for u in extra_raw.split(",") if u.strip())

    leagues_raw = _env("LEAGUE_IDS", "8035")
    league_ids = tuple(x.strip() for x in leagues_raw.split(",") if x.strip())

    return Settings(
        league_ids=league_ids,
        target_url=_env("TARGET_URL", required=True),
        extra_urls=extra_urls,
        scrape_interval_seconds=int(_env("SCRAPE_INTERVAL_SECONDS", "180")),
        db_url=_env("DB_URL", "sqlite:///./data/virtual_sports.db"),
        log_level=_env("LOG_LEVEL", "INFO"),
        log_file=log_file,
        headless=_env("HEADLESS", "true").lower() == "true",
        user_agent=_env(
            "USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        ),
        page_timeout_ms=int(_env("PAGE_TIMEOUT_MS", "30000")),
        jitter_seconds=int(_env("JITTER_SECONDS", "15")),
    )
