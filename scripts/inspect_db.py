"""Quick read-only inspection of the SQLite DB."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scraper.config import load_settings


def main() -> int:
    settings = load_settings()
    db_path = settings.db_url.replace("sqlite:///", "").lstrip("/")
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row

    for table in ("scrape_runs", "events", "odds_snapshots", "results", "rankings_snapshots"):
        n = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}: {n} rows")

    print("\n=== all events ===")
    for r in con.execute(
        "SELECT id, external_id, team_a, team_b, round_info, competition FROM events ORDER BY round_info"
    ):
        print(dict(r))

    print("\n=== results table ===")
    for r in con.execute(
        "SELECT event_id, score_a, score_b, raw_score, finished_at FROM results"
    ):
        print(dict(r))

    print("\n=== rankings_snapshots (top 5 by position) ===")
    for r in con.execute(
        "SELECT competition, team_name, position, points, won, lost, draw "
        "FROM rankings_snapshots ORDER BY position LIMIT 5"
    ):
        print(dict(r))

    print("\n=== sample odds_snapshots ===")
    for r in con.execute(
        "SELECT event_id, odds_home, odds_draw, odds_away, status, captured_at "
        "FROM odds_snapshots LIMIT 5"
    ):
        print(dict(r))

    print("\n=== extra markets (first non-null row) ===")
    row = con.execute(
        "SELECT extra_markets FROM odds_snapshots WHERE extra_markets IS NOT NULL LIMIT 1"
    ).fetchone()
    if row:
        em = json.loads(row[0])
        print(f"markets available: {list(em.keys())}")
        for k, v in list(em.items())[:5]:
            print(f"  {k}: {v}")

    print("\n=== scrape_runs ===")
    for r in con.execute(
        "SELECT id, started_at, finished_at, status, events_seen, snapshots_inserted, results_inserted "
        "FROM scrape_runs"
    ):
        print(dict(r))

    return 0


if __name__ == "__main__":
    sys.exit(main())
