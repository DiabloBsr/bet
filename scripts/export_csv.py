"""Dump tables to CSV for downstream pandas analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scraper.config import load_settings
from scraper.db import init_engine

TABLES = ["events", "odds_snapshots", "results", "scrape_runs"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export DB tables to CSV.")
    parser.add_argument("--out", type=Path, default=Path("./exports"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    engine = init_engine(settings)

    for table in TABLES:
        df = pd.read_sql_table(table, engine)
        out_path = args.out / f"{table}.csv"
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path} rows={len(df)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
