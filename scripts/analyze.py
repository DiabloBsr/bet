"""Basic descriptive statistics on the collected data (étape 7).

Calcule :
  - fréquence des issues domicile / nul / extérieur
  - distribution des scores finaux
  - évolution des cotes (échantillon d'un event)
  - taux de valeurs manquantes côté odds
  - détection de doublons sur events
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from scraper.config import load_settings
from scraper.db import init_engine


def _outcome(row) -> str:
    if row.score_a > row.score_b:
        return "home"
    if row.score_a < row.score_b:
        return "away"
    return "draw"


def main() -> int:
    settings = load_settings()
    engine = init_engine(settings)

    events = pd.read_sql_table("events", engine)
    odds = pd.read_sql_table("odds_snapshots", engine)
    results = pd.read_sql_table("results", engine)

    print(f"== events: {len(events)} | odds_snapshots: {len(odds)} | results: {len(results)} ==\n")

    if not results.empty:
        results = results.assign(issue=results.apply(_outcome, axis=1))
        print("Outcome frequency:")
        print(results["issue"].value_counts(normalize=True).round(3).to_string())

        print("\nTop 10 scores:")
        scores = results["score_a"].astype(str) + "-" + results["score_b"].astype(str)
        print(scores.value_counts().head(10).to_string())

    if not odds.empty:
        print("\nOdds null rate per column:")
        print(odds[["odds_home", "odds_draw", "odds_away"]].isna().mean().round(3).to_string())

        sample_event = odds["event_id"].iloc[0]
        sub = (
            odds[odds["event_id"] == sample_event]
            .sort_values("captured_at")
            [["captured_at", "odds_home", "odds_draw", "odds_away", "status"]]
        )
        print(f"\nOdds evolution for event_id={sample_event}:")
        print(sub.to_string(index=False))

    if not events.empty:
        dup_mask = events.duplicated(subset=["external_id", "source_url"], keep=False)
        # exclut les NULL en external_id (qui ne sont pas des doublons sémantiques)
        real_dups = events[dup_mask & events["external_id"].notna()]
        print(f"\nDuplicate events (same external_id+source_url): {len(real_dups)}")
        print(f"Missing external_id ratio: {events['external_id'].isna().mean():.2%}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
