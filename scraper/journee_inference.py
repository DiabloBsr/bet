"""Inférence de la journée courante pour les events J0 (non assignés par le scraper).

Stratégie : regarder le dernier match fini avec roundNumber valide,
puis estimer la journée du prochain round en cours.
"""
from __future__ import annotations
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.engine import Engine
import pandas as pd


def infer_current_journee(engine: Engine, event_expected_start: datetime | None = None) -> int | None:
    """Retourne la journée la plus probable pour un event J0 à venir.

    Logique :
    1. Cherche le dernier match fini avec round_info != '0' (vraie journée connue)
    2. Compte le nombre d'events scrapés (J0 ou non) entre ce match fini et `event_expected_start`
    3. Ajoute ce delta à la dernière journée connue
    4. Module 38 pour gérer le cycle saison
    """
    # 1. Dernier match fini avec roundNumber valide
    q = """
        SELECT e.round_info, e.expected_start
        FROM events e
        JOIN results r ON r.event_id = e.id
        WHERE e.round_info IS NOT NULL AND e.round_info != '0' AND r.score_a IS NOT NULL AND e.competition = 'InstantLeague-8035'
        ORDER BY r.finished_at DESC LIMIT 1
    """
    row = pd.read_sql(q, engine)
    if row.empty:
        return None
    last_journee = int(row.iloc[0].round_info)
    last_finished_at = pd.to_datetime(row.iloc[0].expected_start)

    # 2. Si event_expected_start fourni, compter les rounds entre les 2 timestamps
    if event_expected_start is None:
        # Retourne la dernière connue + 1 (round suivant probable)
        return ((last_journee) % 38) + 1

    # Convertir en datetime naive pour comparer
    if hasattr(event_expected_start, "tz_localize"):
        event_ts = event_expected_start.tz_localize(None) if event_expected_start.tzinfo else event_expected_start
    else:
        try:
            event_ts = pd.to_datetime(event_expected_start, utc=True).tz_localize(None)
        except Exception:
            event_ts = event_expected_start
    if hasattr(last_finished_at, "tz_localize"):
        last_ts = last_finished_at.tz_localize(None) if last_finished_at.tzinfo else last_finished_at
    else:
        last_ts = last_finished_at

    # 3. Compter distinct rounds entre last_finished_at et event_expected_start
    last_iso = last_ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(last_ts, "strftime") else str(last_ts)
    event_iso = event_ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(event_ts, "strftime") else str(event_ts)
    q2 = f"""
        SELECT COUNT(DISTINCT expected_start) as n_rounds
        FROM events
        WHERE competition = 'InstantLeague-8035' AND expected_start > '{last_iso}' AND expected_start <= '{event_iso}'
    """
    df = pd.read_sql(q2, engine)
    n_steps = int(df.iloc[0].n_rounds) if not df.empty else 1

    # 4. Estimer la journée
    estimated = last_journee + n_steps
    while estimated > 38:
        estimated -= 38  # wrap autour du cycle saison
    return estimated
