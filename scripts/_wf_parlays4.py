"""Walk-forward parlays v4 — sensibilite de bande pour away MS_early.

Si l edge 'away [2.2,3.2) MS_early' est reel, il doit survivre a un leger
elargissement/retrecissement de la bande de cotes (test de robustesse fixe a
priori, pas une optimisation).
"""
from __future__ import annotations

import sys

sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings


def segment_of(round_info):
    try:
        j = int(round_info)
    except (TypeError, ValueError):
        return '?'
    if j <= 3:
        return 'DS'
    if j <= 12:
        return 'MS_early'
    if j <= 25:
        return 'MS_mid'
    if j <= 33:
        return 'MS_late'
    return 'FS'


def stat(name, sub):
    n = len(sub)
    if n == 0:
        print(f'  {name:<40} n=0')
        return
    wr = sub.won.mean()
    roi = (sub.won * (sub.cote - 1) - (1 - sub.won)).mean()
    flag = '' if n >= 30 else ' [INSTABLE n<30]'
    print(f'  {name:<40} n={n:<5} wr={wr*100:5.1f}%  cote={sub.cote.mean():5.2f}  ROI={roi*100:+6.1f}%{flag}')


def main():
    engine = create_engine(load_settings().db_url)
    df = pd.read_sql(
        """
        SELECT e.id ev_id, e.team_a, e.team_b, e.expected_start, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o
             ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.score_b IS NOT NULL
              AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL
              AND o.odds_away IS NOT NULL
        ORDER BY e.expected_start, e.id
        """,
        engine,
    )
    df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
    df['round_id'] = df['expected_start']
    df['segment'] = df['round_info'].map(segment_of)
    rounds = sorted(df.round_id.unique())
    counts = df.round_id.value_counts()
    cum, cut = 0, rounds[-1]
    for rid in rounds:
        cum += counts[rid]
        if cum >= int(len(df) * 0.70):
            cut = rid
            break

    ms = df[df.segment == 'MS_early']
    for lo, hi in [(2.0, 3.5), (2.2, 3.2), (2.4, 3.0), (2.0, 2.6), (2.6, 3.5)]:
        sub = ms[(ms.odds_away >= lo) & (ms.odds_away < hi)].copy()
        sub['cote'] = sub.odds_away
        sub['won'] = (sub.score_b > sub.score_a).astype(int)
        stat(f'away[{lo},{hi}) MS_early [train]', sub[sub.round_id < cut])
        stat(f'away[{lo},{hi}) MS_early [oos]  ', sub[sub.round_id >= cut])
        print()


if __name__ == '__main__':
    main()
