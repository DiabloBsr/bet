"""Prédit TOUS les rounds à venir captés (cotes présentes, pas de résultat),
profiler entraîné UNE fois sur l'historique récent. Affiche par round et
sauvegarde les prédictions (data/_upcoming_preds.json) pour vérification ultérieure.

Usage : _predict_upcoming.py [--day 2026-06-28] [--max-rounds 10]
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scraper.config import load_settings
from scraper.shadow_rng.config import merge_config
from scraper.shadow_rng.profiler import DistributionProfiler
from scraper.shadow_rng.simulators import ShadowRNGSimulator, build_transition_matrix, score_list
from scraper.shadow_rng.ensemble import EnsembleVoter

LG = "InstantLeague-8035"
MAXG = 7
SC = score_list(MAXG)
HD = np.array([int(s.split("-")[0]) for s in SC]); AD = np.array([int(s.split("-")[1]) for s in SC])
M_H, M_X, M_A = (HD > AD).astype(float), (HD == AD).astype(float), (HD < AD).astype(float)

_UPCOMING = """
SELECT e.id event_id, e.expected_start ts, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
LEFT JOIN results r ON r.event_id=e.id
WHERE e.competition=:lg AND r.id IS NULL AND e.expected_start LIKE :day
  AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
ORDER BY e.expected_start, e.id
"""
_HIST = """
SELECT o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE e.competition=:lg AND r.score_a IS NOT NULL
ORDER BY e.expected_start DESC LIMIT :n
"""
_LASTSCORE = """
SELECT r.score_a sa, r.score_b sb FROM events e JOIN results r ON r.event_id=e.id
WHERE e.competition=:lg AND r.score_a IS NOT NULL ORDER BY e.expected_start DESC LIMIT 1
"""


def x12_of(vec):
    p = (float(vec @ M_H), float(vec @ M_X), float(vec @ M_A))
    j = int(np.argmax(p)); return ("1", "X", "2")[j], p[j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2026-06-28")
    ap.add_argument("--max-rounds", type=int, default=12)
    ap.add_argument("--hist", type=int, default=2000)
    a = ap.parse_args()
    eng = create_engine(load_settings().db_url)

    up = pd.read_sql(text(_UPCOMING), eng, params={"lg": LG, "day": a.day + "%"})
    if up.empty:
        print(f"Aucun round à venir capté pour {a.day} (relance/attends le scraper).") ; return
    rounds = list(dict.fromkeys(up.ts.tolist()))[:a.max_rounds]
    up = up[up.ts.isin(rounds)]

    hist = pd.read_sql(text(_HIST), eng, params={"lg": LG, "n": a.hist}).iloc[::-1].reset_index(drop=True)
    last = pd.read_sql(text(_LASTSCORE), eng, params={"lg": LG})
    last_score = (f"{min(int(last.sa.iloc[0]),6)}-{min(int(last.sb.iloc[0]),6)}" if len(last) else None)

    cfg = merge_config(None); ec = cfg["ensemble"]
    prof = DistributionProfiler(cfg).fit(hist)
    snap_main = prof.get_full_snapshot(window=ec["main_window"])
    snap_conf = prof.get_full_snapshot(window=ec["confirm_window"])
    T, marg = build_transition_matrix(prof._real_idx, MAXG ** 2, 1,
                                      cfg["simulators"]["memory_smoothing"], cfg["simulators"]["memory_min_count"])
    for s in (snap_main, snap_conf):
        s["transition"] = {"matrix": T, "marginal": marg}
    sim = ShadowRNGSimulator(cfg).set_profiler_snapshot(snap_main)
    voter = EnsembleVoter(cfg)

    print(f"\n{'='*86}")
    print(f"  ROUNDS À VENIR — {a.day}  ({len(rounds)} rounds, profiler OOS sur {len(hist)} matchs)")
    print(f"  régime={snap_main['regime']['regime']}  anomalies={snap_main['anomalies']}")
    print(f"{'='*86}")

    preds = {}
    for ts in rounds:
        loc = (pd.to_datetime(ts) + pd.Timedelta(hours=2)).strftime("%H:%M")
        print(f"\n  >>> ROUND {loc} (heure locale +2)   [{ts} UTC]")
        print(f"  {'match':<26} {'cotes(1/X/2)':<17} {'Top-3 score':<18} {'1X2'}")
        print(f"  {'-'*26} {'-'*17} {'-'*18} {'-'*7}")
        for r in up[up.ts == ts].itertuples():
            res = sim.simulate_all((float(r.oh), float(r.od), float(r.oa)),
                                   profiler_snapshot=snap_main, last_score=last_score)
            out = voter.format_output({"event_id": r.event_id, "league": LG,
                                       "home_team": r.home, "away_team": r.away}, res, snap_main, snap_conf)
            top3 = [c["score"] for c in out["consensus_top3"]]
            xp, xpp = x12_of(np.asarray(out["consensus_full"], float))
            preds[str(r.event_id)] = {"ts": ts, "home": r.home, "away": r.away,
                                      "top3": top3, "x12": xp, "x12_prob": round(xpp, 3)}
            match = f"{str(r.home)[:12]} v {str(r.away)[:10]}"
            cotes = f"{r.oh:.2f}/{r.od:.2f}/{r.oa:.2f}"
            print(f"  {match:<26} {cotes:<17} {'  '.join(top3):<18} {xp}({100*xpp:.0f}%)")

    out_path = Path(__file__).resolve().parents[1] / "data" / "_upcoming_preds.json"
    out_path.write_text(json.dumps(preds, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*86}")
    print(f"  {len(preds)} prédictions sauvegardées -> {out_path.name} (pour vérification)")
    print(f"{'='*86}\n")


if __name__ == "__main__":
    main()
