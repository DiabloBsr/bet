"""Simulation Shadow-RNG d'UN round (ensemble de matchs à un instant donné),
en OUT-OF-SAMPLE (profiler entraîné seulement sur l'historique AVANT le round).

Affiche, par match : cotes, consensus Top-3 score exact, pick 1X2, puis (si le
résultat existe) le réel + si la simulation est tombée juste. Résumé du round.

Usage : _predict_round.py "2026-06-19 18:23:56"
"""
from __future__ import annotations
import argparse
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
HD = np.array([int(s.split("-")[0]) for s in SC])
AD = np.array([int(s.split("-")[1]) for s in SC])
M_HOME, M_DRAW, M_AWAY = (HD > AD).astype(float), (HD == AD).astype(float), (HD < AD).astype(float)

_ROUND = """
SELECT e.id event_id, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
LEFT JOIN results r ON r.event_id=e.id
WHERE e.competition=:lg AND e.expected_start LIKE :tslike
ORDER BY e.id
"""
_HIST = """
SELECT o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE e.competition=:lg AND r.score_a IS NOT NULL AND e.expected_start < :ts
ORDER BY e.expected_start DESC LIMIT :n
"""
_LAST = """
SELECT r.score_a sa, r.score_b sb FROM events e JOIN results r ON r.event_id=e.id
WHERE e.competition=:lg AND r.score_a IS NOT NULL AND e.expected_start < :ts
ORDER BY e.expected_start DESC LIMIT 1
"""


def x12_of(vec):
    p = (float(vec @ M_HOME), float(vec @ M_DRAW), float(vec @ M_AWAY))
    j = int(np.argmax(p))
    return ("1", "X", "2")[j], p[j]


def actual_x12(sc):
    h, a = map(int, sc.split("-"))
    return "1" if h > a else ("X" if h == a else "2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ts", nargs="?", default="2026-06-19 18:23:56")
    ap.add_argument("--hist", type=int, default=2000)
    a = ap.parse_args()
    eng = create_engine(load_settings().db_url)

    rnd = pd.read_sql(text(_ROUND), eng, params={"lg": LG, "tslike": a.ts + "%"})
    if rnd.empty:
        print(f"Aucun match au round {a.ts}"); return
    hist = pd.read_sql(text(_HIST), eng, params={"lg": LG, "ts": a.ts, "n": a.hist})
    hist = hist.iloc[::-1].reset_index(drop=True)
    last = pd.read_sql(text(_LAST), eng, params={"lg": LG, "ts": a.ts})
    last_score = (f"{min(int(last.sa.iloc[0]),6)}-{min(int(last.sb.iloc[0]),6)}"
                  if len(last) else None)

    cfg = merge_config(None)
    ec = cfg["ensemble"]
    prof = DistributionProfiler(cfg).fit(hist)
    snap_main = prof.get_full_snapshot(window=ec["main_window"])
    snap_conf = prof.get_full_snapshot(window=ec["confirm_window"])
    T, marg = build_transition_matrix(prof._real_idx, MAXG ** 2, 1,
                                      cfg["simulators"]["memory_smoothing"],
                                      cfg["simulators"]["memory_min_count"])
    for s in (snap_main, snap_conf):
        s["transition"] = {"matrix": T, "marginal": marg}
    sim = ShadowRNGSimulator(cfg).set_profiler_snapshot(snap_main)
    voter = EnsembleVoter(cfg)

    print(f"\n{'='*92}")
    print(f"  SIMULATION DU ROUND {a.ts}  (OUT-OF-SAMPLE — profiler sur {len(hist)} matchs AVANT)")
    print(f"  régime={snap_main['regime']['regime']}  anomalies={snap_main['anomalies']}  last_score={last_score}")
    print(f"{'='*92}")
    print(f"  {'match':<28} {'cotes(1/X/2)':<17} {'consensus Top-3':<22} {'1X2':<8} {'RÉEL':<10} {'verdict'}")
    print(f"  {'-'*28} {'-'*17} {'-'*22} {'-'*8} {'-'*10} {'-'*16}")

    n1 = n3 = nx = ntot = 0
    for r in rnd.itertuples():
        if not (r.oh > 1 and r.od > 1 and r.oa > 1):
            continue
        res = sim.simulate_all((float(r.oh), float(r.od), float(r.oa)),
                               profiler_snapshot=snap_main, last_score=last_score)
        out = voter.format_output({"event_id": r.event_id, "league": LG,
                                   "home_team": r.home, "away_team": r.away},
                                  res, snap_main, snap_conf)
        top3 = out["consensus_top3"]
        cv = np.asarray(out["consensus_full"], float)
        xp, xpp = x12_of(cv)
        t3s = "  ".join(f"{c['score']}" for c in top3)
        cotes = f"{r.oh:.2f}/{r.od:.2f}/{r.oa:.2f}"
        match = f"{str(r.home)[:13]} v {str(r.away)[:11]}"
        if r.sa is not None and not pd.isna(r.sa):
            ntot += 1
            real = f"{min(int(r.sa),6)}-{min(int(r.sb),6)}"
            rx = actual_x12(real)
            h1 = real == top3[0]["score"]; h3 = real in [c["score"] for c in top3]; hx = xp == rx
            n1 += h1; n3 += h3; nx += hx
            verdict = ("EXACT-T1 " if h1 else ("top3 " if h3 else "")) + ("1X2-OK" if hx else "1X2-x")
            real_str = f"{real}({rx})"
        else:
            verdict = "(à venir)"; real_str = "-"
        print(f"  {match:<28} {cotes:<17} {t3s:<22} {xp}({100*xpp:.0f}%)  {real_str:<10} {verdict}")

    if ntot:
        print(f"  {'-'*92}")
        print(f"  RÉSUMÉ {ntot} matchs : score exact Top-1 {n1}/{ntot} ({100*n1/ntot:.0f}%)  |  "
              f"Top-3 {n3}/{ntot} ({100*n3/ntot:.0f}%)  |  1X2 {nx}/{ntot} ({100*nx/ntot:.0f}%)")
    print(f"{'='*92}\n")


if __name__ == "__main__":
    main()
