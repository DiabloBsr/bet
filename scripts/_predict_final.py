"""Prédiction LIVE avec le moteur FINAL (lookup 2D favori×λtot, le meilleur du zoo).
Affiche par match : favori 1X2, λtot, zone buts, score FINAL (+ score moteur-sim pour
comparaison). Round courant ou [HH:MM].
Usage: ./.venv/Scripts/python.exe scripts/_predict_final.py [HH:MM]
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.score_final import load_table, predict_final, load_buckets, ensemble_top3
from scraper.market_inversion import (
    devig, exact_invert_1x2, apply_sim_deviations, total_distribution, grid_modal_score, _grid_btts_oui,
)

MG = timezone(timedelta(hours=3))
UNDER, OVER = 2.45, 3.13


def main():
    s = load_settings(); e = create_engine(s.db_url); now = datetime.now(timezone.utc)
    table, glob = load_table()
    buckets, gbuck = load_buckets()
    up = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,e.id ev FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MAX(id) FROM odds_snapshots WHERE event_id=e.id)
        LEFT JOIN results r ON r.event_id=e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL AND e.competition='InstantLeague-8035'""", e)
    up["es"] = pd.to_datetime(up.expected_start, utc=True)
    up = up[up.es > now - pd.Timedelta(minutes=3)]
    up["local"] = up.es.dt.tz_convert(MG).dt.strftime("%H:%M")
    up = up.sort_values(["es", "ev"]).drop_duplicates(["team_a", "team_b", "local"])
    rounds = sorted(up.local.unique())
    if not rounds:
        print("Aucun round futur — relance le scraper."); return
    TARGET = sys.argv[1] if len(sys.argv) > 1 else (rounds[1] if len(rounds) > 1 else rounds[0])
    if TARGET not in rounds:
        print(f"Round {TARGET} absent. Dispo : {rounds}"); return
    ms = up[up.local == TARGET]
    print(f"now {now.astimezone(MG):%H:%M} | ROUND {TARGET} — moteur FINAL ({len(ms)} matchs)\n")
    H = f"{'match':<26}{'fav':>9}{'λtot':>6}{'zone':>7}{'  FINAL':>8}    {'TOP-3 VOTÉ (ensemble 50/50, %)'}"
    print(H); print("-" * 92)
    summary = {"under": [], "over": [], "scores": {}}
    for r in ms.itertuples():
        oh, od, oa = float(r.oh), float(r.od), float(r.oa)
        if oh <= 1 or oa <= 1:
            continue
        q1, qX, q2 = devig(oh, od, oa)
        lh, la = exact_invert_1x2(oh, od, oa); lt = lh + la
        g = apply_sim_deviations(lh, la, "cells"); mt = int(total_distribution(g).argmax())
        fin = predict_final(oh, od, oa, table, glob)["score"]
        ens = ensemble_top3(oh, od, oa, buckets, gbuck, top_n=3)
        top3 = "  ".join(f"{s}({p:.0f}%)" for s, p in ens["top"])
        fav = f"X {qX*100:.0f}%" if qX >= max(q1, q2) else (f"1 {q1*100:.0f}%" if q1 > q2 else f"2 {q2*100:.0f}%")
        zone = "⬇U3.5" if lt < UNDER else ("⬆O2.5" if lt >= OVER else f"tot{mt}")
        cover = sum(p for _, p in ens["top"])
        print(f"{(r.team_a+' v '+r.team_b)[:25]:<26}{fav:>9}{lt:>6.2f}{zone:>7}{fin:>8}    {top3}  [{cover:.0f}%]")
        summary["scores"][f"{r.team_a} v {r.team_b}"] = fin
        if lt < UNDER: summary["under"].append(f"{r.team_a} v {r.team_b}")
        elif lt >= OVER: summary["over"].append(f"{r.team_a} v {r.team_b}")
    print(f"\n  ✓ = FINAL et moteur-sim d'accord (confiance max sur le score)")
    if summary["under"]:
        print(f"  ⬇ Under 3.5 : {', '.join(summary['under'])}")
    if summary["over"]:
        print(f"  ⬆ Over 2.5  : {', '.join(summary['over'])}")
    print("\n  Rappel : FINAL = score le PLUS probable (~12% plafond) ; marché efficient, pas de +EV.")


if __name__ == "__main__":
    main()
