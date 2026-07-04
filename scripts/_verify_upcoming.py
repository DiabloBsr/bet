"""Vérifie les prédictions de _predict_upcoming.py contre les résultats réels
dès qu'ils arrivent en base. Affiche par round + agrégat (score exact / 1X2).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"


def x12(sc):
    h, a = map(int, sc.split("-")); return "1" if h > a else ("X" if h == a else "2")


def main():
    preds = json.loads((Path(__file__).resolve().parents[1] / "data" / "_upcoming_preds.json")
                       .read_text(encoding="utf-8"))
    eng = create_engine(load_settings().db_url)
    ids = [int(k) for k in preds]
    res = pd.read_sql(text(
        "SELECT event_id, score_a sa, score_b sb FROM results "
        "WHERE event_id IN :ids AND score_a IS NOT NULL").bindparams(
            __import__("sqlalchemy").bindparam("ids", expanding=True)),
        eng, params={"ids": ids})
    real = {int(r.event_id): f"{min(int(r.sa),6)}-{min(int(r.sb),6)}" for r in res.itertuples()}

    rounds = {}
    g1 = g3 = gx = gtot = 0
    for eid, p in preds.items():
        if int(eid) not in real:
            continue
        a = real[int(eid)]
        h1 = a == p["top3"][0]; h3 = a in p["top3"]; hx = p["x12"] == x12(a)
        d = rounds.setdefault(p["ts"], {"n": 0, "h1": 0, "h3": 0, "hx": 0, "rows": []})
        d["n"] += 1; d["h1"] += h1; d["h3"] += h3; d["hx"] += hx
        d["rows"].append((p["home"], p["away"], p["top3"], p["x12"], a))
        g1 += h1; g3 += h3; gx += hx; gtot += 1

    print(f"\n{'='*78}")
    print(f"  VÉRIFICATION DES ROUNDS À VENIR — {gtot}/{len(preds)} matchs déjà réglés")
    print(f"{'='*78}")
    for ts in sorted(rounds):
        d = rounds[ts]
        loc = (pd.to_datetime(ts) + pd.Timedelta(hours=2)).strftime("%H:%M")
        print(f"\n  ROUND {loc}  ({d['n']} matchs) : "
              f"exact T1 {d['h1']}/{d['n']} | Top-3 {d['h3']}/{d['n']} | 1X2 {d['hx']}/{d['n']}")
        for home, away, top3, xp, a in d["rows"]:
            mark = ("T1" if a == top3[0] else ("t3" if a in top3 else "  "))
            xm = "OK" if xp == x12(a) else "x "
            print(f"      {str(home)[:12]:<12} v {str(away)[:11]:<11}  pred {top3[0]}/{xp}  -> {a}  [{mark} 1X2:{xm}]")
    if gtot:
        print(f"\n{'='*78}")
        print(f"  AGRÉGAT {gtot} matchs : score exact Top-1 {g1}/{gtot} ({100*g1/gtot:.1f}%) | "
              f"Top-3 {g3}/{gtot} ({100*g3/gtot:.1f}%) | 1X2 {gx}/{gtot} ({100*gx/gtot:.1f}%)")
        print(f"{'='*78}\n")
    else:
        print("  (aucun résultat encore — attends que les rounds se jouent)\n")


if __name__ == "__main__":
    main()
