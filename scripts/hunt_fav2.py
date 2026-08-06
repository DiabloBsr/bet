"""Chasse MAXIMALE sur le favori côte 2.x — CDM + ALL + POR, grille la + fine possible.

Le pari fixe : parier le FAVORI (min cote home/away) quand sa cote est dans [2.0, 3.0).
On le décline sous le plus grand nombre de configurations imaginables, et on ne garde
que ce qui est rentable sur le passé ET le futur (ROI_test − 1.96·IC > 0, ROI_train > 0).

Dimensions croisées (des milliers de cellules) :
  - bande de cote du favori (0.1 de large : 2.0-2.1 ... 2.9-3.0)
  - bande de cote du NUL (fine)
  - bande de cote de l'OUTSIDER (fine)
  - favori à domicile / extérieur
  - config exacte (plancher) ET config fine (demi-point)
  - 2D : (bande favori × bande nul), (bande favori × côté), (bande favori × config)
  - séries d'équipe du favori : W/L/marqué/muet/CS/encaissé × k=1..4
  - 2D : (config × côté), (bande favori × série)

Tout survivant est ré-audité (6 plis + placebo) AVANT d'être annoncé.
Pool des 3 ligues pour maximiser n ; split chrono 70/30 PAR ligue.

    python scripts/hunt_fav2.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "virtual_sports.db"
OUT = ROOT / "data" / "logs" / "fav2_survivors.jsonl"
LEAGUES = {"8065": "CDM", "8043": "ALL", "8044": "POR"}


def _read(code):
    q = ("SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away, "
         "r.score_a, r.score_b FROM events e "
         "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f "
         "  ON f.event_id = e.id "
         "JOIN odds_snapshots o ON o.id = f.mid JOIN results r ON r.event_id = e.id "
         f"WHERE e.competition = 'InstantLeague-{code}' AND r.score_a IS NOT NULL "
         "ORDER BY e.id")
    for attempt in range(7):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=250)
            c.execute("PRAGMA busy_timeout=250000")
            rows = c.execute(q).fetchall()
            c.close()
            return rows
        except sqlite3.Error:
            if attempt == 6:
                raise
            time.sleep(10)
    return []


def _bin(x, w):
    return f"{int(x / w) * w:.2f}"


DEFS = {"W": lambda gf, ga: gf > ga, "L": lambda gf, ga: gf < ga,
        "GF": lambda gf, ga: gf > 0, "NG": lambda gf, ga: gf == 0,
        "CS": lambda gf, ga: ga == 0, "GA": lambda gf, ga: ga > 0}


def _conds(oh, od, oa, fav_side, fav_team, hist):
    """Toutes les étiquettes de condition d'un match à favori côte 2."""
    fo = oh if fav_side == "H" else oa
    fb = _bin(fo, 0.1)                       # bande favori 0.1
    db = _bin(od, 0.25)                      # bande nul
    ob = _bin(oa if fav_side == "H" else oh, 0.5)   # bande outsider
    cfg = f"{min(int(oh),9)}-{min(int(od),9)}-{min(int(oa),9)}"
    cfg2 = f"{_bin(oh,0.5)}/{_bin(od,0.5)}/{_bin(oa,0.5)}"
    C = ["ALL",
         f"favb:{fb}", f"drawb:{db}", f"dogb:{ob}", f"side:{fav_side}",
         f"cfg:{cfg}", f"cfg2:{cfg2}",
         f"favb:{fb}|side:{fav_side}", f"favb:{fb}|drawb:{db}",
         f"favb:{fb}|cfg:{cfg}", f"cfg:{cfg}|side:{fav_side}"]
    # séries de l'équipe favorite
    for nm, fn in DEFS.items():
        run = 0
        for gf, ga in reversed(hist[fav_team]):
            if fn(gf, ga):
                run += 1
            else:
                break
        for k in (1, 2, 3, 4):
            if run >= k:
                C.append(f"fav:{nm}>={k}")
                C.append(f"favb:{fb}|{nm}>={k}")
    return C


def hunt():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("", encoding="utf-8")
    # cle -> [n_tr,s_tr,ss_tr, n_te,s_te,ss_te]
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])
    ncells_matches = 0
    for code, nom in LEAGUES.items():
        rows = _read(code)
        R = [(ta, tb, oh, od, oa, sa, sb) for ta, tb, oh, od, oa, sa, sb in rows
             if all(isinstance(x, (int, float)) and 1 < x < 99 for x in (oh, od, oa))
             and isinstance(sa, int) and isinstance(sb, int)]
        n = len(R)
        cut = int(n * 0.70)
        hist = defaultdict(lambda: deque(maxlen=6))
        used = 0
        for i, (ta, tb, oh, od, oa, sa, sb) in enumerate(R):
            fav_side = "H" if oh <= oa else "A"
            fo = oh if fav_side == "H" else oa
            if 2.0 <= fo < 3.0:
                used += 1
                fav_team = ta if fav_side == "H" else tb
                win = int((sa > sb) if fav_side == "H" else (sb > sa))
                roi = win * fo - 1
                off = 0 if i < cut else 3
                for cond in _conds(oh, od, oa, fav_side, fav_team, hist):
                    a = acc[(nom, cond)]
                    a[off] += 1
                    a[off + 1] += roi
                    a[off + 2] += roi * roi
                    # cellule POOLée (3 ligues)
                    p = acc[("POOL", cond)]
                    p[off] += 1
                    p[off + 1] += roi
                    p[off + 2] += roi * roi
            hist[ta].append((sa, sb))
            hist[tb].append((sb, sa))
        ncells_matches += used
        print(f"  {nom}: {n} matchs, {used} favoris cote-2", flush=True)

    survivors = []
    for (scope, cond), a in acc.items():
        n_tr, s_tr, _s2t, n_te, s_te, ss_te = a
        if n_tr < 300 or n_te < 200:
            continue
        rtr = s_tr / n_tr
        if rtr <= 0:
            continue
        rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0)
        se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rte - se > 0:
            survivors.append({"scope": scope, "cond": cond, "n_tr": int(n_tr),
                              "n_te": int(n_te), "roi_tr": round(100 * rtr, 2),
                              "roi_te": round(100 * rte, 2), "ci95": round(100 * se, 2)})
    survivors.sort(key=lambda s: -(s["roi_te"] - s["ci95"]))
    with OUT.open("a", encoding="utf-8") as fh:
        for s in survivors:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n" + "=" * 78, flush=True)
    print(f"{len(acc)} cellules testees (favori cote-2, grille fine + series).", flush=True)
    if not survivors:
        print("RESULTAT : AUCUN survivant train+test. Mur total sur le favori cote-2.", flush=True)
    else:
        print(f"RESULTAT : {len(survivors)} survivant(s) — A RE-AUDITER :", flush=True)
        for s in survivors[:50]:
            print(f"  {s['scope']:<5} {s['cond']:<28} n_te={s['n_te']:>5} "
                  f"ROItr {s['roi_tr']:+.1f}% ROIte {s['roi_te']:+.1f}% (+-{s['ci95']:.1f})", flush=True)
    print(f"\nsurvivants -> {OUT}", flush=True)
    print("FAV2-DONE", flush=True)


if __name__ == "__main__":
    hunt()
