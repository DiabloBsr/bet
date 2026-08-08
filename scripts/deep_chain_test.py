"""Chaînes profondes : les cellules « gagne toujours » du passé tiennent-elles au futur ?

Le user veut chaîner beaucoup de conditions (équipe × adversaire × heure × …) pour
trouver « telle équipe contre telle équipe à telle heure gagne/perd toujours ».
Le piège : plus on chaîne, plus n s'effondre, plus on trouve des 100%/0% PAR HASARD.

Ce script le DÉMONTRE au lieu de l'affirmer :
  - on construit des chaînes de plus en plus profondes (2 -> 5 conditions) :
      équipe, adversaire, côté (dom/ext), heure, bande de cote.
  - sur le TRAIN (1re moitié chrono), on repère les cellules « gagne 100% » et
    « perd 0% » (avec n_train >= seuil).
  - on regarde ce que ces MÊMES cellules font sur le TEST (2e moitié).
  - une vraie loi resterait à ~100% ; du bruit revient vers la moyenne (~sa cote).

    python scripts/deep_chain_test.py
"""
from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "virtual_sports.db"
MADA = timezone(timedelta(hours=3))


def _read():
    q = ("SELECT e.expected_start, e.team_a, e.team_b, o.odds_home, o.odds_draw, "
         "o.odds_away, r.score_a, r.score_b FROM events e "
         "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f "
         "  ON f.event_id = e.id "
         "JOIN odds_snapshots o ON o.id = f.mid JOIN results r ON r.event_id = e.id "
         "WHERE e.competition = 'InstantLeague-8060' AND r.score_a IS NOT NULL "
         "ORDER BY e.expected_start, e.id")
    for attempt in range(7):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=250)
            c.execute("PRAGMA busy_timeout=250000")
            rows = c.execute(q).fetchall(); c.close(); return rows
        except sqlite3.Error:
            if attempt == 6:
                raise
            time.sleep(10)
    return []


def keys_for(team, opp, side, hr, ob, depth):
    """Chaîne de `depth` conditions (croissante)."""
    parts = [f"t:{team}", f"o:{opp}", f"s:{side}", f"h:{hr}", f"b:{ob}"]
    return "|".join(parts[:depth])


def main():
    rows = [r for r in _read()
            if all(isinstance(x, (int, float)) and 1 < x < 99.5 for x in (r[3], r[4], r[5]))
            and isinstance(r[6], int) and isinstance(r[7], int)]
    n = len(rows); cut = int(n * 0.70)
    print(f"CAN {n} matchs (train {cut}/test {n-cut})")

    for depth in (2, 3, 4, 5):
        # cellule -> [wins_tr, n_tr, wins_te, n_te]
        acc = defaultdict(lambda: [0, 0, 0, 0])
        for i, (es, ta, tb, oh, od, oa, sa, sb) in enumerate(rows):
            try:
                hr = datetime.fromisoformat(str(es)).astimezone(MADA).hour
            except Exception:
                continue
            tr = i < cut
            for team, opp, side, o, gf, ga in (
                    (ta, tb, "H", oh, sa, sb), (tb, ta, "A", oa, sb, sa)):
                ob = int(min(o, 99))            # bande = cote entiere
                k = keys_for(team, opp, side, hr, ob, depth)
                a = acc[k]; w = int(gf > ga)
                if tr:
                    a[0] += w; a[1] += 1
                else:
                    a[2] += w; a[3] += 1

        # cellules "gagne 100%" sur le train (n_tr>=seuil) -> que font-elles au test ?
        for seuil in (3, 5, 8):
            perf, tepool_w, tepool_n = [], 0, 0
            for k, (wt, nt, we, ne) in acc.items():
                if nt >= seuil and wt == nt and ne >= 1:   # 100% gagne au train
                    perf.append(k); tepool_w += we; tepool_n += ne
            if len(perf) < 5:
                continue
            te_rate = tepool_w / tepool_n if tepool_n else 0
            print(f"  depth={depth}  seuil train>={seuil} : {len(perf):>5} cellules 'GAGNE 100%' au train "
                  f"-> au TEST ces memes cellules gagnent {100*te_rate:.1f}% (n_test={tepool_n})", flush=True)
        # symetrique : "perd 100%" (0 victoire)
        for seuil in (5,):
            perf, tw, tn = [], 0, 0
            for k, (wt, nt, we, ne) in acc.items():
                if nt >= seuil and wt == 0 and ne >= 1:
                    tw += we; tn += ne; perf.append(k)
            if len(perf) >= 5 and tn:
                print(f"  depth={depth}  seuil train>={seuil} : {len(perf):>5} cellules 'PERD TOUJOURS' au train "
                      f"-> au TEST elles gagnent {100*tw/tn:.1f}% (n_test={tn})", flush=True)
        print(flush=True)
    print("DEEPCHAIN-DONE", flush=True)


if __name__ == "__main__":
    main()
