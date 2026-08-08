"""Chasse à l'épingle : QUAND une grosse cote (outsider >=10) gagne-t-elle ? — CAN.

On parie l'outsider (cote >=10) à gagner, et on conditionne sur TOUT ce qui est
connu avant le match, sous tous les angles, même bêtes :
  - identité de l'outsider (24 equipes)
  - identité de l'adversaire (24)
  - la PAIRE exacte (qui contre qui)
  - bande de cote de l'outsider (10-20, 20-30, ..., 80-100)
  - série de NON-VICTOIRES de l'outsider avant (k=1..30) — l'idee "il est mur"
  - série SANS MARQUER de l'outsider (k=1..8)
  - série de VICTOIRES de l'adversaire avant (k=2..5)
  - config de cotes
  - croisements : team x nonwin, bande x nonwin, adversaire x nonwin

Filtre : ROI>0 TRAIN et (ROI_test - 1.96*IC)>0 ; n_tr>=150, n_te>=100.
Un survivant est ré-audité (multi-plis + validation) AVANT toute annonce.
Rappel prouvé : l'outsider gagne sa cote devigee, i.i.d., sans memoire.

    python scripts/hunt_bigodds.py
"""
from __future__ import annotations

import json
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "virtual_sports.db"
OUT = ROOT / "data" / "logs" / "bigodds_survivors.jsonl"


def _read():
    q = ("SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away, "
         "r.score_a, r.score_b FROM events e "
         "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f "
         "  ON f.event_id = e.id "
         "JOIN odds_snapshots o ON o.id = f.mid JOIN results r ON r.event_id = e.id "
         "WHERE e.competition = 'InstantLeague-8060' AND r.score_a IS NOT NULL "
         "ORDER BY e.expected_start, e.id")
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


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("", encoding="utf-8")
    rows = [r for r in _read()
            if all(isinstance(x, (int, float)) and 1 < x < 99.5 for x in (r[2], r[3], r[4]))
            and isinstance(r[5], int) and isinstance(r[6], int)]
    n = len(rows)
    cut = int(n * 0.70)
    # historiques par equipe
    res_hist = defaultdict(lambda: deque(maxlen=40))   # (win, scored)
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0, 0, 0.0])  # ntr,str,sstr,nte,ste,sste,wte,pte
    bigcount = 0

    def nowin_run(h):
        r = 0
        for win, sc in reversed(h):
            if win:
                break
            r += 1
        return r

    def nogoal_run(h):
        r = 0
        for win, sc in reversed(h):
            if sc:
                break
            r += 1
        return r

    def winrun(h):
        r = 0
        for win, sc in reversed(h):
            if not win:
                break
            r += 1
        return r

    for i, (ta, tb, oh, od, oa, sa, sb) in enumerate(rows):
        inv = 1 / oh + 1 / od + 1 / oa
        # chaque cote >=10 est un "gros outsider" candidat
        for team, opp, o, gf, ga in ((ta, tb, oh, sa, sb), (tb, ta, oa, sb, sa)):
            if o < 10:
                continue
            bigcount += 1
            win = int(gf > ga)
            roi = win * o - 1
            pdev = (1 / o) / inv
            off = 0 if i < cut else 3
            ob = (f"band:{int(o // 10) * 10}")   # 10,20,...,90
            nw = nowin_run(res_hist[team])
            ng = nogoal_run(res_hist[team])
            ow = winrun(res_hist[opp])
            cfg = f"cfg:{min(int(oh),9)}-{min(int(od),9)}-{min(int(oa),9)}"
            conds = ["ALL", ob, cfg, f"team:{team}", f"opp:{opp}",
                     f"pair:{team}_v_{opp}"]
            for k in (1, 2, 3, 5, 8, 12, 20, 30):
                if nw >= k:
                    conds.append(f"nonwin>={k}")
                    conds.append(f"{ob}|nonwin>={k}")
            for k in (1, 2, 3, 5, 8):
                if ng >= k:
                    conds.append(f"nogoal>={k}")
            for k in (2, 3, 5):
                if ow >= k:
                    conds.append(f"oppwin>={k}")
            for cond in conds:
                a = acc[cond]
                a[off] += 1; a[off + 1] += roi; a[off + 2] += roi * roi
                if off == 3:
                    a[6] += win; a[7] += pdev
        # maj APRES le match (les 2 equipes)
        res_hist[ta].append((int(sa > sb), int(sa > 0)))
        res_hist[tb].append((int(sb > sa), int(sb > 0)))

    survivors = []
    for cond, a in acc.items():
        n_tr, s_tr, _2, n_te, s_te, ss_te, wte, pte = a
        if n_tr < 150 or n_te < 100 or cond == "ALL":
            continue
        rtr = s_tr / n_tr
        if rtr <= 0:
            continue
        rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0)
        se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rte - se > 0:
            survivors.append({"cond": cond, "n_tr": int(n_tr), "n_te": int(n_te),
                              "roi_tr": round(100 * rtr, 2), "roi_te": round(100 * rte, 2),
                              "ci95": round(100 * se, 2),
                              "resid": round(100 * (wte / n_te - pte / n_te), 2)})
    survivors.sort(key=lambda s: -(s["roi_te"] - s["ci95"]))
    with OUT.open("a", encoding="utf-8") as fh:
        for s in survivors:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"{bigcount} instances de grosse cote (>=10), {len(acc)} conditions testees.", flush=True)
    if not survivors:
        print("RESULTAT : AUCUN survivant train+test. Aucune condition ne fait gagner "
              "l'outsider plus que sa cote.", flush=True)
    else:
        print(f"RESULTAT : {len(survivors)} survivant(s) — A RE-AUDITER :", flush=True)
        for s in survivors[:60]:
            print(f"  {s['cond']:<30} n_te={s['n_te']:>5} ROItr {s['roi_tr']:+.1f}% "
                  f"ROIte {s['roi_te']:+.1f}% (+-{s['ci95']:.1f}) residu {s['resid']:+.1f}pp", flush=True)
    print(f"\nsurvivants -> {OUT}", flush=True)
    print("BIGODDS-DONE", flush=True)


if __name__ == "__main__":
    main()
