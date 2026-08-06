"""Chaînages — combine passé × présent (× lui-même) en conditions composées.

Jusqu'ici : conditions ISOLÉES, toutes nulles. Ici on les CHAÎNE : toutes les
paires ET tous les triplets (AND) de features atomiques, sur le pari « favori gagne ».
Inclut le motif des 3 derniers résultats du favori (la séquence chaînée à elle-même)
et la trajectoire de cote (passé→présent).

Features atomiques (toutes connues AVANT le match) :
  PRÉSENT : config 2-3-3 / 3-3-2 / 2-3-2, favori home/away, bande cote favori,
            double-favori, penché Over, penché BTTS.
  PASSÉ   : favori en série W>=2/>=3, série L>=2, favori a gagné/perdu son match
            précédent, cote du favori en baisse (1x, 2x consécutif), adversaire en
            série W>=2 / froid (aucune victoire sur 3), H2H : favori a gagné le
            dernier duel. Motif des 3 derniers résultats du favori (WWW…LLL).

Filtre : ROI>0 TRAIN et (ROI_test − 1.96·IC)>0 ; n_tr>=300, n_te>=200.
Un survivant est ré-audité (multi-plis + placebo) AVANT toute annonce.

    python scripts/hunt_chains.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict, deque
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import predict_trio as pt  # noqa: E402  (_devig_over25, _devig_btts)

DB = ROOT / "data" / "virtual_sports.db"
OUT = ROOT / "data" / "logs" / "chains_survivors.jsonl"
LEAGUES = {"8065": "CDM", "8043": "ALL", "8044": "POR"}


def _read(code):
    q = ("SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away, "
         "o.extra_markets, r.score_a, r.score_b FROM events e "
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


def _streak(hist_team, fn):
    run = 0
    for gf, ga in reversed(hist_team):
        if fn(gf, ga):
            run += 1
        else:
            break
    return run


W = lambda gf, ga: gf > ga
L = lambda gf, ga: gf < ga


def _features(oh, od, oa, xm, fs, fo, ftm, otm, hist, prevres, oddshist, h2hlast):
    """Liste des features actives (connues avant le match)."""
    F = []
    ib = f"{min(int(oh),9)}-{min(int(od),9)}-{min(int(oa),9)}"
    if ib in ("2-3-3", "3-3-2"):
        F.append("cfg_simple")
    if ib == "2-3-2":
        F.append("cfg_2-3-2")
    F.append("fav_H" if fs == "H" else "fav_A")
    F.append("favb_2.0-2.4" if fo < 2.4 else "favb_2.4-3")
    if sum(1 for x in (oh, od, oa) if int(x) == 2) >= 2:
        F.append("double_fav")
    po = pt._devig_over25(xm)
    if po is not None:
        F.append("over_lean" if po >= 0.5 else "under_lean")
    pb = pt._devig_btts(xm)
    if pb is not None and pb >= 0.5:
        F.append("btts_lean")
    # PASSÉ — favori
    sw = _streak(hist[ftm], W)
    if sw >= 2:
        F.append("favW>=2")
    if sw >= 3:
        F.append("favW>=3")
    if _streak(hist[ftm], L) >= 2:
        F.append("favL>=2")
    pr = prevres.get(ftm)
    if pr == "W":
        F.append("favprev_W")
    elif pr == "L":
        F.append("favprev_L")
    oh_prev = oddshist[ftm]
    if len(oh_prev) >= 1 and fo < oh_prev[-1]:
        F.append("cote_baisse1")
        if len(oh_prev) >= 2 and oh_prev[-1] < oh_prev[-2]:
            F.append("cote_baisse2")
    # PASSÉ — adversaire
    if _streak(hist[otm], W) >= 2:
        F.append("oppW>=2")
    if _streak(hist[otm], W) == 0 and _streak(hist[otm], L) >= 1:
        F.append("opp_froid")
    if h2hlast.get((ftm, otm)) == "favW":
        F.append("h2h_favW")
    # motif des 3 derniers résultats du favori (chaîne avec elle-même)
    seq = list(hist[ftm])[-3:]
    if len(seq) == 3:
        pat = "".join("W" if gf > ga else ("L" if gf < ga else "D") for gf, ga in seq)
        F.append(f"seq3_{pat}")
    return F


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("", encoding="utf-8")
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])   # ntr,str,sstr, nte,ste,sste
    for code, nom in LEAGUES.items():
        rows = [r for r in _read(code)
                if all(isinstance(x, (int, float)) and 1 < x < 99 for x in (r[2], r[3], r[4]))
                and isinstance(r[6], int) and isinstance(r[7], int)]
        n = len(rows)
        cut = int(n * 0.70)
        hist = defaultdict(lambda: deque(maxlen=6))
        prevres = {}
        oddshist = defaultdict(lambda: deque(maxlen=4))
        h2hlast = {}
        for i, (ta, tb, oh, od, oa, xm, sa, sb) in enumerate(rows):
            fs = "H" if oh <= oa else "A"
            fo = oh if fs == "H" else oa
            if 1.4 < fo < 4.0:      # favori raisonnable (élargi pour + de chaînes)
                ftm, otm = (ta, tb) if fs == "H" else (tb, ta)
                win = int((sa > sb) if fs == "H" else (sb > sa))
                roi = win * fo - 1
                F = _features(oh, od, oa, xm, fs, fo, ftm, otm, hist, prevres, oddshist, h2hlast)
                off = 0 if i < cut else 3
                # singletons + paires + triplets (chaînes AND)
                combos = list(F)
                combos += ["|".join(c) for c in combinations(sorted(set(F)), 2)]
                combos += ["|".join(c) for c in combinations(sorted(set(F)), 3)]
                for key in combos:
                    a = acc[(nom, key)]
                    a[off] += 1; a[off+1] += roi; a[off+2] += roi*roi
                    ap = acc[("POOL", key)]
                    ap[off] += 1; ap[off+1] += roi; ap[off+2] += roi*roi
            # maj APRÈS
            hist[ta].append((sa, sb)); hist[tb].append((sb, sa))
            prevres[ta] = "W" if sa > sb else ("L" if sa < sb else "D")
            prevres[tb] = "W" if sb > sa else ("L" if sb < sa else "D")
            oddshist[ta].append(oh); oddshist[tb].append(oa)
            fwin = "favW" if ((sa > sb) if oh <= oa else (sb > sa)) else "favLose"
            h2hlast[(ta, tb)] = fwin if oh <= oa else None
            h2hlast[(tb, ta)] = fwin if oa < oh else None
        print(f"  {nom}: {n} matchs traités", flush=True)

    survivors = []
    for (scope, key), a in acc.items():
        n_tr, s_tr, _2, n_te, s_te, ss_te = a
        if n_tr < 300 or n_te < 200:
            continue
        rtr = s_tr / n_tr
        if rtr <= 0:
            continue
        rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0)
        se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rte - se > 0:
            survivors.append({"scope": scope, "chain": key, "n_tr": int(n_tr),
                              "n_te": int(n_te), "roi_tr": round(100*rtr, 2),
                              "roi_te": round(100*rte, 2), "ci95": round(100*se, 2),
                              "depth": key.count("|") + 1})
    survivors.sort(key=lambda s: -(s["roi_te"] - s["ci95"]))
    with OUT.open("a", encoding="utf-8") as fh:
        for s in survivors:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n" + "=" * 78, flush=True)
    print(f"{len(acc)} chaînes testées (singletons + paires + triplets).", flush=True)
    if not survivors:
        print("RESULTAT : AUCUNE chaîne +EV train+test. Le chaînage ne crée pas d'edge.", flush=True)
    else:
        print(f"RESULTAT : {len(survivors)} chaîne(s) survivante(s) — A RE-AUDITER :", flush=True)
        for s in survivors[:50]:
            print(f"  [{s['depth']}] {s['scope']:<5} {s['chain'][:46]:<46} n_te={s['n_te']:>5} "
                  f"tr {s['roi_tr']:+.1f}% te {s['roi_te']:+.1f}% (+-{s['ci95']:.1f})", flush=True)
    print(f"\nsurvivants -> {OUT}", flush=True)
    print("CHAINS-DONE", flush=True)


if __name__ == "__main__":
    main()
