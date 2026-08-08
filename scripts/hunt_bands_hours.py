"""Balayage BANDE DE COTE × HEURE, tous marchés, toutes ligues — y a-t-il une tranche
ou une heure rentable ?

Deux questions du user :
  1. Une TRANCHE DE COTE gagne-t-elle tout le temps (ROI>0) sur toutes les données ?
     -> pour CHAQUE offre de CHAQUE marché (1X2 + extra_markets à prédicat testé),
        on bucketise par bande de cote et on mesure le ROI réel (cote offerte).
  2. Y a-t-il un PIC D'HEURE où les cotes dévigées « se retournent » et l'outsider
     gagne plus ? -> ROI de l'outsider (1X2) par heure Mada, et par bande × heure.

Filtre : n_tr>=300, n_te>=200, ROI>0 train ET (test-1.96*IC)>0.
Accumulateurs glissants (RAM constante), une ligue en mémoire à la fois.

    python scripts/hunt_bands_hours.py
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import cross_market_check as cm  # noqa: E402

DB = ROOT / "data" / "virtual_sports.db"
MADA = timezone(timedelta(hours=3))
LEAGUES = {"8035": "ANG", "8036": "ITA", "8037": "ESP", "8042": "FRA", "8043": "ALL",
           "8044": "POR", "8056": "UCL", "8065": "CDM", "8060": "CAN"}
BANDS = [(1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 8.0),
         (8.0, 15.0), (15.0, 30.0), (30.0, 60.0), (60.0, 100.0)]


def band(o):
    for lo, hi in BANDS:
        if lo <= o < hi:
            return f"{lo:g}-{hi:g}"
    return None


def _read(code):
    q = ("SELECT e.expected_start, o.odds_home, o.odds_draw, o.odds_away, "
         "o.extra_markets, r.score_a, r.score_b FROM events e "
         "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f "
         "  ON f.event_id = e.id "
         "JOIN odds_snapshots o ON o.id = f.mid JOIN results r ON r.event_id = e.id "
         f"WHERE e.competition = 'InstantLeague-{code}' AND r.score_a IS NOT NULL "
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


def add(acc, key, off, roi):
    a = acc[key]; a[off] += 1; a[off + 1] += roi; a[off + 2] += roi * roi


def main():
    band_all = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])   # bande (tous marchés)
    hour_out = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])   # heure (outsider 1X2)
    bh_out = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])     # bande×heure (outsider)
    mkt_band = defaultdict(lambda: [0, 0.0, 0.0, 0, 0.0, 0.0])   # marché|bande
    ntot = 0
    for code, nom in LEAGUES.items():
        rows = _read(code)
        rows = [r for r in rows if all(isinstance(x, (int, float)) and 1 < x < 99.5
                for x in (r[1], r[2], r[3])) and isinstance(r[5], int) and isinstance(r[6], int)]
        n = len(rows); cut = int(n * 0.70); ntot += n
        for i, (es, oh, od, oa, xm, sa, sb) in enumerate(rows):
            off = 0 if i < cut else 3
            try:
                hr = datetime.fromisoformat(str(es)).astimezone(MADA).hour
            except Exception:
                hr = -1
            # 1X2 (colonnes)
            res = "1" if sa > sb else ("2" if sb > sa else "X")
            for lbl, o in (("1", oh), ("X", od), ("2", oa)):
                b = band(o)
                if b:
                    roi = (o - 1) if lbl == res else -1
                    add(band_all, b, off, roi)
                    add(mkt_band, f"1X2|{b}", off, roi)
            # outsider (max cote entre home/away) par heure et bande×heure
            if oh >= oa:
                o_out, w = oh, int(sa > sb)
            else:
                o_out, w = oa, int(sb > sa)
            roi = w * o_out - 1
            if hr >= 0:
                add(hour_out, f"h{hr:02d}", off, roi)
                bb = band(o_out)
                if bb:
                    add(bh_out, f"{bb}|h{hr:02d}", off, roi)
            # extra_markets (prédicats testés)
            try:
                mk = json.loads(xm) if isinstance(xm, str) else (xm or {})
            except Exception:
                mk = {}
            for nomk, cle, fab in cm.MARCHES:
                m = cm._market(mk, cle)
                if not isinstance(m, dict):
                    continue
                for label, o in m.items():
                    if not isinstance(o, (int, float)) or not 1 < o < 99.5:
                        continue
                    pred = fab(label)
                    if pred is None:
                        continue
                    b = band(o)
                    if not b:
                        continue
                    roi = (o - 1) if pred(sa, sb) else -1
                    add(band_all, b, off, roi)
                    add(mkt_band, f"{nomk}|{b}", off, roi)
        print(f"  {nom}: {n} matchs", flush=True)

    def report(title, acc, topneg=True):
        print(f"\n=== {title} ===", flush=True)
        out = []
        for k, a in acc.items():
            n_tr, s_tr, _2, n_te, s_te, ss_te = a
            if n_tr < 300 or n_te < 200:
                continue
            rtr = s_tr / n_tr; rte = s_te / n_te
            var = max(ss_te / n_te - rte * rte, 0.0)
            se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
            out.append((k, n_tr + n_te, 100 * rtr, 100 * rte, 100 * se, rtr > 0 and rte - se > 0))
        for k, nn, rtr, rte, se, surv in sorted(out, key=lambda x: -(x[3])):
            flag = "  <<< +EV train+test" if surv else ""
            print(f"  {k:<22} n={nn:>8} ROItr {rtr:+.1f}% ROIte {rte:+.1f}% (+-{se:.1f}){flag}", flush=True)
        return any(s for *_, s in out)

    print(f"\n{ntot} matchs, 9 ligues.", flush=True)
    s1 = report("ROI par BANDE DE COTE (tous marchés confondus)", band_all)
    s2 = report("ROI de l'OUTSIDER par HEURE (Mada)", hour_out)
    # bande×heure : ne montrer que les +EV eventuels
    surv_bh = []
    for k, a in bh_out.items():
        n_tr, s_tr, _2, n_te, s_te, ss_te = a
        if n_tr < 300 or n_te < 200:
            continue
        rtr = s_tr / n_tr; rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0); se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rtr > 0 and rte - se > 0:
            surv_bh.append((k, rtr, rte, se))
    print(f"\n=== BANDE×HEURE outsider : {len(surv_bh)} survivant(s) +EV ===", flush=True)
    for k, rtr, rte, se in surv_bh[:30]:
        print(f"  {k:<18} ROItr {100*rtr:+.1f}% ROIte {100*rte:+.1f}% (+-{100*se:.1f})", flush=True)
    # marché×bande : survivants
    surv_mb = []
    for k, a in mkt_band.items():
        n_tr, s_tr, _2, n_te, s_te, ss_te = a
        if n_tr < 300 or n_te < 200:
            continue
        rtr = s_tr / n_tr; rte = s_te / n_te
        var = max(ss_te / n_te - rte * rte, 0.0); se = 1.96 * (var ** 0.5) / np.sqrt(n_te)
        if rtr > 0 and rte - se > 0:
            surv_mb.append((k, rtr, rte, se))
    print(f"\n=== MARCHÉ×BANDE : {len(surv_mb)} survivant(s) +EV ===", flush=True)
    for k, rtr, rte, se in sorted(surv_mb, key=lambda x: -(x[2] - x[3]))[:40]:
        print(f"  {k:<26} ROItr {100*rtr:+.1f}% ROIte {100*rte:+.1f}% (+-{100*se:.1f})", flush=True)
    if not (s1 or s2 or surv_bh or surv_mb):
        print("\nRESULTAT : AUCUNE bande, heure, ou marché×bande n'est +EV train+test.", flush=True)
    print("BANDS-HOURS-DONE", flush=True)


if __name__ == "__main__":
    main()
