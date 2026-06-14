# -*- coding: utf-8 -*-
"""WF4 cups - script 3: settlement-based calibration + ROI scan on cups vs 8035.
Markets settled from opening snapshot extra_markets:
- "Total de buts" (exact totals 0-6)
- "+/-" (< 3.5 / > 3.5)
- "G/NG" (BTTS)
- 1X2 by favorite-odds bucket (incl E2 replication [1.10-1.20])
- home/away symmetry check in 8065
- favorite reliability by round third (early/mid/late)
READ-ONLY. Counts every ROI cell scanned.
"""
import sys, json, math, collections
sys.path.insert(0, ".")
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)

LEAGUES = ["InstantLeague-8065", "InstantLeague-8056", "InstantLeague-8060",
           "InstantLeague-8035"]

corrupted = set()
d = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
corrupted = set(int(k) for k in d["events"].keys())

ROWS = {}
with e.connect() as c:
    for comp in LEAGUES:
        rows = c.execute(text("""
            SELECT e.id, e.round_info, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
                   o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, r.goals_json
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE e.competition = :comp
              AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
        """), {"comp": comp}).fetchall()
        recs = []
        for (eid, ri, sa, sb, hta, htb, oh, od, oa, em, gj) in rows:
            if eid in corrupted:
                continue
            if oh is None or oh <= 1 or od <= 1 or oa <= 1:
                continue
            if hta is not None and htb is not None and (hta > sa or htb > sb):
                continue
            ok = True
            if gj:
                try:
                    g = json.loads(gj)
                    if g is not None and len(g) != sa + sb:
                        ok = False
                except Exception:
                    pass
            if not ok:
                continue
            try:
                emd = json.loads(em) if em else {}
            except Exception:
                emd = {}
            recs.append(dict(eid=eid, rnd=int(ri) if ri and ri.isdigit() else -1,
                             sa=sa, sb=sb, oh=oh, od=od, oa=oa, em=emd))
        ROWS[comp] = recs

n_tests = 0
out = {"n_per_league": {k: len(v) for k, v in ROWS.items()}, "scans": {}}

def roi_cell(bets):
    """bets = list of (odds, won_bool). returns dict"""
    n = len(bets)
    if n == 0:
        return None
    stake = n
    ret = sum(o for o, w in bets if w)
    wins = sum(1 for _, w in bets if w)
    roi = (ret - stake) / stake
    avg_odds = float(np.mean([o for o, _ in bets]))
    # binomial p-value vs breakeven win prob 1/avg_odds_of_winners... use exact per-bet EV test:
    # simple: p-value via normal approx on profit with per-bet variance
    profits = [(o - 1) if w else -1.0 for o, w in bets]
    mu = float(np.mean(profits)); sd = float(np.std(profits, ddof=1)) if n > 1 else 1.0
    z = mu / (sd / math.sqrt(n)) if sd > 0 else 0.0
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return dict(n=n, wr=round(wins / n, 4), roi=round(roi, 4), avg_odds=round(avg_odds, 3),
                z=round(z, 2), p=round(float(p), 5))

# ---------- 1. "Total de buts" exact totals ----------
for comp in LEAGUES:
    res = {}
    for tot_sel in ["0", "1", "2", "3", "4", "5", "6"]:
        bets = []
        for r in ROWS[comp]:
            mk = r["em"].get("Total de buts")
            if not mk or tot_sel not in mk:
                continue
            o = float(mk[tot_sel])
            if o >= 100:
                continue
            t = r["sa"] + r["sb"]
            won = (t == int(tot_sel)) if tot_sel != "6" else (t >= 6)
            bets.append((o, won))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[tot_sel] = cell
    out["scans"].setdefault("total_buts", {})[comp] = res

# ---------- 2. +/- 3.5 ----------
for comp in LEAGUES:
    res = {}
    for sel in ["< 3.5", "> 3.5"]:
        bets = []
        for r in ROWS[comp]:
            mk = r["em"].get("+/-")
            if not mk or sel not in mk:
                continue
            o = float(mk[sel])
            if o >= 100:
                continue
            t = r["sa"] + r["sb"]
            won = (t < 3.5) if sel == "< 3.5" else (t > 3.5)
            bets.append((o, won))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[sel] = cell
    out["scans"].setdefault("ou35", {})[comp] = res

# ---------- 3. G/NG ----------
for comp in LEAGUES:
    res = {}
    for sel in ["Oui", "Non"]:
        bets = []
        for r in ROWS[comp]:
            mk = r["em"].get("G/NG")
            if not mk or sel not in mk:
                continue
            o = float(mk[sel])
            if o >= 100:
                continue
            btts = r["sa"] > 0 and r["sb"] > 0
            won = btts if sel == "Oui" else (not btts)
            bets.append((o, won))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[sel] = cell
    out["scans"].setdefault("btts", {})[comp] = res

# ---------- 4. 1X2 favorite buckets ----------
BUCKETS = [(1.0, 1.10), (1.10, 1.20), (1.20, 1.35), (1.35, 1.50), (1.50, 1.70),
           (1.70, 2.00), (2.00, 2.50)]
for comp in LEAGUES:
    res = {}
    for lo, hi in BUCKETS:
        bets = []
        for r in ROWS[comp]:
            fav_o = min(r["oh"], r["oa"])
            if not (lo <= fav_o < hi):
                continue
            fav_home = r["oh"] <= r["oa"]
            won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
            bets.append((fav_o, won))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[f"[{lo}-{hi})"] = cell
    out["scans"].setdefault("fav_1x2", {})[comp] = res

# ---------- 5. Draw backing by implied prob bucket ----------
for comp in LEAGUES:
    res = {}
    for lo, hi in [(2.5, 3.2), (3.2, 4.0), (4.0, 6.0), (6.0, 12.0)]:
        bets = []
        for r in ROWS[comp]:
            if not (lo <= r["od"] < hi):
                continue
            bets.append((r["od"], r["sa"] == r["sb"]))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[f"odds[{lo}-{hi})"] = cell
    out["scans"].setdefault("draw_back", {})[comp] = res

# ---------- 6. Home vs away symmetry in 8065 (and others) ----------
for comp in LEAGUES:
    recs = ROWS[comp]
    hg = float(np.mean([r["sa"] for r in recs]))
    ag = float(np.mean([r["sb"] for r in recs]))
    hw = sum(1 for r in recs if r["sa"] > r["sb"]) / len(recs)
    aw = sum(1 for r in recs if r["sb"] > r["sa"]) / len(recs)
    # implied home/away win prob (normalized)
    pi_h = float(np.mean([(1/r["oh"]) / (1/r["oh"] + 1/r["od"] + 1/r["oa"]) for r in recs]))
    pi_a = float(np.mean([(1/r["oa"]) / (1/r["oh"] + 1/r["od"] + 1/r["oa"]) for r in recs]))
    out["scans"].setdefault("home_away", {})[comp] = dict(
        n=len(recs), home_goals=round(hg, 3), away_goals=round(ag, 3),
        home_wr=round(hw, 4), away_wr=round(aw, 4),
        implied_home=round(pi_h, 4), implied_away=round(pi_a, 4))

# ---------- 7. Favorite reliability by round third ----------
for comp in ["InstantLeague-8065", "InstantLeague-8056", "InstantLeague-8060"]:
    maxr = {"InstantLeague-8065": 94, "InstantLeague-8056": 70, "InstantLeague-8060": 46}[comp]
    res = {}
    for name, lo, hi in [("early", 1, maxr // 3), ("mid", maxr // 3 + 1, 2 * maxr // 3),
                         ("late", 2 * maxr // 3 + 1, maxr)]:
        bets = []
        for r in ROWS[comp]:
            if not (lo <= r["rnd"] <= hi):
                continue
            fav_o = min(r["oh"], r["oa"])
            if fav_o > 2.0:
                continue
            fav_home = r["oh"] <= r["oa"]
            won = (r["sa"] > r["sb"]) if fav_home else (r["sb"] > r["sa"])
            bets.append((fav_o, won))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[f"{name}[{lo}-{hi}]"] = cell
    out["scans"].setdefault("fav_by_phase", {})[comp] = res

out["n_tests_scanned"] = n_tests
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("exports/wf4_cups_roi_scan.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
