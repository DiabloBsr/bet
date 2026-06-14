# -*- coding: utf-8 -*-
"""WF4 cups - script 5:
A. Grid-identity check: per match, invert (lh,la) from 1X2 -> Poisson total P(k);
   compare with 'Total de buts' market implied fair P(k). Control = 8035 (identity proven).
B. Mi-tps 1X2 ROI (1/X/2) per league.
C. HT/FT X/X ROI per league.
D. Back away by odds bucket in 8056.
E. FTTS favori (E1 replication): FTTS side of fav if fav 1X2 odds <= 1.5, per cup.
READ-ONLY.
"""
import sys, json, math, collections
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm
from scipy.optimize import least_squares
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
LEAGUES = ["InstantLeague-8065", "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8035"]
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json", encoding="utf-8"))["events"].keys())

KMAX = 16
FACT = [math.factorial(k) for k in range(KMAX + 1)]

def pois_vec(lam):
    return np.array([math.exp(-lam) * lam**k / FACT[k] for k in range(KMAX + 1)])

def grid_pd(lh, la):
    ph = pois_vec(lh); pa = pois_vec(la)
    g = np.outer(ph, pa)
    return np.tril(g, -1).sum(), np.trace(g)

def invert_lambda(p_home, p_draw):
    def f(x):
        gh, gd = grid_pd(x[0], x[1])
        return [gh - p_home, gd - p_draw]
    r = least_squares(f, x0=[1.2, 0.9], bounds=([0.01, 0.01], [8, 8]), xtol=1e-12)
    if max(abs(v) for v in r.fun) > 1e-5:
        return None
    return float(r.x[0]), float(r.x[1])

ROWS = {}
with e.connect() as c:
    for comp in LEAGUES:
        rows = c.execute(text("""
            SELECT e.id, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
                   o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, r.goals_json
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE e.competition = :comp
              AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
        """), {"comp": comp}).fetchall()
        recs = []
        for (eid, sa, sb, hta, htb, oh, od, oa, em, gj) in rows:
            if eid in corrupted or oh is None or oh <= 1 or od <= 1 or oa <= 1:
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
            recs.append(dict(eid=eid, sa=sa, sb=sb, hta=hta, htb=htb,
                             oh=oh, od=od, oa=oa, em=emd, gj=gj))
        ROWS[comp] = recs

n_tests = 0
out = {}

def roi_cell(bets):
    n = len(bets)
    if n == 0:
        return None
    ret = sum(o for o, w in bets if w)
    wins = sum(1 for _, w in bets if w)
    profits = [(o - 1) if w else -1.0 for o, w in bets]
    mu = float(np.mean(profits)); sd = float(np.std(profits, ddof=1)) if n > 1 else 1.0
    z = mu / (sd / math.sqrt(n)) if sd > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return dict(n=n, wr=round(wins / n, 4), roi=round((ret - n) / n, 4),
                avg_odds=round(float(np.mean([o for o, _ in bets])), 3),
                z=round(z, 2), p=round(float(p), 5))

# ---------- A. grid identity: 1X2-Poisson total vs totals-market implied ----------
out["grid_identity"] = {}
for comp in LEAGUES:
    recs = ROWS[comp]
    sub = recs if len(recs) <= 2500 else recs[::max(1, len(recs) // 2000)]
    gaps = []  # per-match max abs gap over k=0..5 between Poisson(mu) pk and market fair pk
    gap_by_k = collections.defaultdict(list)
    used = 0
    for r in sub:
        mk = r["em"].get("Total de buts")
        if not mk:
            continue
        s = 1/r["oh"] + 1/r["od"] + 1/r["oa"]
        inv = invert_lambda((1/r["oh"])/s, (1/r["od"])/s)
        if inv is None:
            continue
        mu_t = inv[0] + inv[1]
        pv = pois_vec(mu_t)
        # market fair probs: normalize the 7 selections (0..6, 6=6+)
        try:
            inv_odds = {k: 1/float(v) for k, v in mk.items() if float(v) < 100}
        except Exception:
            continue
        if len(inv_odds) < 5:
            continue
        ssum = sum(1/float(v) for v in mk.values())  # includes capped, margin norm
        ok = True
        for k in ["0", "1", "2", "3", "4", "5"]:
            if k not in mk or float(mk[k]) >= 100:
                continue
            fair = (1/float(mk[k])) / ssum
            theo = pv[int(k)]
            gap_by_k[k].append(fair - theo)
        used += 1
    res = {"n_used": used}
    for k in ["0", "1", "2", "3", "4", "5"]:
        v = gap_by_k[k]
        if v:
            res[f"gap_k{k}_mean"] = round(float(np.mean(v)), 4)
            res[f"gap_k{k}_mae"] = round(float(np.mean(np.abs(v))), 4)
    out["grid_identity"][comp] = res

# ---------- B. Mi-tps 1X2 ----------
out["ht_1x2"] = {}
for comp in LEAGUES:
    res = {}
    for sel in ["1", "X", "2"]:
        bets = []
        for r in ROWS[comp]:
            if r["hta"] is None:
                continue
            mk = r["em"].get("Mi-tps 1X2")
            if not mk or sel not in mk:
                continue
            o = float(mk[sel])
            if o >= 100:
                continue
            if sel == "1":
                w = r["hta"] > r["htb"]
            elif sel == "2":
                w = r["htb"] > r["hta"]
            else:
                w = r["hta"] == r["htb"]
            bets.append((o, w))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[sel] = cell
    out["ht_1x2"][comp] = res

# ---------- C. HT/FT X/X ----------
out["htft_xx"] = {}
for comp in LEAGUES:
    bets = []
    for r in ROWS[comp]:
        if r["hta"] is None:
            continue
        mk = r["em"].get("HT/FT")
        if not mk or "X/X" not in mk:
            continue
        o = float(mk["X/X"])
        if o >= 100:
            continue
        w = (r["hta"] == r["htb"]) and (r["sa"] == r["sb"])
        bets.append((o, w))
    cell = roi_cell(bets)
    n_tests += 1
    out["htft_xx"][comp] = cell

# ---------- D. back away in 8056 by odds bucket ----------
out["away_8056"] = {}
for lo, hi in [(1.3, 1.8), (1.8, 2.5), (2.5, 4.0), (4.0, 8.0), (8.0, 30.0)]:
    bets = []
    for r in ROWS["InstantLeague-8056"]:
        if lo <= r["oa"] < hi:
            bets.append((r["oa"], r["sb"] > r["sa"]))
    cell = roi_cell(bets)
    n_tests += 1
    if cell:
        out["away_8056"][f"[{lo}-{hi})"] = cell

# ---------- E. FTTS favori (E1) ----------
out["ftts_fav"] = {}
for comp in LEAGUES:
    res = {}
    for cap in [1.5, 1.8]:
        bets = []
        for r in ROWS[comp]:
            fav_o = min(r["oh"], r["oa"])
            if fav_o > cap:
                continue
            mk = r["em"].get("FTTS")
            if not mk:
                continue
            sel = "1" if r["oh"] <= r["oa"] else "2"
            if sel not in mk:
                continue
            o = float(mk[sel])
            if o >= 100:
                continue
            # settle from goals_json: first goal team
            if not r["gj"]:
                continue
            try:
                g = json.loads(r["gj"])
            except Exception:
                continue
            if not g:
                if r["sa"] + r["sb"] == 0:
                    w = False  # no goal -> FTTS 1/2 loses
                else:
                    continue
            else:
                first = sorted(g, key=lambda x: (x.get("minute", 0)))[0]
                ft = first.get("team")
                w = (ft == "Home") if sel == "1" else (ft == "Away")
            bets.append((o, w))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[f"fav<={cap}"] = cell
    out["ftts_fav"][comp] = res

out["n_tests_scanned_script5"] = n_tests
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("exports/wf4_cups_deep2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
