# -*- coding: utf-8 -*-
"""WF4 cups - script 4: deep dive.
1. Score exact per-cell ROI scan (3 cups + 8035 baseline), odds<100.
2. 8060 tails: 'Total de buts 0' and '> 3.5' calibration by odds bucket + reconciliation
   (expected wins from fair prob vs actual wins).
3. Mi-tps CS 0-0 in 8060.
4. fav_by_phase full dump for 8065/8056 (check noise).
READ-ONLY.
"""
import sys, json, math, collections
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
LEAGUES = ["InstantLeague-8065", "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8035"]
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json", encoding="utf-8"))["events"].keys())

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
            recs.append(dict(eid=eid, sa=sa, sb=sb, hta=hta, htb=htb,
                             oh=oh, od=od, oa=oa, em=emd))
        ROWS[comp] = recs

n_tests = 0
out = {}

def roi_cell(bets):
    n = len(bets)
    if n == 0:
        return None
    ret = sum(o for o, w in bets if w)
    wins = sum(1 for _, w in bets if w)
    roi = (ret - n) / n
    profits = [(o - 1) if w else -1.0 for o, w in bets]
    mu = float(np.mean(profits)); sd = float(np.std(profits, ddof=1)) if n > 1 else 1.0
    z = mu / (sd / math.sqrt(n)) if sd > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return dict(n=n, wr=round(wins / n, 4), roi=round(roi, 4),
                avg_odds=round(float(np.mean([o for o, _ in bets])), 3),
                z=round(z, 2), p=round(float(p), 5))

# ---------- 1. Score exact per-cell ----------
out["score_exact_cells"] = {}
for comp in LEAGUES:
    cells = collections.defaultdict(list)
    for r in ROWS[comp]:
        mk = r["em"].get("Score exact")
        if not mk:
            continue
        actual = f'{r["sa"]}-{r["sb"]}'
        for sel, o in mk.items():
            o = float(o)
            if o >= 100:
                continue
            cells[sel].append((o, sel == actual))
    res = {}
    for sel, bets in sorted(cells.items()):
        if len(bets) < 150:
            continue
        cell = roi_cell(bets)
        n_tests += 1
        res[sel] = cell
    out["score_exact_cells"][comp] = res

# ---------- 2. 8060 tails by odds bucket ----------
out["tails_8060"] = {}
comp = "InstantLeague-8060"
for label, getmk, getsel, won_fn in [
    ("total0", "Total de buts", "0", lambda r: r["sa"] + r["sb"] == 0),
    ("over35", "+/-", "> 3.5", lambda r: r["sa"] + r["sb"] >= 4),
    ("under35", "+/-", "< 3.5", lambda r: r["sa"] + r["sb"] <= 3),
]:
    buckets = collections.defaultdict(list)
    exp_wins, act_wins, n_all = 0.0, 0, 0
    margin = 1.12 if label == "total0" else 1.06
    for r in ROWS[comp]:
        mk = r["em"].get(getmk)
        if not mk or getsel not in mk:
            continue
        o = float(mk[getsel])
        if o >= 100:
            continue
        w = won_fn(r)
        exp_wins += (1 / o) / margin
        act_wins += int(w)
        n_all += 1
        if o < 3:
            b = "<3"
        elif o < 5:
            b = "3-5"
        elif o < 8:
            b = "5-8"
        elif o < 15:
            b = "8-15"
        elif o < 30:
            b = "15-30"
        else:
            b = "30+"
        buckets[b].append((o, w))
    res = {"expected_wins_fair": round(exp_wins, 1), "actual_wins": act_wins, "n": n_all,
           "ratio_act_over_exp": round(act_wins / exp_wins, 3) if exp_wins else None}
    for b in ["<3", "3-5", "5-8", "8-15", "15-30", "30+"]:
        if b in buckets:
            cell = roi_cell(buckets[b])
            n_tests += 1
            res[f"odds {b}"] = cell
    out["tails_8060"][label] = res

# ---------- 3. Mi-tps CS 0-0 (+ 2eme mi-tps CS 0-0) in the 3 cups ----------
out["ht_cs_00"] = {}
for comp in LEAGUES:
    res = {}
    for mkt, won_fn in [("Mi-tps CS", lambda r: r["hta"] == 0 and r["htb"] == 0),
                        ("2ème mi-tps - CS", lambda r: (r["sa"] - r["hta"]) == 0 and (r["sb"] - r["htb"]) == 0)]:
        bets = []
        for r in ROWS[comp]:
            if r["hta"] is None:
                continue
            mk = r["em"].get(mkt)
            if not mk or "0-0" not in mk:
                continue
            o = float(mk["0-0"])
            if o >= 100:
                continue
            bets.append((o, won_fn(r)))
        cell = roi_cell(bets)
        n_tests += 1
        if cell:
            res[mkt + " 0-0"] = cell
    out["ht_cs_00"][comp] = res

# ---------- 4. total-goal distribution actual vs Poisson-grid implied (8060) ----------
# compare empirical total dist with the mean Poisson(mu_match) mixture using inverted lambdas?
# cheaper: use 'Total de buts' implied fair probs averaged vs actuals
out["total_dist_8060"] = {}
comp = "InstantLeague-8060"
for sel in ["0", "1", "2", "3", "4", "5", "6"]:
    imp, act, n = 0.0, 0, 0
    for r in ROWS[comp]:
        mk = r["em"].get("Total de buts")
        if not mk or sel not in mk:
            continue
        o = float(mk[sel])
        imp += (1 / o) / 1.12  # even-margin fair prob (approx, capped cells excluded below)
        t = r["sa"] + r["sb"]
        won = (t == int(sel)) if sel != "6" else (t >= 6)
        act += int(won)
        n += 1
    out["total_dist_8060"][sel] = dict(n=n, implied_fair=round(imp / n, 4), actual=round(act / n, 4),
                                       ratio=round((act / n) / (imp / n), 3))

out["n_tests_scanned_script4"] = n_tests
print(json.dumps(out, indent=1, ensure_ascii=False))
with open("exports/wf4_cups_deep.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
