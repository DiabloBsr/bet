# -*- coding: utf-8 -*-
"""WF4 cups - script 2: margins + calibration of cups vs championships.
- 1X2 margin per league (opening snapshot)
- extra-market margins per league (sample)
- lambda inversion (Poisson grid) -> mu priced vs actual goals
- 1X2 calibration by implied-prob bucket (favorites, draws)
READ-ONLY.
"""
import sys, json, math, collections
sys.path.insert(0, ".")
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)

LEAGUES = ["InstantLeague-8065", "InstantLeague-8056", "InstantLeague-8060",
           "InstantLeague-8035",  # baseline championship
           "InstantLeague-8036", "InstantLeague-8043"]  # 2 more championships for contrast

corrupted = set()
try:
    d = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
    corrupted = set(int(k) for k in d["events"].keys())
except Exception as ex:
    print("WARN corrupted load:", ex)

# ---------- Poisson grid helpers ----------
KMAX = 16
FACT = [math.factorial(k) for k in range(KMAX + 1)]

def pois_vec(lam):
    v = np.array([math.exp(-lam) * lam**k / FACT[k] for k in range(KMAX + 1)])
    return v

def grid_probs(lh, la):
    ph = pois_vec(lh); pa = pois_vec(la)
    g = np.outer(ph, pa)
    p_home = np.tril(g, -1).sum()
    p_draw = np.trace(g)
    p_away = np.triu(g, 1).sum()
    return p_home, p_draw, p_away

def invert_lambda(p_home, p_draw):
    """solve (lh, la) such that grid matches (p_home, p_draw). Newton-ish via scipy."""
    from scipy.optimize import least_squares
    def f(x):
        lh, la = x
        gh, gd, _ = grid_probs(lh, la)
        return [gh - p_home, gd - p_draw]
    try:
        r = least_squares(f, x0=[1.5, 1.2], bounds=([0.01, 0.01], [8, 8]), xtol=1e-12)
        if max(abs(v) for v in r.fun) > 1e-5:
            return None
        return float(r.x[0]), float(r.x[1])
    except Exception:
        return None

# ---------- pull data ----------
DATA = {}
with e.connect() as c:
    for comp in LEAGUES:
        rows = c.execute(text("""
            SELECT e.id, e.round_info, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
                   o.odds_home, o.odds_draw, o.odds_away, r.goals_json
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE e.competition = :comp
              AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
        """), {"comp": comp}).fetchall()
        recs = []
        n_guard = 0
        for (eid, ri, sa, sb, hta, htb, oh, od, oa, gj) in rows:
            if eid in corrupted:
                continue
            if oh is None or od is None or oa is None or oh <= 1 or od <= 1 or oa <= 1:
                continue
            # settlement guard for unaudited leagues
            if hta is not None and htb is not None and (hta > sa or htb > sb):
                n_guard += 1
                continue
            if gj:
                try:
                    g = json.loads(gj)
                    if g is not None and len(g) != sa + sb:
                        n_guard += 1
                        continue
                except Exception:
                    pass
            recs.append(dict(eid=eid, rnd=int(ri) if ri and ri.isdigit() else -1,
                             sa=sa, sb=sb, oh=oh, od=od, oa=oa))
        DATA[comp] = recs
        print(comp, "n_clean:", len(recs), "guarded_out:", n_guard)

out = {"leagues": {}}

for comp, recs in DATA.items():
    n = len(recs)
    if n == 0:
        continue
    margins = [1/r["oh"] + 1/r["od"] + 1/r["oa"] - 1 for r in recs]
    margins = np.array(margins)
    # lambda inversion on a subsample for speed (or all if small)
    sub = recs if n <= 4000 else recs[::max(1, n // 3000)]
    mus, lhs, las, fails = [], [], [], 0
    act_goals_sub = []
    for r in sub:
        s = 1/r["oh"] + 1/r["od"] + 1/r["oa"]
        ph, pd_ = (1/r["oh"]) / s, (1/r["od"]) / s
        inv = invert_lambda(ph, pd_)
        if inv is None:
            fails += 1
            continue
        lh, la = inv
        lhs.append(lh); las.append(la); mus.append(lh + la)
        act_goals_sub.append(r["sa"] + r["sb"])
    act_goals = [r["sa"] + r["sb"] for r in recs]
    draws = sum(1 for r in recs if r["sa"] == r["sb"])
    # implied draw prob (normalized) vs actual
    p_draw_imp = np.mean([(1/r["od"]) / (1/r["oh"] + 1/r["od"] + 1/r["oa"]) for r in recs])
    out["leagues"][comp] = {
        "n": n,
        "margin_1x2_mean": round(float(margins.mean()), 4),
        "margin_1x2_std": round(float(margins.std()), 4),
        "mu_priced_mean": round(float(np.mean(mus)), 3) if mus else None,
        "lh_mean": round(float(np.mean(lhs)), 3) if lhs else None,
        "la_mean": round(float(np.mean(las)), 3) if las else None,
        "goals_actual_mean_sub": round(float(np.mean(act_goals_sub)), 3) if act_goals_sub else None,
        "goals_actual_mean_all": round(float(np.mean(act_goals)), 3),
        "goal_bias_sub": round(float(np.mean(act_goals_sub) - np.mean(mus)), 3) if mus else None,
        "invert_fails": fails,
        "n_sub": len(sub),
        "draw_rate_actual": round(draws / n, 4),
        "draw_prob_implied_norm": round(float(p_draw_imp), 4),
    }

# ---------- extra market margins (sample of opening snapshots) ----------
with e.connect() as c:
    for comp in LEAGUES:
        rows = c.execute(text("""
            SELECT o.extra_markets FROM odds_snapshots o
            JOIN events e ON e.id = o.event_id
            WHERE e.competition = :comp AND o.extra_markets IS NOT NULL
              AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
            LIMIT 400
        """), {"comp": comp}).fetchall()
        agg = collections.defaultdict(list)
        for (em,) in rows:
            try:
                d = json.loads(em)
            except Exception:
                continue
            if not isinstance(d, dict):
                continue
            for mkt, sels in d.items():
                if not isinstance(sels, dict) or not sels:
                    continue
                try:
                    s = sum(1/float(v) for v in sels.values())
                except Exception:
                    continue
                agg[mkt].append(s - 1)
        if comp in out["leagues"]:
            out["leagues"][comp]["extra_margins"] = {
                m: round(float(np.mean(v)), 4) for m, v in sorted(agg.items()) if len(v) >= 30
            }
            out["leagues"][comp]["n_em_snapshots"] = len(rows)

print(json.dumps(out, indent=1, ensure_ascii=False))
with open("exports/wf4_cups_margins.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)
