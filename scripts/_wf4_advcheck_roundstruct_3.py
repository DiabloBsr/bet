# -*- coding: utf-8 -*-
# ADVERSARIAL CHECK 3 (finding: lag-1 autocorrelation of round surprises, watchlist/rejected):
#  A) independent re-computation of per-league lag-1 + Fisher combination from the pkl
#  B) artifact hunt on 8036: est-gap distribution, team overlap and round_info between paired rounds
#  C) fresh DB pull -> updated lag-1 per league (full) + pure OOS segments after pkl max est
#     + updated 9-league Fisher combination
# READ-ONLY DB. Output -> exports/wf4_advcheck_roundstruct3.json
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from datetime import datetime

rng = np.random.default_rng(7)
B = 10000

def ts(s): return datetime.fromisoformat(s).timestamp()

def prep(recs):
    for r in recs:
        imp = np.array([1/r["oh"], 1/r["od"], 1/r["oa"]])
        fair = imp / imp.sum()
        res = 0 if r["sa"] > r["sb"] else (1 if r["sa"] == r["sb"] else 2)
        fav = 0 if r["oh"] <= r["oa"] else 2
        r["p_fav"] = fair[fav]
        r["x_fav"] = 1.0 if res == fav else 0.0
    return recs

def lag1(recs_l, max_gap=600, min_matches=5, est_min=None, min_pairs=10):
    g = defaultdict(list)
    for r in recs_l:
        g[r["est"]].append(r)
    ests = sorted(g.keys())
    surpr = {e: float(np.mean([r["x_fav"] - r["p_fav"] for r in g[e]])) for e in ests}
    nm = {e: len(g[e]) for e in ests}
    pairs, meta = [], []
    for i in range(len(ests) - 1):
        e1, e2 = ests[i], ests[i + 1]
        if ts(e2) - ts(e1) <= max_gap and nm[e1] >= min_matches and nm[e2] >= min_matches:
            if est_min is not None and e1 <= est_min:
                continue
            pairs.append((surpr[e1], surpr[e2]))
            t1 = set((r["ta"], r["tb"]) for r in g[e1])
            t2 = set((r["ta"], r["tb"]) for r in g[e2])
            r1 = set(r.get("rnd") for r in g[e1]); r2 = set(r.get("rnd") for r in g[e2])
            meta.append(dict(gap=ts(e2) - ts(e1), overlap=len(t1 & t2),
                             same_rnd=(r1 == r2 and len(r1) == 1)))
    if len(pairs) < min_pairs:
        return None, meta
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    c = float(np.corrcoef(a, b)[0, 1])
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(B)])
    p = float((1 + np.sum(np.abs(perm) >= abs(c))) / (B + 1))
    return dict(n_pairs=len(pairs), corr=c, p=p), meta

def fisher_combine(d):
    zs = np.array([math.atanh(v["corr"]) for v in d.values()])
    ws = np.array([v["n_pairs"] - 3 for v in d.values()], float)
    zbar = float((ws * zs).sum() / ws.sum())
    se = 1 / math.sqrt(ws.sum())
    from scipy.stats import norm, chi2
    z = zbar / se
    p = float(2 * (1 - norm.cdf(abs(z))))
    Q = float((ws * (zs - zbar) ** 2).sum())
    p_het = float(1 - chi2.cdf(Q, len(zs) - 1))
    return dict(r=math.tanh(zbar), z=z, p=p, Q=Q, p_het=p_het)

out = {}

# ---------- A) reproduce from pkl ----------
recs = prep(pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb")))
leagues = sorted(set(r["comp"] for r in recs))
repro = {}
for lg in leagues:
    t, meta = lag1([r for r in recs if r["comp"] == lg], min_pairs=60)
    if t:
        repro[lg] = t
        print(f"A {lg}: n={t['n_pairs']} corr={t['corr']:+.4f} p={t['p']:.4f}")
comb = fisher_combine(repro)
print(f"A combined: r={comb['r']:+.4f} z={comb['z']:.2f} p={comb['p']:.4f} Q={comb['Q']:.1f} p_het={comb['p_het']:.4f}")
out["repro_pkl"] = dict(per_league=repro, combined=comb,
                        n_total=sum(v["n_pairs"] for v in repro.values()))

# ---------- B) artifact hunt 8036 ----------
t36, meta36 = lag1([r for r in recs if r["comp"] == "InstantLeague-8036"])
gaps = [m["gap"] for m in meta36]
out["artifact_8036"] = dict(
    n_pairs=len(meta36),
    gap_min=min(gaps), gap_med=float(np.median(gaps)), gap_max=max(gaps),
    pairs_with_team_overlap=sum(1 for m in meta36 if m["overlap"] > 0),
    pairs_same_round_info=sum(1 for m in meta36 if m["same_rnd"]),
)
print("B 8036 pairs:", out["artifact_8036"])
g = defaultdict(set)
for r in recs:
    if r["comp"] == "InstantLeague-8036":
        g[r["est"]].add(r.get("rnd"))
multi = {e: sorted(map(str, v)) for e, v in g.items() if len(v) > 1}
out["artifact_8036"]["ests_with_mixed_round_info"] = len(multi)
print("B 8036 est-groups mixing round_info:", len(multi), list(multi.items())[:3])

# ---------- C) fresh DB pull ----------
from scraper.config import load_settings
from sqlalchemy import create_engine, text
e = create_engine(load_settings().db_url)
with open("exports/corrupted_events.json", encoding="utf-8") as f:
    CORRUPT = set(int(k) for k in json.load(f)["events"].keys())
SQL = """
SELECT ev.id, ev.competition, ev.expected_start, ev.round_info, ev.team_a, ev.team_b,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots o ON o.event_id = ev.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
  AND ev.competition IN ({comps})
"""
comps = ",".join("'" + c + "'" for c in leagues)
with e.connect() as conn:
    rows = conn.execute(text(SQL.format(comps=comps))).fetchall()
fresh, seen = [], set()
for (eid, comp, est, rnd, ta, tb, sa, sb, hta, htb, gj, oh, od, oa) in rows:
    if eid in CORRUPT: continue
    if oh is None or od is None or oa is None or oh <= 1 or od <= 1 or oa <= 1: continue
    if hta is not None and htb is not None and (hta > sa or htb > sb): continue
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb: continue
        except Exception:
            pass
    key = (comp, est, ta, tb)
    if key in seen: continue
    seen.add(key)
    fresh.append(dict(id=eid, comp=comp, est=est, rnd=rnd, ta=ta, tb=tb,
                      sa=sa, sb=sb, oh=float(oh), od=float(od), oa=float(oa)))
prep(fresh)
print(f"C fresh rows kept: {len(fresh)} (pkl had {len(recs)})")

upd = {}
for lg in leagues:
    t, _ = lag1([r for r in fresh if r["comp"] == lg], min_pairs=60)
    if t:
        upd[lg] = t
comb_u = fisher_combine(upd)
print("C updated per-league:", {k.split('-')[1]: (v['n_pairs'], round(v['corr'], 3), round(v['p'], 4)) for k, v in upd.items()})
print(f"C updated combined: r={comb_u['r']:+.4f} z={comb_u['z']:.2f} p={comb_u['p']:.4f} Q={comb_u['Q']:.1f} p_het={comb_u['p_het']:.4f}")
out["updated_full"] = dict(per_league=upd, combined=comb_u,
                           n_total=sum(v["n_pairs"] for v in upd.values()))

# pure OOS per league (rounds strictly after pkl max est for that league)
oos_all = {}
for lg in leagues:
    cut = max((r["est"] for r in recs if r["comp"] == lg), default=None)
    t, _ = lag1([r for r in fresh if r["comp"] == lg], est_min=cut, min_pairs=10)
    if t:
        oos_all[lg] = t
        print(f"C OOS {lg}: n={t['n_pairs']} corr={t['corr']:+.4f} p={t['p']:.4f}")
out["oos_per_league"] = oos_all
if len(oos_all) >= 2:
    comb_oos = fisher_combine(oos_all)
    out["oos_combined"] = comb_oos
    print(f"C OOS combined: r={comb_oos['r']:+.4f} p={comb_oos['p']:.4f}")

with open("exports/wf4_advcheck_roundstruct3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("done -> exports/wf4_advcheck_roundstruct3.json")
