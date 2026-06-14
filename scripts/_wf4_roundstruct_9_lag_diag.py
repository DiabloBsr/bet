# -*- coding: utf-8 -*-
# WF4 ROUND-STRUCTURE part 4: diagnose the 8036 lag-1 anomaly (drift artifact vs real coupling)
# lags 1..6 per league + split-half on 8036 + goals channel
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from datetime import datetime

rng = np.random.default_rng(99)
B = 5000
recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))
for r in recs:
    imp = np.array([1/r["oh"], 1/r["od"], 1/r["oa"]])
    fair = imp / imp.sum()
    res = 0 if r["sa"] > r["sb"] else (1 if r["sa"] == r["sb"] else 2)
    fav = 0 if r["oh"] <= r["oa"] else 2
    r["p_fav"] = fair[fav]; r["x_fav"] = 1.0 if res == fav else 0.0
    r["tot"] = r["sa"] + r["sb"]; r["mu"] = r["lh"] + r["la"]

def ts(s): return datetime.fromisoformat(s).timestamp()

out = {}
for lgname in ["InstantLeague-8036", "InstantLeague-8042", "InstantLeague-8043",
               "InstantLeague-8037", "InstantLeague-8044", "InstantLeague-8060"]:
    rl = [r for r in recs if r["comp"] == lgname]
    g = defaultdict(list)
    for r in rl: g[r["est"]].append(r)
    ests = sorted(g.keys())
    surpr = {e: float(np.mean([r["x_fav"] - r["p_fav"] for r in g[e]])) for e in ests}
    gsurpr = {e: float(np.mean([r["tot"] - r["mu"] for r in g[e]])) for e in ests}
    d = {}
    for lag in range(1, 7):
        pf, pg = [], []
        for i in range(len(ests) - lag):
            if ts(ests[i + lag]) - ts(ests[i]) <= 600 * lag:
                pf.append((surpr[ests[i]], surpr[ests[i + lag]]))
                pg.append((gsurpr[ests[i]], gsurpr[ests[i + lag]]))
        if len(pf) < 50: continue
        a = np.array([x[0] for x in pf]); b = np.array([x[1] for x in pf])
        cf = float(np.corrcoef(a, b)[0, 1])
        ag = np.array([x[0] for x in pg]); bg = np.array([x[1] for x in pg])
        cg = float(np.corrcoef(ag, bg)[0, 1])
        d[lag] = dict(n=len(pf), corr_fav=cf, corr_goals=cg)
    out[lgname] = d
    print(lgname, " ".join(f"L{k}:fav{v['corr_fav']:+.3f}/g{v['corr_goals']:+.3f}(n={v['n']})" for k, v in d.items()))

# split-half 8036 lag-1
rl = [r for r in recs if r["comp"] == "InstantLeague-8036"]
g = defaultdict(list)
for r in rl: g[r["est"]].append(r)
ests = sorted(g.keys())
half = len(ests) // 2
for name, sub in [("H1", ests[:half]), ("H2", ests[half:])]:
    surpr = {e: float(np.mean([r["x_fav"] - r["p_fav"] for r in g[e]])) for e in sub}
    pairs = [(surpr[sub[i]], surpr[sub[i+1]]) for i in range(len(sub)-1)
             if ts(sub[i+1]) - ts(sub[i]) <= 600]
    a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
    c = float(np.corrcoef(a, b)[0, 1])
    perm = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(B)])
    p = (1 + np.sum(np.abs(perm) >= abs(c))) / (B + 1)
    print(f"8036 lag1 {name}: n={len(pairs)} corr={c:+.4f} p={p:.4f}")
    out[f"8036_split_{name}"] = dict(n=len(pairs), corr=c, p=float(p))

with open("exports/wf4_roundstruct_part4.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("done")
