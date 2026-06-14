# -*- coding: utf-8 -*-
# WF4 ROUND-STRUCTURE part 2:
#  A) pooled tests redone with PER-LEAGUE centering + within-league permutation
#  B) complete-round distribution of fav-wins / draws vs Poisson-binomial MC
#  C) cross-round lag-1: surprises in round N -> round N+1 (actionable conditioning)
# Output -> exports/wf4_roundstruct_part2.json
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict, Counter

rng = np.random.default_rng(123)
B = 10000
recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))
for r in recs:
    imp = np.array([1/r["oh"], 1/r["od"], 1/r["oa"]])
    fair = imp / imp.sum()
    r["p1"], r["px"], r["p2"] = fair.tolist()
    res = 0 if r["sa"] > r["sb"] else (1 if r["sa"] == r["sb"] else 2)
    r["res"] = res
    fav = 0 if r["oh"] <= r["oa"] else 2
    r["p_fav"] = fair[fav]; r["x_fav"] = 1.0 if res == fav else 0.0
    r["x_draw"] = 1.0 if res == 1 else 0.0
    r["x_home"] = 1.0 if res == 0 else 0.0
    r["tot"] = r["sa"] + r["sb"]; r["mu"] = r["lh"] + r["la"]

out = {}

# ---------- A) pooled with per-league centering ----------
def xprod_league_centered(recs_all, xk, pk, B=B):
    leagues = sorted(set(r["comp"] for r in recs_all))
    g = defaultdict(list)
    for r in recs_all:
        g[(r["comp"], r["est"])].append(r)
    keep = [r for k in g for r in g[k] if len(g[k]) >= 2]
    e = np.array([r[xk] - r[pk] for r in keep])
    lg = np.array([leagues.index(r["comp"]) for r in keep])
    for li in range(len(leagues)):  # per-league centering
        m = lg == li
        e[m] -= e[m].mean()
    rmap = {}
    inv = np.array([rmap.setdefault((r["comp"], r["est"]), len(rmap)) for r in keep])
    nR = len(rmap)
    def C_fast(ev):
        s = np.bincount(inv, weights=ev, minlength=nR)
        s2 = np.bincount(inv, weights=ev*ev, minlength=nR)
        return 0.5 * float((s*s - s2).sum())
    C_obs = C_fast(e)
    cnt = np.bincount(inv, minlength=nR)
    npairs = float((cnt*(cnt-1)//2).sum())
    rho = C_obs / (npairs * e.var())
    idx_by_lg = [np.where(lg == li)[0] for li in range(len(leagues))]
    perm = np.empty(B)
    for b in range(B):
        ep = e.copy()
        for idx in idx_by_lg:
            ep[idx] = e[rng.permutation(idx)]
        perm[b] = C_fast(ep)
    p = (1 + np.sum(np.abs(perm - perm.mean()) >= abs(C_obs - perm.mean()))) / (B + 1)
    z = (C_obs - perm.mean()) / perm.std()
    return dict(rho=float(rho), z=float(z), p=float(p), n_matches=len(keep), n_rounds=nR)

new = [r for r in recs if r["comp"] != "InstantLeague-8035"]
out["pooled_lgcentered"] = {}
for scope, data in [("pooled-newleagues", new), ("pooled-9", recs)]:
    d = {}
    for name, xk, pk in [("fav_win", "x_fav", "p_fav"), ("draw", "x_draw", "px"),
                          ("home_win", "x_home", "p1"), ("goals", "tot", "mu")]:
        d[name] = xprod_league_centered(data, xk, pk)
        print(f"A {scope:20s} {name:9s} rho={d[name]['rho']:+.5f} z={d[name]['z']:+.2f} p={d[name]['p']:.4f}")
    out["pooled_lgcentered"][scope] = d

# per-league goal bias for info
print("\nper-league goal bias (real - mu_price):")
bias = {}
for lgname in sorted(set(r["comp"] for r in recs)):
    rl = [r for r in recs if r["comp"] == lgname]
    b = float(np.mean([r["tot"] - r["mu"] for r in rl]))
    bias[lgname] = dict(bias=b, n=len(rl))
    print(f"  {lgname}: {b:+.4f} (n={len(rl)})")
out["goal_bias_per_league"] = bias

# ---------- B) complete-round distribution ----------
def dist_test(recs_l, full_size, xk, pk, B=5000):
    g = defaultdict(list)
    for r in recs_l:
        g[(r["comp"], r["est"])].append(r)
    rounds = [v for v in g.values() if len(v) == full_size]
    if len(rounds) < 80:
        return None
    obs_counts = Counter(int(sum(r[xk] for r in v)) for v in rounds)
    ps = [np.array([r[pk] for r in v]) for v in rounds]
    # MC null distribution of histogram -> chi2-like statistic
    K = full_size + 1
    obs_hist = np.array([obs_counts.get(k, 0) for k in range(K)], float)
    exp_hist = np.zeros(K)
    sims_stat = np.empty(B)
    # expected histogram via exact Poisson-binomial per round (DP)
    def pb_pmf(p):
        f = np.array([1.0])
        for pi in p:
            f = np.convolve(f, [1 - pi, pi])
        return f
    pmfs = [pb_pmf(p) for p in ps]
    for f in pmfs:
        exp_hist[:len(f)] += f
    chi_obs = float(np.sum((obs_hist - exp_hist) ** 2 / np.maximum(exp_hist, 1e-9)))
    for b in range(B):
        h = np.zeros(K)
        for p in ps:
            k = int((rng.random(len(p)) < p).sum())
            h[k] += 1
        sims_stat[b] = np.sum((h - exp_hist) ** 2 / np.maximum(exp_hist, 1e-9))
    pval = (1 + np.sum(sims_stat >= chi_obs)) / (B + 1)
    return dict(n_rounds=len(rounds), obs=obs_hist.tolist(), exp=exp_hist.tolist(),
                chi=chi_obs, mc_mean=float(sims_stat.mean()), p=float(pval))

out["dist_complete_rounds"] = {}
SIZES = {"InstantLeague-8035": 10, "InstantLeague-8036": 10, "InstantLeague-8037": 10,
         "InstantLeague-8042": 9, "InstantLeague-8043": 9, "InstantLeague-8044": 9,
         "InstantLeague-8056": 18, "InstantLeague-8060": 12, "InstantLeague-8065": 24}
for lgname, sz in SIZES.items():
    rl = [r for r in recs if r["comp"] == lgname]
    d = {}
    for name, xk, pk in [("fav_win", "x_fav", "p_fav"), ("draw", "x_draw", "px")]:
        t = dist_test(rl, sz, xk, pk)
        if t:
            d[name] = t
            print(f"B {lgname} {name}: n_rounds={t['n_rounds']} chi={t['chi']:.1f} (mc {t['mc_mean']:.1f}) p={t['p']:.4f}")
    if d:
        out["dist_complete_rounds"][lgname] = d

# ---------- C) cross-round lag-1 ----------
# round-level surprise = mean(x_fav - p_fav); consecutive rounds in same league by est order
def lag1_test(recs_l, max_gap_sec=600, B=B):
    g = defaultdict(list)
    for r in recs_l:
        g[r["est"]].append(r)
    ests = sorted(g.keys())
    from datetime import datetime
    def ts(s):
        return datetime.fromisoformat(s).timestamp()
    surpr = {est: float(np.mean([r["x_fav"] - r["p_fav"] for r in g[est]])) for est in ests}
    nmatch = {est: len(g[est]) for est in ests}
    pairs = []
    for i in range(len(ests) - 1):
        if ts(ests[i + 1]) - ts(ests[i]) <= max_gap_sec and nmatch[ests[i]] >= 5 and nmatch[ests[i + 1]] >= 5:
            pairs.append((surpr[ests[i]], surpr[ests[i + 1]]))
    if len(pairs) < 60:
        return None
    a = np.array([p[0] for p in pairs]); bb = np.array([p[1] for p in pairs])
    r_obs = float(np.corrcoef(a, bb)[0, 1])
    perm = np.empty(B)
    for k in range(B):
        perm[k] = np.corrcoef(a, rng.permutation(bb))[0, 1]
    p = (1 + np.sum(np.abs(perm) >= abs(r_obs))) / (B + 1)
    return dict(n_pairs=len(pairs), corr=r_obs, p=float(p))

out["lag1"] = {}
for lgname in SIZES:
    rl = [r for r in recs if r["comp"] == lgname]
    t = lag1_test(rl)
    if t:
        out["lag1"][lgname] = t
        print(f"C {lgname} lag1: n_pairs={t['n_pairs']} corr={t['corr']:+.4f} p={t['p']:.4f}")

with open("exports/wf4_roundstruct_part2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("done")
