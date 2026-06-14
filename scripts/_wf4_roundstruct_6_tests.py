# -*- coding: utf-8 -*-
# WF4 ROUND-STRUCTURE - core independence tests
#  T1 fav-wins per round: cross-product permutation + variance ratio (MC binomial)
#  T2 draws per round: idem
#  T3 home-wins per round: idem
#  T4 total goals per round: cross-product permutation + variance ratio vs permuted
#  T5 complete-round distribution of fav wins vs Poisson-binomial MC
#  T6 cross-round lag-1 correlation of round-level surprise
# Output -> exports/wf4_roundstruct_tests.json
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

rng = np.random.default_rng(42)
B = 10000

recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))

# enrich
for r in recs:
    imp = np.array([1/r["oh"], 1/r["od"], 1/r["oa"]])
    fair = imp / imp.sum()
    r["p1"], r["px"], r["p2"] = fair.tolist()
    res = 0 if r["sa"] > r["sb"] else (1 if r["sa"] == r["sb"] else 2)
    r["res"] = res
    fav = 0 if r["oh"] <= r["oa"] else 2
    r["p_fav"] = fair[fav]
    r["x_fav"] = 1.0 if res == fav else 0.0
    r["x_draw"] = 1.0 if res == 1 else 0.0
    r["x_home"] = 1.0 if res == 0 else 0.0
    r["tot"] = r["sa"] + r["sb"]
    r["mu"] = r["lh"] + r["la"]

LEAGUES = sorted(set(r["comp"] for r in recs))

def build_rounds(recs_l):
    g = defaultdict(list)
    for r in recs_l:
        g[(r["comp"], r["est"])].append(r)
    return g

def cross_prod_test(values, expecteds, round_ids, B=B, blocks=None):
    """values: x_i, expecteds: p_i; residual centered; C = sum_r sum_{i<j} e_i e_j.
    Permutation of residuals across matches (within blocks if given). Returns C, rho, p."""
    e = np.asarray(values, float) - np.asarray(expecteds, float)
    e = e - e.mean()
    # map round ids to ints
    rmap = {}
    inv = np.array([rmap.setdefault(k, len(rmap)) for k in round_ids])
    nR = len(rmap)
    def C_fast(ev):
        s = np.bincount(inv, weights=ev, minlength=nR)
        s2 = np.bincount(inv, weights=ev*ev, minlength=nR)
        return 0.5 * float((s*s - s2).sum())
    C_obs = C_fast(e)
    npairs = 0
    cnt = np.bincount(inv, minlength=nR)
    npairs = float((cnt*(cnt-1)//2).sum())
    var_e = float(e.var())
    rho = C_obs / (npairs * var_e) if npairs > 0 and var_e > 0 else 0.0
    if blocks is None:
        perm_C = np.empty(B)
        for b in range(B):
            perm_C[b] = C_fast(rng.permutation(e))
    else:
        blk = np.asarray(blocks)
        perm_C = np.empty(B)
        idx_by_blk = [np.where(blk == u)[0] for u in np.unique(blk)]
        for b in range(B):
            ep = e.copy()
            for idx in idx_by_blk:
                ep[idx] = e[rng.permutation(idx)]
            perm_C[b] = C_fast(ep)
    p = (1 + np.sum(np.abs(perm_C - perm_C.mean()) >= abs(C_obs - perm_C.mean()))) / (B + 1)
    z = (C_obs - perm_C.mean()) / perm_C.std() if perm_C.std() > 0 else 0.0
    return dict(C=C_obs, rho=rho, z=float(z), p=float(p), npairs=npairs,
                perm_mean=float(perm_C.mean()), perm_std=float(perm_C.std()))

def var_ratio_binom(xs, ps, round_ids, B=B):
    """Variance ratio: sum_r (S_r-E_r)^2 / sum_r V_r vs MC with independent Bernoulli(p)."""
    x = np.asarray(xs, float); p = np.asarray(ps, float)
    rmap = {}
    inv = np.array([rmap.setdefault(k, len(rmap)) for k in round_ids])
    nR = len(rmap)
    S = np.bincount(inv, weights=x, minlength=nR)
    E = np.bincount(inv, weights=p, minlength=nR)
    V = np.bincount(inv, weights=p*(1-p), minlength=nR)
    VR_obs = float(((S - E)**2).sum() / V.sum())
    sims = np.empty(B)
    for b in range(B):
        xs_b = (rng.random(len(p)) < p).astype(float)
        Sb = np.bincount(inv, weights=xs_b, minlength=nR)
        sims[b] = ((Sb - E)**2).sum() / V.sum()
    p_hi = (1 + np.sum(sims >= VR_obs)) / (B + 1)
    p_lo = (1 + np.sum(sims <= VR_obs)) / (B + 1)
    return dict(VR=VR_obs, mc_mean=float(sims.mean()), mc_std=float(sims.std()),
                p_two=float(2 * min(p_hi, p_lo)), p_hi=float(p_hi), p_lo=float(p_lo), nrounds=nR)

def var_ratio_perm(vals, mus, round_ids, B=B):
    """Goals: variance of round residual-sums vs permuted (keeps marginals)."""
    e = np.asarray(vals, float) - np.asarray(mus, float)
    e = e - e.mean()
    rmap = {}
    inv = np.array([rmap.setdefault(k, len(rmap)) for k in round_ids])
    nR = len(rmap)
    def stat(ev):
        S = np.bincount(inv, weights=ev, minlength=nR)
        return float((S**2).sum())
    obs = stat(e)
    sims = np.empty(B)
    for b in range(B):
        sims[b] = stat(rng.permutation(e))
    p_hi = (1 + np.sum(sims >= obs)) / (B + 1)
    p_lo = (1 + np.sum(sims <= obs)) / (B + 1)
    return dict(stat=obs, perm_mean=float(sims.mean()), perm_std=float(sims.std()),
                ratio=obs / float(sims.mean()),
                p_two=float(2 * min(p_hi, p_lo)), nrounds=nR)

results = {"tests": []}
n_tests = 0

def run_league(tag, recs_l, blocks_map=None):
    global n_tests
    # only rounds with >=2 matches
    g = build_rounds(recs_l)
    keep = [r for k in g for r in g[k] if len(g[k]) >= 2]
    if len(keep) < 100:
        return
    rid = [(r["comp"], r["est"]) for r in keep]
    blocks = None
    if blocks_map:
        blocks = [blocks_map[(r["comp"], r["est"])] for r in keep]
    out = {"scope": tag, "n_matches": len(keep), "n_rounds": len(set(rid))}
    for name, xk, pk in [("fav_win", "x_fav", "p_fav"), ("draw", "x_draw", "px"),
                          ("home_win", "x_home", "p1")]:
        out["xprod_" + name] = cross_prod_test([r[xk] for r in keep], [r[pk] for r in keep], rid, blocks=blocks)
        out["vr_" + name] = var_ratio_binom([r[xk] for r in keep], [r[pk] for r in keep], rid)
        n_tests += 2
    out["xprod_goals"] = cross_prod_test([r["tot"] for r in keep], [r["mu"] for r in keep], rid, blocks=blocks)
    out["vr_goals"] = var_ratio_perm([r["tot"] for r in keep], [r["mu"] for r in keep], rid)
    n_tests += 2
    results["tests"].append(out)
    # console
    print("=" * 80)
    print(tag, "matches:", len(keep), "rounds:", len(set(rid)))
    for k in ["xprod_fav_win", "xprod_draw", "xprod_home_win", "xprod_goals"]:
        d = out[k]
        print(f"  {k:18s} rho={d['rho']:+.5f} z={d['z']:+.2f} p={d['p']:.4f}")
    for k in ["vr_fav_win", "vr_draw", "vr_home_win"]:
        d = out[k]
        print(f"  {k:18s} VR={d['VR']:.4f} (mc {d['mc_mean']:.3f}±{d['mc_std']:.3f}) p_two={d['p_two']:.4f}")
    d = out["vr_goals"]
    print(f"  vr_goals           ratio={d['ratio']:.4f} p_two={d['p_two']:.4f}")

# per league
for lg in LEAGUES:
    recs_l = [r for r in recs if r["comp"] == lg]
    run_league(lg, recs_l)

# pooled new leagues
new = [r for r in recs if r["comp"] != "InstantLeague-8035"]
run_league("pooled-newleagues", new)
# pooled all
run_league("pooled-9", recs)

results["n_tests_scanned"] = n_tests
with open("exports/wf4_roundstruct_tests.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=1)
print("n_tests_scanned:", n_tests)
