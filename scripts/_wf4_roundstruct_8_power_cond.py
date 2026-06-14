# -*- coding: utf-8 -*-
# WF4 ROUND-STRUCTURE part 3:
#  P) POWER CHECK: inject synthetic intra-round correlation -> does the test detect it?
#  D) adjacent-event-id pairs within round (shared RNG stream hypothesis)
#  E) concrete conditional strategy: round N has >=k favorite losses -> back favs / dogs round N+1
#  F) per-match dispersion of totals (descriptive)
# Output -> exports/wf4_roundstruct_part3.json
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from datetime import datetime

rng = np.random.default_rng(7)
B = 4000
recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))
for r in recs:
    imp = np.array([1/r["oh"], 1/r["od"], 1/r["oa"]])
    fair = imp / imp.sum()
    r["p1"], r["px"], r["p2"] = fair.tolist()
    res = 0 if r["sa"] > r["sb"] else (1 if r["sa"] == r["sb"] else 2)
    r["res"] = res
    fav = 0 if r["oh"] <= r["oa"] else 2
    r["fav"] = fav
    r["p_fav"] = fair[fav]; r["x_fav"] = 1.0 if res == fav else 0.0
    r["o_fav"] = r["oh"] if fav == 0 else r["oa"]
    r["o_dog"] = r["oa"] if fav == 0 else r["oh"]
    r["x_dog"] = 1.0 if (res != 1 and res != fav) else 0.0
    r["tot"] = r["sa"] + r["sb"]; r["mu"] = r["lh"] + r["la"]

out = {}
L35 = [r for r in recs if r["comp"] == "InstantLeague-8035"]

def xprod_p(e, inv, nR, B=B):
    def C_fast(ev):
        s = np.bincount(inv, weights=ev, minlength=nR)
        s2 = np.bincount(inv, weights=ev*ev, minlength=nR)
        return 0.5 * float((s*s - s2).sum())
    C_obs = C_fast(e)
    perm = np.empty(B)
    for b in range(B):
        perm[b] = C_fast(rng.permutation(e))
    p = (1 + np.sum(np.abs(perm - perm.mean()) >= abs(C_obs - perm.mean()))) / (B + 1)
    return p, (C_obs - perm.mean()) / perm.std()

# ---------- P) power check on the real 8035 round structure ----------
g = defaultdict(list)
for r in L35:
    g[r["est"]].append(r)
keep = [r for k in g for r in g[k] if len(g[k]) >= 2]
rmap = {}
inv = np.array([rmap.setdefault(r["est"], len(rmap)) for r in keep])
nR = len(rmap)
ps = np.array([r["p_fav"] for r in keep])
power_res = {}
for rho_inj in [0.0, 0.02, 0.05, 0.10]:
    rejections = 0
    NSIM = 40
    pvals = []
    for sim in range(NSIM):
        # gaussian copula with intra-round equicorrelation rho_inj
        zr = rng.normal(size=nR)
        zi = rng.normal(size=len(keep))
        z = np.sqrt(rho_inj) * zr[inv] + np.sqrt(1 - rho_inj) * zi
        from scipy.stats import norm
        u = norm.cdf(z)
        x = (u < ps).astype(float)
        e = x - ps; e -= e.mean()
        p, zstat = xprod_p(e, inv, nR, B=600)
        pvals.append(p)
        if p < 0.01:
            rejections += 1
    power_res[str(rho_inj)] = dict(reject_rate_p01=rejections / NSIM, median_p=float(np.median(pvals)))
    print(f"P rho_inj={rho_inj}: reject@0.01 = {rejections}/{NSIM}, median p = {np.median(pvals):.4f}")
out["power_check_8035_struct"] = power_res

# ---------- D) adjacent-event-id pairs within rounds ----------
def adjacent_test(recs_l, tag):
    g = defaultdict(list)
    for r in recs_l:
        g[(r["comp"], r["est"])].append(r)
    pairs_fav, pairs_goal = [], []
    for k, v in g.items():
        if len(v) < 2:
            continue
        v = sorted(v, key=lambda r: r["id"])
        for i in range(len(v) - 1):
            if v[i + 1]["id"] - v[i]["id"] == 1:  # strictly adjacent ids
                pairs_fav.append((v[i]["x_fav"] - v[i]["p_fav"], v[i + 1]["x_fav"] - v[i + 1]["p_fav"]))
                pairs_goal.append((v[i]["tot"] - v[i]["mu"], v[i + 1]["tot"] - v[i + 1]["mu"]))
    res = {}
    for name, pp in [("fav", pairs_fav), ("goals", pairs_goal)]:
        if len(pp) < 100:
            continue
        a = np.array([x[0] for x in pp]); b = np.array([x[1] for x in pp])
        r_obs = float(np.corrcoef(a, b)[0, 1])
        perm = np.empty(B)
        for k2 in range(B):
            perm[k2] = np.corrcoef(a, rng.permutation(b))[0, 1]
        p = (1 + np.sum(np.abs(perm) >= abs(r_obs))) / (B + 1)
        res[name] = dict(n_pairs=len(pp), corr=r_obs, p=float(p))
        print(f"D {tag} adjacent-id {name}: n={len(pp)} corr={r_obs:+.4f} p={p:.4f}")
    return res

out["adjacent_id"] = {}
out["adjacent_id"]["8035"] = adjacent_test(L35, "8035")
out["adjacent_id"]["pooled-newleagues"] = adjacent_test([r for r in recs if r["comp"] != "InstantLeague-8035"], "new")

# ---------- E) conditional next-round strategy (8035, walk-forward split) ----------
def ts(s):
    return datetime.fromisoformat(s).timestamp()
ests = sorted(g2 for g2 in set(r["est"] for r in L35))
g35 = defaultdict(list)
for r in L35:
    g35[r["est"]].append(r)
# build consecutive pairs (gap <= 10 min, both rounds >=5 obs matches)
pairs = []
for i in range(len(ests) - 1):
    if ts(ests[i + 1]) - ts(ests[i]) <= 600 and len(g35[ests[i]]) >= 5 and len(g35[ests[i + 1]]) >= 5:
        pairs.append((ests[i], ests[i + 1]))
print("E consecutive round pairs 8035:", len(pairs))

split_t = sorted(r["est"] for r in L35)[int(len(L35) * 0.7)]
strat = {}
for cond_name, cond in [("ge3favlost", lambda v: sum(1 - r["x_fav"] for r in v) >= 3),
                         ("ge5favlost", lambda v: sum(1 - r["x_fav"] for r in v) >= 5),
                         ("le1favlost", lambda v: sum(1 - r["x_fav"] for r in v) <= 1)]:
    for side in ["fav", "dog"]:
        for scope_name, flt in [("full", lambda e1: True), ("test", lambda e1: e1 > split_t)]:
            stakes = wins = 0; pnl = 0.0; odds_sum = 0.0
            for (e1, e2) in pairs:
                if not flt(e2):
                    continue
                if cond(g35[e1]):
                    for r in g35[e2]:
                        o = r["o_fav"] if side == "fav" else r["o_dog"]
                        x = r["x_fav"] if side == "fav" else r["x_dog"]
                        stakes += 1; odds_sum += o
                        if x > 0.5:
                            wins += 1; pnl += o - 1
                        else:
                            pnl -= 1
            if stakes:
                strat[f"{cond_name}_{side}_{scope_name}"] = dict(
                    n=stakes, wr=wins / stakes, roi=pnl / stakes, avg_odds=odds_sum / stakes)
                print(f"E {cond_name:10s} back-{side} [{scope_name}]: n={stakes} wr={wins/stakes:.3f} roi={pnl/stakes*100:+.2f}% avg_odds={odds_sum/stakes:.2f}")
# baseline
for side in ["fav", "dog"]:
    stakes = wins = 0; pnl = 0.0; odds_sum = 0.0
    for r in L35:
        o = r["o_fav"] if side == "fav" else r["o_dog"]
        x = r["x_fav"] if side == "fav" else r["x_dog"]
        stakes += 1; odds_sum += o
        pnl += (o - 1) if x > 0.5 else -1
        wins += 1 if x > 0.5 else 0
    strat[f"baseline_{side}"] = dict(n=stakes, wr=wins / stakes, roi=pnl / stakes, avg_odds=odds_sum / stakes)
    print(f"E baseline back-{side}: n={stakes} wr={wins/stakes:.3f} roi={pnl/stakes*100:+.2f}% avg_odds={odds_sum/stakes:.2f}")
out["conditional_strategy_8035"] = strat

# ---------- F) descriptive dispersion ----------
tots = np.array([r["tot"] for r in L35])
mus = np.array([r["mu"] for r in L35])
out["dispersion_8035"] = dict(mean_tot=float(tots.mean()), var_tot=float(tots.var()),
                              mean_mu=float(mus.mean()),
                              var_resid=float((tots - mus).var()))
print("F 8035 totals: mean", tots.mean(), "var", tots.var(), "var/mean", tots.var()/tots.mean())

with open("exports/wf4_roundstruct_part3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("done")
