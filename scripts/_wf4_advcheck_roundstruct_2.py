# -*- coding: utf-8 -*-
# ADVERSARIAL CHECK 2: independent re-implementation of the intra-round independence tests
#  A) merged rounds (cluster expected_start within 120s) -> does correlation hide across fragments?
#  B) temporal split halves on 8035 (lucky-period check)
#  C) equicorrelation estimate + bootstrap CI over rounds (independent of permutation machinery)
#  D) same-round vs cross-round 2-leg favorite parlays at offered odds (practical combo check)
#  E) conservative baseline back-fav ROI on 8035 test split (last 30%) + bootstrap CI
# Output -> exports/wf4_advcheck_roundstruct.json
import sys, json, pickle
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from datetime import datetime

rng = np.random.default_rng(2024)
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
    r["tot"] = r["sa"] + r["sb"]; r["mu"] = r["lh"] + r["la"]
    r["ts"] = datetime.fromisoformat(r["est"]).timestamp()

out = {}

def merged_round_ids(recs_l, gap=120.0):
    """Cluster matches of one league by expected_start with <=gap seconds linkage."""
    rs = sorted(recs_l, key=lambda r: r["ts"])
    rid = {}
    cur = 0; last = None
    for r in rs:
        if last is not None and r["ts"] - last > gap:
            cur += 1
        rid[r["id"]] = cur
        last = r["ts"]
    return rid

def xprod(e, inv, nR, B=B):
    def C_fast(ev):
        s = np.bincount(inv, weights=ev, minlength=nR)
        s2 = np.bincount(inv, weights=ev*ev, minlength=nR)
        return 0.5 * float((s*s - s2).sum())
    C_obs = C_fast(e)
    cnt = np.bincount(inv, minlength=nR)
    npairs = float((cnt*(cnt-1)//2).sum())
    rho = C_obs / (npairs * e.var()) if npairs and e.var() > 0 else 0.0
    perm = np.empty(B)
    for b in range(B):
        perm[b] = C_fast(rng.permutation(e))
    p = (1 + np.sum(np.abs(perm - perm.mean()) >= abs(C_obs - perm.mean()))) / (B + 1)
    return dict(rho=float(rho), p=float(p), npairs=npairs,
                z=float((C_obs - perm.mean()) / perm.std()))

def run_tests(tag, recs_l, rid_map):
    res = {"tag": tag, "n": len(recs_l)}
    groups = defaultdict(int)
    for r in recs_l:
        groups[rid_map[r["id"]]] += 1
    keep = [r for r in recs_l if groups[rid_map[r["id"]]] >= 2]
    rmap = {}
    inv = np.array([rmap.setdefault(rid_map[r["id"]], len(rmap)) for r in keep])
    nR = len(rmap)
    res["n_kept"] = len(keep); res["n_rounds"] = nR
    for name, xk, pk in [("fav", "x_fav", "p_fav"), ("draw", "x_draw_", "px"),
                          ("goals", "tot", "mu")]:
        if xk == "x_draw_":
            e = np.array([(1.0 if r["res"] == 1 else 0.0) - r[pk] for r in keep])
        else:
            e = np.array([r[xk] - r[pk] for r in keep])
        e -= e.mean()
        res[name] = xprod(e, inv, nR)
        print(f"  {tag:28s} {name:6s} rho={res[name]['rho']:+.5f} z={res[name]['z']:+.2f} p={res[name]['p']:.4f} (pairs={res[name]['npairs']:.0f})")
    return res, keep, inv, nR

L35 = [r for r in recs if r["comp"] == "InstantLeague-8035"]

# ---------- A) merged rounds ----------
print("A) merged-round grouping (120s linkage)")
out["merged"] = {}
rid35 = merged_round_ids(L35)
out["merged"]["8035"], keep35, inv35, nR35 = run_tests("8035-merged120s", L35, rid35)
# pooled-9 merged per league
ridall = {}
for lg in sorted(set(r["comp"] for r in recs)):
    rl = [r for r in recs if r["comp"] == lg]
    rm = merged_round_ids(rl)
    for r in rl:
        ridall[r["id"]] = (lg, rm[r["id"]])
out["merged"]["pooled-9"], _, _, _ = run_tests("pooled9-merged120s", recs, ridall)

# ---------- B) temporal halves 8035 (merged rounds) ----------
print("B) temporal halves 8035")
med = np.median([r["ts"] for r in L35])
out["halves"] = {}
for tag, sel in [("H1", [r for r in L35 if r["ts"] <= med]), ("H2", [r for r in L35 if r["ts"] > med])]:
    out["halves"][tag], _, _, _ = run_tests("8035-" + tag, sel, rid35)

# ---------- C) equicorrelation + bootstrap CI over rounds (8035 merged + pooled) ----------
print("C) equicorrelation bootstrap")
def equicorr_boot(keep, rid_map, nboot=2000):
    rounds = defaultdict(list)
    for r in keep:
        rounds[rid_map[r["id"]]].append(r)
    rounds = [v for v in rounds.values() if len(v) >= 2]
    def rho_of(sample):
        num = 0.0; den = 0.0
        allx = []
        for v in sample:
            es = [r["x_fav"] - r["p_fav"] for r in v]
            allx.extend(es)
        m = np.mean(allx)
        var = np.var(allx)
        for v in sample:
            es = np.array([r["x_fav"] - r["p_fav"] for r in v]) - m
            s = es.sum(); s2 = (es**2).sum()
            num += 0.5 * (s*s - s2)
            den += len(v) * (len(v) - 1) / 2
        return num / (den * var)
    obs = rho_of(rounds)
    bs = np.empty(nboot)
    n = len(rounds)
    for b in range(nboot):
        idx = rng.integers(0, n, n)
        bs[b] = rho_of([rounds[i] for i in idx])
    return dict(rho=float(obs), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                n_rounds=n)
out["equicorr_8035_merged"] = equicorr_boot(L35, rid35)
print("  8035 merged rho_fav:", out["equicorr_8035_merged"])
out["equicorr_pooled9_merged"] = equicorr_boot(recs, ridall)
print("  pooled-9 merged rho_fav:", out["equicorr_pooled9_merged"])

# ---------- D) parlays: same-round vs cross-round 2-leg favorites ----------
print("D) 2-leg favorite parlays at offered odds")
def parlay_rois():
    rounds = defaultdict(list)
    for r in recs:
        rounds[ridall[r["id"]]].append(r)
    same, cross = [], []
    keys = sorted(rounds.keys())
    prev_by_lg = {}
    for k in keys:
        v = sorted(rounds[k], key=lambda r: r["o_fav"])
        if len(v) >= 2:
            a, b = v[0], v[1]   # two strongest favorites in the round
            same.append((a["o_fav"] * b["o_fav"], a["x_fav"] * b["x_fav"]))
        lg = k[0]
        if lg in prev_by_lg and len(v) >= 1:
            pv = prev_by_lg[lg]
            a = sorted(pv, key=lambda r: r["o_fav"])[0]
            b = sorted(v, key=lambda r: r["o_fav"])[0]
            cross.append((a["o_fav"] * b["o_fav"], a["x_fav"] * b["x_fav"]))
        prev_by_lg[lg] = v
    def roi(arr):
        pnl = sum((o - 1) if x > 0.5 else -1 for o, x in arr)
        return pnl / len(arr), len(arr), float(np.mean([o for o, _ in arr])), float(np.mean([x for _, x in arr]))
    rs, ns, avs, wrs = roi(same)
    rc, nc, avc, wrc = roi(cross)
    # bootstrap diff
    sa = np.array([(o - 1) if x > 0.5 else -1.0 for o, x in same])
    ca = np.array([(o - 1) if x > 0.5 else -1.0 for o, x in cross])
    diffs = np.empty(3000)
    for b in range(3000):
        diffs[b] = rng.choice(sa, len(sa)).mean() - rng.choice(ca, len(ca)).mean()
    return dict(same=dict(n=ns, roi=rs, avg_odds=avs, wr=wrs),
                cross=dict(n=nc, roi=rc, avg_odds=avc, wr=wrc),
                diff_roi=rs - rc, diff_ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))])
out["parlays"] = parlay_rois()
print("  same-round :", out["parlays"]["same"])
print("  cross-round:", out["parlays"]["cross"])
print("  diff roi:", round(out["parlays"]["diff_roi"], 4), "ci95:", out["parlays"]["diff_ci95"])

# ---------- E) conservative baseline ROI back-fav 8035, last-30% test split ----------
print("E) baseline back-fav 8035, walk-forward test split")
sorted35 = sorted(L35, key=lambda r: r["ts"])
cut = sorted35[int(0.7 * len(sorted35))]["ts"]
test = [r for r in sorted35 if r["ts"] > cut]
pnl = np.array([(r["o_fav"] - 1) if r["x_fav"] > 0.5 else -1.0 for r in test])
bs = np.array([rng.choice(pnl, len(pnl)).mean() for _ in range(3000)])
out["baseline_fav_8035_test"] = dict(
    n=len(test), wr=float(np.mean([r["x_fav"] for r in test])),
    roi=float(pnl.mean()), avg_odds=float(np.mean([r["o_fav"] for r in test])),
    roi_ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
print(" ", out["baseline_fav_8035_test"])

with open("exports/wf4_advcheck_roundstruct.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("done")
