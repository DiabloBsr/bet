# -*- coding: utf-8 -*-
"""WF4 cross-league part 2 — deconfounded engine-identity tests.

Uses caches written by _wf4_crossleague_1.py.
A. Goals bias: mean(total) - mean(mu) per league (the +0.12 simulator bias).
B. Score-grid deviations (2-1, 1-2, 3-3, draws) vs Poisson expectation per group.
C. O/U 3.5 calibration from "+/-" market per league.
D. 8035 time-restricted to new-league window (engine change vs league diff).
E. E2 edge (favorite 1.10-1.20) transposition.
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from scipy.stats import poisson, chi2 as chi2dist, binomtest, ttest_1samp

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
CHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
REF = "InstantLeague-8035"
NEWWIN = "2026-06-12 00:00:00"   # new leagues start 2026-06-12 01:07

with open("exports/_wf4_cl_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)
with open("exports/_wf4_cl_extra.json", "r", encoding="utf-8") as f:
    extra = json.load(f)

for e in events:
    e["total"] = e["sa"] + e["sb"]
    e["mu"] = (e["lh"] or 0) + (e["la"] or 0)

GROUPS = {l: lambda e, l=l: e["league"] == l for l in LEAGUES}
GROUPS["POOLED-CHAMP-NEW"] = lambda e: e["league"] in CHAMP
GROUPS["POOLED-CUP"] = lambda e: e["league"] in CUP
GROUPS["POOLED-ALL-NEW"] = lambda e: e["league"] != REF
GROUPS["8035-recentwin"] = lambda e: e["league"] == REF and e["ts"] >= NEWWIN
GROUPS["8035-old"] = lambda e: e["league"] == REF and e["ts"] < NEWWIN

n_tests = 0
out = {}

# ---------- A. goals bias: mean total - mean mu ----------
print("=" * 70)
print("A. GOALS BIAS (obs mean total - priced mu)")
res_a = []
for g, sel in GROUPS.items():
    evs = [e for e in events if sel(e) and e["lh"]]
    if len(evs) < 50:
        continue
    diff = np.array([e["total"] - e["mu"] for e in evs])
    t, p = ttest_1samp(diff, 0.0)
    n_tests += 1
    res_a.append({"group": g, "n": len(evs),
                  "mean_total": round(float(np.mean([e["total"] for e in evs])), 3),
                  "mean_mu": round(float(np.mean([e["mu"] for e in evs])), 3),
                  "bias": round(float(diff.mean()), 4),
                  "se": round(float(diff.std() / math.sqrt(len(diff))), 4),
                  "p": float(p)})
    r = res_a[-1]
    print(f"  {g:20s} n={r['n']:5d} total={r['mean_total']:.3f} mu={r['mean_mu']:.3f} "
          f"bias={r['bias']:+.3f} (se {r['se']:.3f}) p={r['p']:.4g}")
out["goals_bias"] = res_a

# ---------- B. score-grid deviations vs Poisson expectation ----------
print("=" * 70)
print("B. SCORE-GRID DEVIATIONS (obs/exp Poisson, no margin)")
KEY_SCORES = [(1, 0), (0, 1), (1, 1), (0, 0), (2, 1), (1, 2), (2, 2), (3, 3),
              (2, 0), (0, 2)]
res_b = {}
for g, sel in GROUPS.items():
    evs = [e for e in events if sel(e) and e["lh"]]
    if len(evs) < 300:
        continue
    n = len(evs)
    lh = np.array([e["lh"] for e in evs])
    la = np.array([e["la"] for e in evs])
    rows = []
    for (i, j) in KEY_SCORES:
        exp = float(np.sum(poisson.pmf(i, lh) * poisson.pmf(j, la)))
        obs = sum(1 for e in evs if e["sa"] == i and e["sb"] == j)
        pv = binomtest(obs, n, exp / n).pvalue if exp > 0 else 1.0
        n_tests += 1
        rows.append({"score": f"{i}-{j}", "obs": obs, "exp": round(exp, 1),
                     "ratio": round(obs / exp, 3) if exp > 0 else None,
                     "p": float(pv)})
    # draws total
    exp_draw = float(np.sum(sum(poisson.pmf(k, lh) * poisson.pmf(k, la) for k in range(11))))
    obs_draw = sum(1 for e in evs if e["sa"] == e["sb"])
    pv = binomtest(obs_draw, n, exp_draw / n).pvalue
    n_tests += 1
    rows.append({"score": "ALL-DRAWS", "obs": obs_draw, "exp": round(exp_draw, 1),
                 "ratio": round(obs_draw / exp_draw, 3), "p": float(pv)})
    res_b[g] = {"n": n, "cells": rows}
for g in ["InstantLeague-8035", "8035-recentwin", "POOLED-CHAMP-NEW", "POOLED-CUP",
          "POOLED-ALL-NEW"]:
    if g not in res_b:
        continue
    print(f"  {g} (n={res_b[g]['n']}):")
    for r in res_b[g]["cells"]:
        flag = " ***" if r["p"] < 0.01 else ("  *" if r["p"] < 0.05 else "")
        print(f"    {r['score']:9s} obs={r['obs']:5d} exp={r['exp']:7.1f} "
              f"ratio={r['ratio']:.3f} p={r['p']:.4g}{flag}")
out["score_grid"] = res_b

# ---------- C. O/U 3.5 calibration from '+/-' market ----------
print("=" * 70)
print("C. O/U 3.5 CALIBRATION ('+/-' market, devig 2-way)")
res_c = []
ou = {}
n_no_market = 0
for e in events:
    x = extra.get(str(e["id"]))
    if not x:
        n_no_market += 1
        continue
    try:
        d = json.loads(x) if isinstance(x, str) else x
        m = d.get("+/-")
        u, o = m.get("< 3.5"), m.get("> 3.5")
        if not u or not o or u <= 1 or o <= 1:
            n_no_market += 1
            continue
    except Exception:
        n_no_market += 1
        continue
    s = 1 / u + 1 / o
    e["p_over35"] = (1 / o) / s
    e["over35"] = 1 if e["total"] > 3.5 else 0
print(f"  events without usable +/- market: {n_no_market}")
BUCK = [(0.0, 0.25), (0.25, 0.35), (0.35, 0.45), (0.45, 1.01)]
for g, sel in GROUPS.items():
    evs = [e for e in events if sel(e) and "p_over35" in e]
    if len(evs) < 100:
        continue
    obs = sum(e["over35"] for e in evs)
    exp = sum(e["p_over35"] for e in evs)
    n = len(evs)
    pv = binomtest(obs, n, exp / n).pvalue
    n_tests += 1
    row = {"group": g, "n": n, "obs_rate": round(obs / n, 4),
           "exp_rate": round(exp / n, 4), "p": float(pv), "buckets": []}
    for (a, b) in BUCK:
        ev2 = [e for e in evs if a <= e["p_over35"] < b]
        if len(ev2) < 50:
            continue
        o2, x2 = sum(e["over35"] for e in ev2), sum(e["p_over35"] for e in ev2)
        pv2 = binomtest(o2, len(ev2), x2 / len(ev2)).pvalue
        n_tests += 1
        row["buckets"].append({"b": f"[{a}-{b})", "n": len(ev2),
                               "obs": round(o2 / len(ev2), 4),
                               "exp": round(x2 / len(ev2), 4), "p": float(pv2)})
    res_c.append(row)
    flag = " ***" if row["p"] < 0.01 else ""
    print(f"  {g:20s} n={n:5d} obs_over={row['obs_rate']:.4f} exp={row['exp_rate']:.4f} "
          f"p={row['p']:.4g}{flag}")
    for b in row["buckets"]:
        fl = " ***" if b["p"] < 0.01 else ""
        print(f"      {b['b']:12s} n={b['n']:5d} obs={b['obs']:.4f} exp={b['exp']:.4f} p={b['p']:.4g}{fl}")
out["ou35_calibration"] = res_c

# ---------- D. 1X2 calibration per bucket vs OWN odds (deconfounded) ----------
print("=" * 70)
print("D. 1X2 CALIBRATION PER BUCKET vs OWN ODDS")
BUCKETS = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.72),
           (0.72, 1.01)]
res_d = {}
for g, sel in GROUPS.items():
    evs = [e for e in events if sel(e)]
    if len(evs) < 300:
        continue
    rows = []
    for (a, b) in BUCKETS:
        ev2 = [e for e in evs if a <= e["ph"] < b]
        if len(ev2) < 80:
            continue
        n = len(ev2)
        obs = np.array([sum(1 for e in ev2 if e["out"] == o) for o in "HDA"], float)
        exp = np.array([sum(e["p" + o] for e in ev2) for o in "hda"])
        chi2 = float(np.sum((obs - exp) ** 2 / exp))
        p = float(chi2dist.sf(chi2, 2))
        n_tests += 1
        rows.append({"b": f"ph[{a}-{b})", "n": n,
                     "obs": [round(x / n, 4) for x in obs],
                     "exp": [round(x / n, 4) for x in exp], "p": p})
    res_d[g] = rows
for g in ["InstantLeague-8035", "8035-recentwin", "8035-old", "POOLED-ALL-NEW",
          "POOLED-CUP", "POOLED-CHAMP-NEW"]:
    print(f"  {g}:")
    for r in res_d.get(g, []):
        fl = " ***" if r["p"] < 0.01 else ("  *" if r["p"] < 0.05 else "")
        print(f"    {r['b']:14s} n={r['n']:5d} obs={r['obs']} exp={r['exp']} p={r['p']:.4g}{fl}")
out["calib_buckets_own"] = res_d

# ---------- E. E2 edge transposition: favorite odds in [1.10, 1.20] ----------
print("=" * 70)
print("E. E2 FAVORITE [1.10-1.20] TRANSPOSITION (1u at opening odds)")
res_e = []
def settle_fav(evs):
    bets = []
    for e in evs:
        if e["oh"] <= e["oa"]:
            odds, win = e["oh"], e["out"] == "H"
        else:
            odds, win = e["oa"], e["out"] == "A"
        if 1.10 <= odds <= 1.20:
            bets.append((odds, win))
    if not bets:
        return None
    n = len(bets)
    wr = sum(w for _, w in bets) / n
    roi = (sum(o * w for o, w in bets) - n) / n
    avg = sum(o for o, _ in bets) / n
    # p-value vs break-even prob 1/avg_odds... use implied prob from own odds (devig?)
    pv = binomtest(sum(w for _, w in bets), n, 1.0 / avg).pvalue
    return {"n": n, "wr": round(wr, 4), "roi": round(roi, 4),
            "avg_odds": round(avg, 4), "p_vs_breakeven": float(pv)}
# 8035 walk-forward: 70/30 temporal split
evs35 = sorted([e for e in events if e["league"] == REF], key=lambda e: e["ts"])
cut = int(len(evs35) * 0.70)
for g, evs in [("8035-walkforward-test", evs35[cut:]),
               ("8035-full", evs35),
               ("POOLED-ALL-NEW", [e for e in events if e["league"] != REF]),
               ("POOLED-CHAMP-NEW", [e for e in events if e["league"] in CHAMP]),
               ("POOLED-CUP", [e for e in events if e["league"] in CUP])]:
    r = settle_fav(evs)
    if r:
        n_tests += 1
        r["group"] = g
        res_e.append(r)
        print(f"  {g:22s} n={r['n']:5d} wr={r['wr']:.4f} roi={r['roi']:+.4f} "
              f"avg_odds={r['avg_odds']:.3f} p={r['p_vs_breakeven']:.4g}")
out["e2_favorite"] = res_e

out["n_tests_part2"] = n_tests
with open("exports/wf4_crossleague_part2.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_crossleague_part2.json; n_tests part2 =", n_tests)
