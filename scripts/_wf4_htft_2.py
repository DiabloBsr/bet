# -*- coding: utf-8 -*-
"""WF4 HT/FT miner - step 2: conditional scan.

Grid: 9 outcomes x (7 favorite buckets from opening 1X2 + 6 outcome-odds bands
+ 4 fav x comeback combos). Train = 8035 first 70%. Candidates (train n>=80,
ROI>=+4%) re-evaluated on 8035-test and pooled-newleagues. All tests counted.
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_pickle("exports/_wf4_htft_data.pkl")
OUTCOMES = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
def col(k): return "o_" + k.replace("/", "")

e35 = d[d["lg"] == "InstantLeague-8035"].sort_values("ts")
cut = int(len(e35) * 0.7)
tr, te = e35.iloc[:cut], e35.iloc[cut:]
new = d[d["lg"] != "InstantLeague-8035"]

FAV_BUCKETS = {
    "homefav_xtr": lambda s: s["oh"] <= 1.30,
    "homefav_str": lambda s: (s["oh"] > 1.30) & (s["oh"] <= 1.60),
    "homefav_mid": lambda s: (s["oh"] > 1.60) & (s["oh"] <= 2.00),
    "balanced":    lambda s: (s["oh"] > 2.00) & (s["oa"] > 2.00),
    "awayfav_mid": lambda s: (s["oa"] > 1.60) & (s["oa"] <= 2.00),
    "awayfav_str": lambda s: (s["oa"] > 1.30) & (s["oa"] <= 1.60),
    "awayfav_xtr": lambda s: s["oa"] <= 1.30,
    "homefav_any": lambda s: s["oh"] < s["oa"],
    "awayfav_any": lambda s: s["oa"] < s["oh"],
}
ODDS_BANDS = [(3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 100)]

def evalsel(sub, k, mask=None, lo=None, hi=None):
    c_ = col(k)
    s = sub[sub[c_] < 99.99]
    if mask is not None:
        s = s[mask(s)]
    if lo is not None:
        s = s[(s[c_] >= lo) & (s[c_] < hi)]
    n = len(s)
    if n == 0:
        return {"n": 0}
    w = (s["res"] == k).astype(int)
    profit = w * s[c_] - 1.0
    roi = float(profit.mean())
    p = float(stats.ttest_1samp(profit, 0.0).pvalue) if n > 5 and profit.std() > 0 else 1.0
    return {"n": int(n), "wins": int(w.sum()), "freq": round(float(w.mean()), 4),
            "avg_odds": round(float(s[c_].mean()), 3), "roi": round(roi, 4),
            "p": round(p, 5)}

n_tests = 0
results = []
for k in OUTCOMES:
    for bname, bfn in FAV_BUCKETS.items():
        r = evalsel(tr, k, mask=bfn); n_tests += 1
        results.append({"outcome": k, "cond": f"fav:{bname}", **r})
    for lo, hi in ODDS_BANDS:
        r = evalsel(tr, k, lo=lo, hi=hi); n_tests += 1
        results.append({"outcome": k, "cond": f"odds:[{lo},{hi})", **r})
    # fav bucket x odds>=3 (objective: high odds)
    for bname in ["homefav_any", "awayfav_any", "homefav_str", "awayfav_str"]:
        bfn = FAV_BUCKETS[bname]
        r = evalsel(tr, k, mask=lambda s, f=bfn: f(s) & (s[col(k)] >= 3)); n_tests += 1
        results.append({"outcome": k, "cond": f"fav:{bname}&odds>=3", **r})

print(f"TRAIN scan: {n_tests} tests")
cands = [r for r in results if r["n"] >= 80 and r.get("roi", -9) >= 0.04]
cands.sort(key=lambda r: r["p"])
print(f"candidates train (n>=80, ROI>=4%): {len(cands)}")
fmt = "{:<5} {:<28} {:>5} {:>5} {:>7} {:>7} {:>8} {:>8}"
print(fmt.format("out", "cond", "n", "wins", "freq", "odds", "ROI", "p"))
for r in cands:
    print(fmt.format(r["outcome"], r["cond"], r["n"], r["wins"], r["freq"],
                     r["avg_odds"], f"{r['roi']:+.4f}", r["p"]))

print("\n=== validation of candidates on 8035-TEST and pooled-newleagues ===")
val = []
for r in cands:
    k, cond = r["outcome"], r["cond"]
    if cond.startswith("fav:") and "&odds>=3" in cond:
        bn = cond[4:].replace("&odds>=3", "")
        bfn = FAV_BUCKETS[bn]
        m = lambda s, f=bfn, kk=k: f(s) & (s[col(kk)] >= 3)
        a_te = evalsel(te, k, mask=m); a_nw = evalsel(new, k, mask=m)
    elif cond.startswith("fav:"):
        bfn = FAV_BUCKETS[cond[4:]]
        a_te = evalsel(te, k, mask=bfn); a_nw = evalsel(new, k, mask=bfn)
    else:
        lo, hi = json.loads(cond[5:].replace("(", "[").replace(")", "]"))
        a_te = evalsel(te, k, lo=lo, hi=hi); a_nw = evalsel(new, k, lo=lo, hi=hi)
    val.append({"outcome": k, "cond": cond, "train": r, "test": a_te, "new": a_nw})
    print(f"{k} {cond}: TEST n={a_te['n']} roi={a_te.get('roi')} p={a_te.get('p')} | "
          f"NEW n={a_nw['n']} roi={a_nw.get('roi')} p={a_nw.get('p')}")

json.dump({"n_tests_scanned": n_tests, "train_scan": results, "validation": val},
          open("exports/wf4_htft_step2.json", "w"), indent=1)
print(f"\nn_tests_scanned (train grid) = {n_tests}")
