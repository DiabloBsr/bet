# -*- coding: utf-8 -*-
"""WF4 HT/FT miner v2 - step 4: conditional scan, walk-forward.

Grid on 8035-TRAIN (first 70% by expected_start):
  - 9 outcomes x 9 favorite buckets (opening 1X2)
  - 9 outcomes x 6 outcome-odds bands
  - 9 outcomes x (homefav/awayfav any) x odds>=3
  - 4 favorite-relative combos (comeback/late) x 4 strength conditions
Candidates (train n>=100, ROI>=+4%) -> validated on 8035-TEST + pooled-newleagues.
All train tests counted in n_tests_scanned. Appends results to exports/wf4_htft.json.
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_pickle("exports/_wf4_htft3_data.pkl")
OUTCOMES = ["1/1", "1/X", "1/2", "X/1", "X/X", "X/2", "2/1", "2/X", "2/2"]
def col(k): return "o_" + k.replace("/", "")

e35 = d[d["lg"] == "InstantLeague-8035"].sort_values("ts").reset_index(drop=True)
cut = int(len(e35) * 0.7)
tr, te = e35.iloc[:cut], e35.iloc[cut:]
new = d[d["lg"] != "InstantLeague-8035"]
print(f"8035 train={len(tr)} test={len(te)} new={len(new)}")

FAV_BUCKETS = {
    "homefav_any": lambda s: s["oh"] < s["oa"],
    "awayfav_any": lambda s: s["oa"] < s["oh"],
    "homefav_xtr": lambda s: s["oh"] <= 1.30,
    "homefav_str": lambda s: (s["oh"] > 1.30) & (s["oh"] <= 1.60),
    "homefav_mid": lambda s: (s["oh"] > 1.60) & (s["oh"] <= 2.00),
    "awayfav_xtr": lambda s: s["oa"] <= 1.30,
    "awayfav_str": lambda s: (s["oa"] > 1.30) & (s["oa"] <= 1.60),
    "awayfav_mid": lambda s: (s["oa"] > 1.60) & (s["oa"] <= 2.00),
    "balanced":    lambda s: (s["oh"] > 2.00) & (s["oa"] > 2.00),
}
ODDS_BANDS = [(3, 5), (5, 8), (8, 12), (12, 20), (20, 40), (40, 100)]

def stat_block(o, w):
    """o = odds array, w = win 0/1 array. Returns metrics + 2 p-values."""
    n = len(o)
    if n == 0: return {"n": 0}
    profit_v = w * o - 1.0
    profit = profit_v.sum()
    roi = profit / n
    p0 = 1.0 / o
    var0 = (o * o * p0 * (1 - p0)).sum()
    z = profit / math.sqrt(var0) if var0 > 0 else 0.0
    p_be = float(1 - stats.norm.cdf(z))          # break-even null, one-sided
    p_t = float(stats.ttest_1samp(profit_v, 0.0).pvalue) if n > 5 and profit_v.std() > 0 else 1.0
    return {"n": int(n), "wins": int(w.sum()), "wr": round(float(w.mean()), 4),
            "avg_odds": round(float(o.mean()), 3), "roi": round(float(roi), 4),
            "p_be": round(p_be, 6), "p_t": round(p_t, 6)}

def eval_single(sub, k, mask=None, lo=None, hi=None):
    c_ = col(k)
    s = sub[sub[c_] < 99.99]
    if mask is not None: s = s[mask(s)]
    if lo is not None: s = s[(s[c_] >= lo) & (s[c_] < hi)]
    return stat_block(s[c_].values, (s["res"] == k).astype(float).values)

def eval_combo(sub, pick_fn):
    """pick_fn(row) -> outcome key or None. 1u on that outcome."""
    os_, ws_ = [], []
    for t in sub.itertuples():
        k = pick_fn(t)
        if k is None: continue
        o = getattr(t, col(k))
        if o >= 99.99: continue
        os_.append(o); ws_.append(1.0 if t.res == k else 0.0)
    return stat_block(np.array(os_), np.array(ws_))

# ---- combos: outcome chosen relative to the favorite side
def mk_combo(home_pick, away_pick, strength=None):
    def fn(t):
        if strength is not None:
            f = min(t.oh, t.oa)
            if not (strength[0] < f <= strength[1]): return None
        if t.oh < t.oa: return home_pick
        if t.oa < t.oh: return away_pick
        return None
    return fn

STRENGTHS = {"any": None, "xtr": (1.0, 1.30), "str": (1.30, 1.60), "mid": (1.60, 2.00)}
COMBOS = {
    "fav_full_comeback":  ("2/1", "1/2"),   # fav trails HT, wins FT
    "fav_half_comeback":  ("2/X", "1/X"),   # fav trails HT, draws FT
    "fav_late_win":       ("X/1", "X/2"),   # draw HT, fav wins
    "dog_late_win":       ("X/2", "X/1"),   # draw HT, dog wins (drama)
}

n_tests = 0
scan = []
for k in OUTCOMES:
    for bn, bf in FAV_BUCKETS.items():
        scan.append({"sel": k, "cond": f"fav:{bn}", **eval_single(tr, k, mask=bf)}); n_tests += 1
    for lo, hi in ODDS_BANDS:
        scan.append({"sel": k, "cond": f"odds:[{lo},{hi})", **eval_single(tr, k, lo=lo, hi=hi)}); n_tests += 1
    for bn in ["homefav_any", "awayfav_any"]:
        bf = FAV_BUCKETS[bn]
        scan.append({"sel": k, "cond": f"fav:{bn}&odds>=3",
                     **eval_single(tr, k, mask=lambda s, f=bf, kk=k: f(s) & (s[col(kk)] >= 3))}); n_tests += 1
for cn, (hp, ap) in COMBOS.items():
    for sn, srng in STRENGTHS.items():
        scan.append({"sel": f"{hp}|{ap}", "cond": f"combo:{cn}/{sn}",
                     **eval_combo(tr, mk_combo(hp, ap, srng))}); n_tests += 1

print(f"train tests: {n_tests}")
cands = [r for r in scan if r.get("n", 0) >= 100 and r.get("roi", -9) >= 0.04]
cands.sort(key=lambda r: r.get("p_be", 1))
fmt = "{:<9} {:<26} {:>5} {:>5} {:>7} {:>8} {:>8} {:>9} {:>9}"
print(fmt.format("sel", "cond", "n", "wins", "wr", "odds", "ROI", "p_be", "p_t"))
for r in cands:
    print(fmt.format(r["sel"], r["cond"], r["n"], r["wins"], r["wr"], r["avg_odds"],
                     f"{r['roi']:+.4f}", r["p_be"], r["p_t"]))

print("\n=== validation: 8035-TEST + pooled-newleagues ===")
val = []
def revalidate(r, sub):
    cond, k = r["cond"], r["sel"]
    if cond.startswith("combo:"):
        cn, sn = cond[6:].split("/")
        hp, ap = COMBOS[cn]
        return eval_combo(sub, mk_combo(hp, ap, STRENGTHS[sn]))
    if "&odds>=3" in cond:
        bf = FAV_BUCKETS[cond[4:].replace("&odds>=3", "")]
        return eval_single(sub, k, mask=lambda s, f=bf, kk=k: f(s) & (s[col(kk)] >= 3))
    if cond.startswith("fav:"):
        return eval_single(sub, k, mask=FAV_BUCKETS[cond[4:]])
    lo, hi = [float(x) for x in cond[6:-1].split(",")]
    return eval_single(sub, k, lo=lo, hi=hi)

for r in cands:
    a_te, a_nw = revalidate(r, te), revalidate(r, new)
    a_all = revalidate(r, d)
    val.append({"sel": r["sel"], "cond": r["cond"], "train": r, "test8035": a_te,
                "newleagues": a_nw, "pooled9": a_all})
    print(f"{r['sel']:<9} {r['cond']:<26} TEST n={a_te.get('n',0):>4} roi={a_te.get('roi',0):+.4f} p_be={a_te.get('p_be',1):.4f} | "
          f"NEW n={a_nw.get('n',0):>5} roi={a_nw.get('roi',0):+.4f} p_be={a_nw.get('p_be',1):.4f} | "
          f"ALL n={a_all.get('n',0):>5} roi={a_all.get('roi',0):+.4f}")

out = json.load(open("exports/wf4_htft.json", encoding="utf-8"))
out["scan_step4"] = {"n_tests_scanned": n_tests, "train_grid": scan,
                     "candidates_validation": val}
json.dump(out, open("exports/wf4_htft.json", "w"), indent=1)
print(f"\nn_tests_scanned={n_tests}; saved to exports/wf4_htft.json")
