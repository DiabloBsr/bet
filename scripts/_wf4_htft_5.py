# -*- coding: utf-8 -*-
"""WF4 HT/FT miner v2 - step 5: last axis (mu regime via opening draw odds) +
overround per league + bootstrap on the least-bad cells.

- mu proxy: opening draw odds od (high od ~ high expected goals). Terciles
  computed on 8035-TRAIN only, applied unchanged to test/new (no leakage).
- 9 outcomes x 3 od-terciles, train -> validate survivors.
- Overround of the HT/FT book per league (documentation).
Appends to exports/wf4_htft.json.
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

q1, q2 = tr["od"].quantile([1/3, 2/3])
print(f"od terciles (train): {q1:.3f} / {q2:.3f}")
TERC = {"lowmu": lambda s: s["od"] >= q2,           # high draw odds = ... check direction
        "midmu": lambda s: (s["od"] > q1) & (s["od"] < q2),
        "highmu": lambda s: s["od"] <= q1}
# NB direction: in Poisson grids, higher mu => draw less likely => HIGHER draw odds.
# So od >= q2 is HIGH mu. Fix labels accordingly:
TERC = {"highmu_od_hi": lambda s: s["od"] >= q2,
        "midmu": lambda s: (s["od"] > q1) & (s["od"] < q2),
        "lowmu_od_lo": lambda s: s["od"] <= q1}

def stat_block(o, w):
    n = len(o)
    if n == 0: return {"n": 0}
    pv = w * o - 1.0
    profit = pv.sum(); p0 = 1.0 / o
    var0 = (o * o * p0 * (1 - p0)).sum()
    z = profit / math.sqrt(var0) if var0 > 0 else 0.0
    return {"n": int(n), "wins": int(w.sum()), "wr": round(float(w.mean()), 4),
            "avg_odds": round(float(o.mean()), 3), "roi": round(float(profit / n), 4),
            "p_be": round(float(1 - stats.norm.cdf(z)), 6)}

def ev(sub, k, mask):
    c_ = col(k)
    s = sub[sub[c_] < 99.99]
    s = s[mask(s)]
    return stat_block(s[c_].values, (s["res"] == k).astype(float).values)

n_tests = 0
scan = []
for k in OUTCOMES:
    for tn, tf in TERC.items():
        r = ev(tr, k, tf); n_tests += 1
        scan.append({"sel": k, "cond": f"mu:{tn}", **r})
cands = [r for r in scan if r.get("n", 0) >= 100 and r.get("roi", -9) >= 0.04]
cands.sort(key=lambda r: r.get("p_be", 1))
print(f"mu-axis train tests: {n_tests}, candidates: {len(cands)}")
val = []
for r in cands:
    tf = TERC[r["cond"][3:]]
    a_te, a_nw = ev(te, r["sel"], tf), ev(new, r["sel"], tf)
    val.append({"sel": r["sel"], "cond": r["cond"], "train": r, "test8035": a_te, "newleagues": a_nw})
    print(f"{r['sel']} {r['cond']}: TRAIN n={r['n']} roi={r['roi']:+.4f} p={r['p_be']:.4f} | "
          f"TEST n={a_te.get('n',0)} roi={a_te.get('roi',0):+.4f} p={a_te.get('p_be',1):.4f} | "
          f"NEW n={a_nw.get('n',0)} roi={a_nw.get('roi',0):+.4f} p={a_nw.get('p_be',1):.4f}")

# ---- overround per league (HT/FT book, opening)
print("\n=== HT/FT overround per league (sum 1/odds, capped sel. excluded -> lower bound) ===")
ovr = {}
for lg, sub in d.groupby("lg"):
    inv = np.zeros(len(sub))
    for k in OUTCOMES:
        o = sub[col(k)].values
        inv = inv + np.where(o < 99.99, 1.0 / o, 0.0)
    ovr[lg] = round(float(np.mean(inv)), 4)
    print(f"{lg}: {ovr[lg]}")

# ---- bootstrap on the two least-bad pooled cells (2/X pooled, X/2 pooled)
print("\n=== bootstrap 10k, pooled-9 blind, least-bad outcomes ===")
boot = {}
rng = np.random.default_rng(42)
for k in ["2/X", "X/2"]:
    c_ = col(k)
    s = d[d[c_] < 99.99]
    pv = ((s["res"] == k).astype(float) * s[c_] - 1.0).values
    means = np.array([rng.choice(pv, size=len(pv), replace=True).mean() for _ in range(10000)])
    boot[k] = {"roi": round(float(pv.mean()), 4),
               "ci95": [round(float(np.percentile(means, 2.5)), 4),
                        round(float(np.percentile(means, 97.5)), 4)],
               "p_roi_pos": round(float((means <= 0).mean()), 4)}
    print(k, boot[k])

out = json.load(open("exports/wf4_htft.json", encoding="utf-8"))
out["scan_step5_mu"] = {"n_tests_scanned": n_tests, "od_terciles_train": [float(q1), float(q2)],
                        "grid": scan, "validation": val}
out["overround_htft_by_league"] = ovr
out["bootstrap_pooled_leastbad"] = boot
json.dump(out, open("exports/wf4_htft.json", "w"), indent=1)
print("appended to exports/wf4_htft.json")
