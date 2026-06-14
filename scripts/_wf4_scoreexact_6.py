# -*- coding: utf-8 -*-
# WF4 - score exact - etape 6: scans complementaires + bilan
# (1) selection sur TRAIN-8035: cellules a ROI train > 0 (n>=300) -> eval TEST + NEWLEAGUES
# (2) toujours-2-1 / toujours-1-2 par ligue (cote moyenne, ROI, z) — documentation anti-finding
# (3) basket "cellules genereuses" (marge cellule mediane TRAIN < 0) -> eval TEST + NEWL
# (4) bilan: ROI pooled-9 de toutes les cellules — y a-t-il UNE cellule p<0.01 positive?
# Sortie: exports/wf4_scoreexact_final.json
import json, math
import numpy as np
from scipy.stats import norm

with open("exports/wf4_scoreexact_data.json", encoding="utf-8") as f:
    data = json.load(f)
rows = [r for r in data["rows"]
        if r["oh"] and r["od"] and r["oa"] and r["oh"] > 1 and r["od"] > 1 and r["oa"] > 1]
lz = np.load("exports/wf4_scoreexact_lambdas.npz")
lh, la = lz["lh"], lz["la"]
n = len(rows)
comp = np.array([r["comp"] for r in rows])
start = np.array([r["start"] for r in rows])
sa = np.array([r["sa"] for r in rows])
sb = np.array([r["sb"] for r in rows])

ks = np.arange(16)
logfact = np.array([math.lgamma(k + 1) for k in ks])
pa_ = np.exp(-lh[:, None] + ks[None, :] * np.log(lh[:, None]) - logfact[None, :])
pb_ = np.exp(-la[:, None] + ks[None, :] * np.log(la[:, None]) - logfact[None, :])
cp = pa_[:, :7, None] * pb_[:, None, :7]

cells = ["%d-%d" % (i, j) for i in range(7) for j in range(7) if i + j <= 6]
cell_ij = [(int(c[0]), int(c[2])) for c in cells]
se_odds = np.full((n, len(cells)), np.nan)
for k, r in enumerate(rows):
    for ci, c in enumerate(cells):
        v = r["se"].get(c)
        if v is not None and float(v) < 99.5:
            se_odds[k, ci] = float(v)

m8035 = comp == "InstantLeague-8035"
order = np.argsort(start)
r8035 = order[m8035[order]]
cut = int(len(r8035) * 0.7)
TRAIN = np.zeros(n, bool); TRAIN[r8035[:cut]] = True
TEST = np.zeros(n, bool); TEST[r8035[cut:]] = True
NEWL = ~m8035


def eval_bets(mask2d):
    """mask2d[n, ncells] bool -> stats"""
    idx, cis = np.where(mask2d)
    if len(idx) == 0:
        return None
    odds = se_odds[idx, cis]
    win = (sa[idx] == np.array([cell_ij[c][0] for c in cis])) & \
          (sb[idx] == np.array([cell_ij[c][1] for c in cis]))
    ret = odds * win - 1
    q0 = 1 / odds
    var0 = (q0 * (odds - 1) ** 2 + (1 - q0)).sum()
    z = ret.sum() / math.sqrt(var0) if var0 > 0 else 0.0
    return {"n": len(idx), "wins": int(win.sum()), "wr": round(float(win.mean()), 4),
            "avg_odds": round(float(odds.mean()), 2),
            "roi_pct": round(float(ret.mean()) * 100, 2),
            "z": round(float(z), 2), "p_one_sided": round(float(1 - norm.cdf(z)), 5)}


out = {}

# ---------- (1) cellules ROI>0 sur TRAIN ----------
print("=== (1) cellules ROI train > 0 (n>=300) -> TEST / NEWL ===")
sel = []
for ci, (i, j) in enumerate(cell_ij):
    m = TRAIN & ~np.isnan(se_odds[:, ci])
    if m.sum() < 300:
        continue
    win = (sa[m] == i) & (sb[m] == j)
    roi = (se_odds[m, ci] * win - 1).mean()
    if roi > 0:
        sel.append((cells[ci], ci, round(float(roi) * 100, 2), int(m.sum())))
out["train_positive_cells"] = {}
for cname, ci, roitr, ntr in sorted(sel, key=lambda x: -x[2]):
    res = {}
    for scope, base in (("TEST", TEST), ("NEWL", NEWL)):
        mk = np.zeros((n, len(cells)), bool)
        mk[base & ~np.isnan(se_odds[:, ci]), ci] = True
        res[scope] = eval_bets(mk)
    out["train_positive_cells"][cname] = {"train_roi_pct": roitr, "train_n": ntr, **res}
    t, nl = res["TEST"], res["NEWL"]
    print("%-4s train %+6.2f%% (n=%d) | TEST n=%5d ROI=%+7.2f%% z=%+5.2f | NEWL n=%5d ROI=%+7.2f%% z=%+5.2f" % (
        cname, roitr, ntr, t["n"], t["roi_pct"], t["z"], nl["n"], nl["roi_pct"], nl["z"]))

# ---------- (2) toujours 2-1 / 1-2 par ligue ----------
print("\n=== (2) toujours 2-1 / 1-2 par ligue (full pool par ligue) ===")
out["per_league_always"] = {}
for lg in sorted(set(comp)):
    ent = {}
    for cname in ("2-1", "1-2"):
        ci = cells.index(cname)
        mk = np.zeros((n, len(cells)), bool)
        mk[(comp == lg) & ~np.isnan(se_odds[:, ci]), ci] = True
        ent[cname] = eval_bets(mk)
    out["per_league_always"][lg] = ent
    a, b = ent["2-1"], ent["1-2"]
    print("%s: 2-1 n=%5d odds=%5.2f ROI=%+7.2f%% z=%+5.2f | 1-2 n=%5d odds=%5.2f ROI=%+7.2f%% z=%+5.2f" % (
        lg, a["n"], a["avg_odds"], a["roi_pct"], a["z"], b["n"], b["avg_odds"], b["roi_pct"], b["z"]))

# ---------- (3) basket cellules genereuses (marge mediane TRAIN < 0) ----------
print("\n=== (3) basket marge<0 sur TRAIN -> TEST / NEWL ===")
gen = []
for ci, (i, j) in enumerate(cell_ij):
    m = TRAIN & ~np.isnan(se_odds[:, ci])
    if m.sum() < 300:
        continue
    marg = float(np.median(1 / (se_odds[m, ci] * cp[m, i, j]) - 1))
    if marg < 0:
        gen.append((cells[ci], ci, marg))
print("cellules genereuses train:", [(c, round(m, 3)) for c, _, m in gen])
out["generous_basket_cells"] = [c for c, _, _ in gen]
res = {}
for scope, base in (("TEST", TEST), ("NEWL", NEWL)):
    mk = np.zeros((n, len(cells)), bool)
    for _, ci, _ in gen:
        mk[base & ~np.isnan(se_odds[:, ci]), ci] = True
    res[scope] = eval_bets(mk)
    r = res[scope]
    print("%s: n=%d odds=%.2f ROI=%+.2f%% z=%+.2f p=%.4f" % (
        scope, r["n"], r["avg_odds"], r["roi_pct"], r["z"], r["p_one_sided"]))
out["generous_basket"] = res

# ---------- (4) bilan pooled-9 toutes cellules ----------
print("\n=== (4) pooled-9: existe-t-il une cellule ROI>0 p<0.01 ? ===")
best = None
out["pooled9_cells"] = {}
for ci, (i, j) in enumerate(cell_ij):
    mk = np.zeros((n, len(cells)), bool)
    mk[~np.isnan(se_odds[:, ci]), ci] = True
    r = eval_bets(mk)
    if r is None or r["n"] < 150:
        continue
    out["pooled9_cells"][cells[ci]] = r
    if best is None or r["z"] > best[1]["z"]:
        best = (cells[ci], r)
print("meilleure cellule pooled-9: %s -> %s" % best)

with open("exports/wf4_scoreexact_final.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("ecrit: exports/wf4_scoreexact_final.json")
