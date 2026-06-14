# -*- coding: utf-8 -*-
# WF4 - score exact - etape 4: exploration des cracks
# (a) hypothese "le book price la VRAIE distribution + marge flat":
#     pour chaque cellule x groupe: (1+margin_med) / real_over_grid = 1 + marge_flat ?
# (b) value observable par event: v = cote_offerte x p_grille(cellule)
#     -> ROI par quintile de v pour 2-1 / 1-2 / 1-1 / 2-2 (pooled9 et 8035)
# (c) deviation real/grid par profil de match (8035 TRAIN 70% seulement):
#     buckets de favspread |lh-la| et de mu total -> ou le boost 2-1/1-2 est-il max?
# Entrees: exports/wf4_scoreexact_data.json + exports/wf4_scoreexact_lambdas.npz
import json, math
import numpy as np

with open("exports/wf4_scoreexact_data.json", encoding="utf-8") as f:
    data = json.load(f)
rows = [r for r in data["rows"]
        if r["oh"] and r["od"] and r["oa"] and r["oh"] > 1 and r["od"] > 1 and r["oa"] > 1]
lz = np.load("exports/wf4_scoreexact_lambdas.npz")
lh, la, ids = lz["lh"], lz["la"], lz["ids"]
assert len(rows) == len(lh) and all(r["id"] == i for r, i in zip(rows[:50], ids[:50]))

comp = np.array([r["comp"] for r in rows])
start = np.array([r["start"] for r in rows])
sa = np.array([r["sa"] for r in rows])
sb = np.array([r["sb"] for r in rows])
n = len(rows)
ks = np.arange(16)
logfact = np.array([math.lgamma(k + 1) for k in ks])
pa_ = np.exp(-lh[:, None] + ks[None, :] * np.log(lh[:, None]) - logfact[None, :])
pb_ = np.exp(-la[:, None] + ks[None, :] * np.log(la[:, None]) - logfact[None, :])
cp = pa_[:, :7, None] * pb_[:, None, :7]

cells = ["%d-%d" % (i, j) for i in range(7) for j in range(7) if i + j <= 6]
se_odds = np.full((n, len(cells)), np.nan)
for k, r in enumerate(rows):
    for ci, c in enumerate(cells):
        v = r["se"].get(c)
        if v is not None:
            se_odds[k, ci] = float(v)

with open("exports/wf4_scoreexact_cells.json", encoding="utf-8") as f:
    cellsjson = json.load(f)

# ---------- (a) book-knows-truth ----------
print("=== (a) (1+margin)/ratio - 1 = marge flat implicite par cellule ===")
for gname in ("8035", "champs5", "cups3"):
    g = cellsjson["groups"][gname]
    ents = []
    for cname, v in g.items():
        if v["n_offered"] >= 1000 and v["exp_wins_grid"] >= 30:
            flat = (1 + v["cell_margin_med"]) / v["real_over_grid"] - 1
            ents.append((cname, v["real_over_grid"], v["cell_margin_med"], round(flat, 3)))
    ents.sort(key=lambda x: x[3])
    print(gname, "->", ents)

# ---------- (b) ROI par quintile de value observable ----------
print("\n=== (b) ROI par quintile de v = cote x p_grille (pooled9) ===")
m8035 = comp == "InstantLeague-8035"
order = np.argsort(start)
r8035 = order[m8035[order]]
cut = int(len(r8035) * 0.7)
train_idx = np.zeros(n, bool); train_idx[r8035[:cut]] = True
test_idx = np.zeros(n, bool); test_idx[r8035[cut:]] = True
newl = ~m8035

for cname in ("2-1", "1-2", "1-1", "2-2", "1-0", "2-0"):
    ci = cells.index(cname)
    i, j = int(cname[0]), int(cname[2])
    ok = ~np.isnan(se_odds[:, ci]) & (se_odds[:, ci] < 99.5)
    v = se_odds[:, ci] * cp[:, i, j]
    win = (sa == i) & (sb == j)
    ret = se_odds[:, ci] * win - 1
    qs = np.nanquantile(v[ok], [0.2, 0.4, 0.6, 0.8])
    print("%s: quintiles v=%s" % (cname, np.round(qs, 3)))
    for qlo, qhi, lab in [(-1, qs[0], "Q1"), (qs[0], qs[1], "Q2"), (qs[1], qs[2], "Q3"),
                          (qs[2], qs[3], "Q4"), (qs[3], 99, "Q5")]:
        m = ok & (v > qlo) & (v <= qhi)
        if m.sum() < 30:
            continue
        wr = win[m].mean()
        roi = ret[m].mean()
        # ratio reel/grille dans le bucket
        ratio = win[m].sum() / cp[m, i, j].sum()
        print("  %s n=%5d v_avg=%.3f odds=%6.2f ratio=%.3f ROI=%+7.2f%%" % (
            lab, m.sum(), v[m].mean(), np.nanmean(se_odds[m, ci]), ratio, roi.mean() * 100))

# ---------- (b2) idem mais seulement nouvelles ligues ----------
print("\n=== (b2) v-quintiles, nouvelles ligues (pooled-newleagues) ===")
for cname in ("2-1", "1-2"):
    ci = cells.index(cname)
    i, j = int(cname[0]), int(cname[2])
    ok = newl & ~np.isnan(se_odds[:, ci]) & (se_odds[:, ci] < 99.5)
    v = se_odds[:, ci] * cp[:, i, j]
    win = (sa == i) & (sb == j)
    ret = se_odds[:, ci] * win - 1
    qs = np.nanquantile(v[ok], [0.2, 0.4, 0.6, 0.8])
    print("%s: quintiles v=%s" % (cname, np.round(qs, 3)))
    for qlo, qhi, lab in [(-1, qs[0], "Q1"), (qs[0], qs[1], "Q2"), (qs[1], qs[2], "Q3"),
                          (qs[2], qs[3], "Q4"), (qs[3], 99, "Q5")]:
        m = ok & (v > qlo) & (v <= qhi)
        if m.sum() < 30:
            continue
        ratio = win[m].sum() / cp[m, i, j].sum()
        print("  %s n=%5d v_avg=%.3f odds=%6.2f ratio=%.3f ROI=%+7.2f%%" % (
            lab, m.sum(), v[m].mean(), np.nanmean(se_odds[m, ci]), ratio,
            ret[m].mean() * 100))

# ---------- (c) deviation par profil, 8035 TRAIN uniquement ----------
print("\n=== (c) ratio reel/grille par profil (8035 TRAIN, n=%d) ===" % train_idx.sum())
spread = lh - la
mu = lh + la
for cname in ("2-1", "1-2", "1-1", "2-2"):
    ci = cells.index(cname)
    i, j = int(cname[0]), int(cname[2])
    win = (sa == i) & (sb == j)
    print(cname)
    for feat, name, bins in ((spread, "spread", [-9, -0.5, 0, 0.5, 1.0, 9]),
                             (mu, "mu_tot", [0, 2.4, 2.8, 3.2, 9])):
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = train_idx & (feat > lo) & (feat <= hi) & ~np.isnan(se_odds[:, ci]) & (se_odds[:, ci] < 99.5)
            if m.sum() < 100:
                continue
            ratio = win[m].sum() / cp[m, i, j].sum()
            roi = (se_odds[m, ci] * win[m] - 1).mean()
            print("  %s (%.1f,%.1f] n=%5d ratio=%.3f ROI=%+7.2f%%" % (
                name, lo, hi, m.sum(), ratio, roi * 100))
