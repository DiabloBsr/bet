# -*- coding: utf-8 -*-
# WF4 - score exact - etape 5: backtest walk-forward des strategies
# Split temporel 8035: 70% train / 30% test (par expected_start).
# Les multiplicateurs/regles sont definis sur TRAIN-8035 uniquement, evalues sur:
#   - 8035-TEST (walk-forward)
#   - pooled-newleagues (8 ligues, OOS total vs train 8035)
# Strategies (mise 1u a la cote offerte, cellules cote<99.5 uniquement):
#   S01 toujours 2-1
#   S02 toujours 1-2
#   S03 combo 2-1 + 1-2 (2 mises par match)
#   S04 2-1 si favori home modere (oh in [1.5, 2.2])
#   S05 1-2 si favori away modere (oa in [1.5, 2.2])
#   S06 score modal de la grille pure
#   S07 score modal V2-style (grille x multiplicateurs TRAIN par cellule)
#   S08 2-1 si away favori (spread lh-la <= -0.5)
#   S09 1-2 si mu total <= 2.4
#   S10 2-1 si mu total in (2.4, 2.8]
#   S11 cellule de meilleure EV TRAIN-ajustee si EV>1.0 (max 1 mise/match)
#   S12 combo low-mu: 2-1 ET 1-2 si mu <= 2.4
#   S13 1-2 si mu <= 2.4 ET marge cellule observable <= mediane train (price filter)
# p-value: z-test H0 EV=0 a la cote offerte (one-sided, profitable)
# Sortie: exports/wf4_scoreexact_strats.json
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
oh = np.array([r["oh"] for r in rows])
oa = np.array([r["oa"] for r in rows])

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
print("train=%d test=%d newleagues=%d" % (TRAIN.sum(), TEST.sum(), NEWL.sum()))

mu = lh + la
spread = lh - la

# ---------- multiplicateurs TRAIN par cellule ----------
mult = np.ones(len(cells))
for ci, (i, j) in enumerate(cell_ij):
    m = TRAIN & ~np.isnan(se_odds[:, ci])
    exp_w = cp[m, i, j].sum()
    if exp_w >= 10:
        mult[ci] = ((sa[m] == i) & (sb[m] == j)).sum() / exp_w
print("multiplicateurs train:", {c: round(float(v), 3) for c, v in zip(cells, mult)
                                 if abs(v - 1) > 0.1})

ci21 = cells.index("2-1")
ci12 = cells.index("1-2")

# ---------- price-filter train (pour S13, defini sur TRAIN seulement) ----------
m = TRAIN & ~np.isnan(se_odds[:, ci12]) & (mu <= 2.4)
marg12 = 1 / (se_odds[:, ci12] * cp[:, 1, 2]) - 1
med_marg12_train = float(np.median(marg12[m]))
lo = m & (marg12 <= med_marg12_train)
hi = m & (marg12 > med_marg12_train)
for lab, mm in (("marge<=med", lo), ("marge>med", hi)):
    win = (sa[mm] == 1) & (sb[mm] == 2)
    roi = (se_odds[mm, ci12] * win - 1).mean()
    print("TRAIN check S13 %s: n=%d ROI=%+.2f%%" % (lab, mm.sum(), roi * 100))

# ---------- definition des strategies ----------
# une strategie = liste de (event_idx, cell_idx) paris
def bets_always(ci, base):
    m = base & ~np.isnan(se_odds[:, ci])
    return [(k, ci) for k in np.where(m)[0]]

def bets_modal(base, multv):
    out = []
    adj = cp[:, [i for i, j in cell_ij], [j for i, j in cell_ij]] * multv[None, :]
    for k in np.where(base)[0]:
        row = adj[k].copy()
        row[np.isnan(se_odds[k])] = -1
        ci = int(row.argmax())
        if row[ci] > 0:
            out.append((k, ci))
    return out

def bets_bestev(base, multv, thr=1.0):
    out = []
    adj = cp[:, [i for i, j in cell_ij], [j for i, j in cell_ij]] * multv[None, :]
    ev = adj * se_odds  # nan si pas offert
    for k in np.where(base)[0]:
        row = ev[k]
        if np.all(np.isnan(row)):
            continue
        ci = int(np.nanargmax(row))
        if row[ci] > thr:
            out.append((k, ci))
    return out

ALL = np.ones(n, bool)
strats = {
    "S01_always_2-1": lambda base: bets_always(ci21, base),
    "S02_always_1-2": lambda base: bets_always(ci12, base),
    "S03_combo_21_12": lambda base: bets_always(ci21, base) + bets_always(ci12, base),
    "S04_21_homefav_1.5-2.2": lambda base: bets_always(ci21, base & (oh >= 1.5) & (oh <= 2.2)),
    "S05_12_awayfav_1.5-2.2": lambda base: bets_always(ci12, base & (oa >= 1.5) & (oa <= 2.2)),
    "S06_modal_grid": lambda base: bets_modal(base, np.ones(len(cells))),
    "S07_modal_v2_trainmult": lambda base: bets_modal(base, mult),
    "S08_21_awayfav_spread": lambda base: bets_always(ci21, base & (spread <= -0.5)),
    "S09_12_mu_le_2.4": lambda base: bets_always(ci12, base & (mu <= 2.4)),
    "S10_21_mu_2.4_2.8": lambda base: bets_always(ci21, base & (mu > 2.4) & (mu <= 2.8)),
    "S11_bestev_thr1.0": lambda base: bets_bestev(base, mult, 1.0),
    "S12_combo_lowmu": lambda base: (bets_always(ci21, base & (mu <= 2.4))
                                     + bets_always(ci12, base & (mu <= 2.4))),
    "S13_12_lowmu_cheap": lambda base: bets_always(
        ci12, base & (mu <= 2.4) & (marg12 <= med_marg12_train)),
}

def evaluate(bets):
    if not bets:
        return None
    idx = np.array([b[0] for b in bets])
    cis = np.array([b[1] for b in bets])
    odds = se_odds[idx, cis]
    win = (sa[idx] == np.array([cell_ij[c][0] for c in cis])) & \
          (sb[idx] == np.array([cell_ij[c][1] for c in cis]))
    ret = odds * win - 1
    nb = len(bets)
    q0 = 1 / odds
    var0 = (q0 * (odds - 1) ** 2 + (1 - q0)).sum()
    z = ret.sum() / math.sqrt(var0) if var0 > 0 else 0.0
    return {
        "n": nb, "wins": int(win.sum()), "wr": round(float(win.mean()), 4),
        "avg_odds": round(float(odds.mean()), 2),
        "roi_pct": round(float(ret.mean()) * 100, 2),
        "z": round(float(z), 2), "p_one_sided": round(float(1 - norm.cdf(z)), 5),
    }

scopes = {"8035_TRAIN": TRAIN, "8035_TEST": TEST, "NEWLEAGUES": NEWL}
out = {}
for sname, fn in strats.items():
    out[sname] = {}
    for scname, base in scopes.items():
        r = evaluate(fn(base))
        out[sname][scname] = r
    tr, te, nl = out[sname]["8035_TRAIN"], out[sname]["8035_TEST"], out[sname]["NEWLEAGUES"]
    def fmt(r):
        if r is None:
            return "n=0"
        return "n=%5d wr=%.3f odds=%6.2f ROI=%+7.2f%% z=%+5.2f p=%.4f" % (
            r["n"], r["wr"], r["avg_odds"], r["roi_pct"], r["z"], r["p_one_sided"])
    print("%-24s TRAIN %s" % (sname, fmt(tr)))
    print("%-24s TEST  %s" % ("", fmt(te)))
    print("%-24s NEWL  %s" % ("", fmt(nl)))

with open("exports/wf4_scoreexact_strats.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print("ecrit: exports/wf4_scoreexact_strats.json")
