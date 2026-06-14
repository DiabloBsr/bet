"""Backtest : la prediction de score exact / total via inversion de grille +
deviations simulateur (Pass B) bat-elle la lecture brute du pricing (Pass A) ?
Compare aussi au baseline empirique par bucket et au plafond recalcule.

Arms (toutes history-free sauf bucket_empirical, qui s'entraine sur le TRAIN) :
  pricing       : grille Poisson(lam) issue du 1X2 (Pass A)
  sim_dc        : Pass B mode 'dc'    (rescale mu + Dixon-Coles rho)
  sim_cells     : Pass B mode 'cells' (rescale mu + boosts 2-1/1-2/2-2)
  bucket_emp    : score/total modal du bucket de favori, appris sur le TRAIN

Split forward chrono 70/30 par expected_start. Exclut corrupted_events.

Usage: ./.venv/Scripts/python.exe scripts/_inversion_backtest.py [--all-leagues]
"""
import sys
sys.path.insert(0, ".")
import argparse
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.analysis_utils import load_corrupted_ids
from scraper.market_inversion import (
    exact_invert_1x2, apply_sim_deviations, grid_predictions, total_distribution,
)
from scraper.predictor_v2 import poisson_score_grid

CAP = 6

ap = argparse.ArgumentParser()
ap.add_argument("--all-leagues", action="store_true")
args = ap.parse_args()

e = create_engine(load_settings().db_url)
corrupted = load_corrupted_ids()

comp_filter = "" if args.all_leagues else "AND e.competition='InstantLeague-8035'"
df = pd.read_sql(f"""
    SELECT e.id, e.competition, e.expected_start,
           o.odds_home oh, o.odds_draw od, o.odds_away oa,
           r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.expected_start IS NOT NULL
      AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
      {comp_filter}
""", e)
df = df[~df.id.isin(corrupted)].copy()
df["expected_start"] = pd.to_datetime(df.expected_start, utc=True, errors="coerce")
df = df.dropna(subset=["expected_start"]).sort_values("expected_start").reset_index(drop=True)

df["score"] = df.sa.astype(int).astype(str) + "-" + df.sb.astype(int).astype(str)
df["total"] = (df.sa.astype(int) + df.sb.astype(int)).clip(upper=CAP)
df["fav"] = df[["oh", "oa"]].min(axis=1)
print(f"n total (corrupted exclus) = {len(df)}  | leagues={'ALL' if args.all_leagues else '8035'}")

# --- split forward chrono 70/30 (par ligue pour rester apples-to-apples) ---
def split_chrono(g, frac=0.70):
    k = int(len(g) * frac)
    return g.iloc[:k], g.iloc[k:]

train_parts, test_parts = [], []
for _, g in df.groupby("competition"):
    tr, te = split_chrono(g)
    train_parts.append(tr); test_parts.append(te)
train = pd.concat(train_parts); test = pd.concat(test_parts).reset_index(drop=True)
print(f"train={len(train)}  test={len(test)}")

# --- baseline empirique par bucket (appris sur TRAIN) ---
BINS = [1.0, 1.2, 1.35, 1.5, 1.7, 2.0, 2.4, 3.0, 99]
train = train.copy(); test = test.copy()
train["bucket"] = pd.cut(train.fav, BINS)
test["bucket"] = pd.cut(test.fav, BINS)
bucket_score, bucket_total = {}, {}
for b, g in train.groupby("bucket", observed=True):
    if len(g) < 50:
        continue
    bucket_score[b] = g.score.value_counts().index[0]
    bucket_total[b] = int(g.total.value_counts().index[0])

# --- precompute lambda par event de TEST (1X2 exact) ---
print("inversion lambda sur le test...")
lams = [exact_invert_1x2(r.oh, r.od, r.oa) for r in test.itertuples()]
test["lam_h"] = [l[0] for l in lams]
test["lam_a"] = [l[1] for l in lams]

# --- arms : pour chaque event, top1/top3 score + total le + probable ---
def preds_from_grid(grid, k=3):
    gp = grid_predictions(grid, top_k=k)
    return gp["top_scores"][0][0], [s for s, _ in gp["top_scores"]], gp["most_likely_total"]

ARMS = ["pricing", "sim_dc", "sim_cells", "bucket_emp"]
hit_top1 = {a: [] for a in ARMS}
hit_top3 = {a: [] for a in ARMS}
hit_tot = {a: [] for a in ARMS}

for r in test.itertuples():
    lh, la = r.lam_h, r.lam_a
    gA = poisson_score_grid(lh, la, 0.0, 8)
    gDC = apply_sim_deviations(lh, la, "dc")
    gCE = apply_sim_deviations(lh, la, "cells")
    for arm, grid in (("pricing", gA), ("sim_dc", gDC), ("sim_cells", gCE)):
        t1, t3, tot = preds_from_grid(grid)
        hit_top1[arm].append(r.score == t1)
        hit_top3[arm].append(r.score in t3)
        hit_tot[arm].append(r.total == min(tot, CAP))
    # bucket empirique
    bs = bucket_score.get(r.bucket); bt = bucket_total.get(r.bucket)
    hit_top1["bucket_emp"].append(bs is not None and r.score == bs)
    hit_top3["bucket_emp"].append(False)  # 1 score modal seulement
    hit_tot["bucket_emp"].append(bt is not None and r.total == min(bt, CAP))

print("\n=== ARM COMPARISON (test OOS) ===")
print(f"{'arm':<12} {'exact_top1':>11} {'exact_top3':>11} {'total_top1':>11}")
for a in ARMS:
    t1 = 100 * np.mean(hit_top1[a]); t3 = 100 * np.mean(hit_top3[a]); tt = 100 * np.mean(hit_tot[a])
    t3s = f"{t3:>10.1f}%" if a != "bucket_emp" else f"{'n/a':>11}"
    print(f"{a:<12} {t1:>10.1f}% {t3s} {tt:>10.1f}%")

# plafond recalcule (in-sample sur tout df, oracle de reference)
df["bucket"] = pd.cut(df.fav, BINS)
ceil_rows = []
for b, g in df.groupby("bucket", observed=True):
    if len(g) < 150: continue
    vc = g.score.value_counts(normalize=True)
    ceil_rows.append((len(g), vc.iloc[0], vc.iloc[:3].sum()))
cw = np.array([r[0] for r in ceil_rows]); cw = cw / cw.sum()
print(f"\nplafond empirique (oracle, in-sample) : Top1 {100*sum(cw[i]*ceil_rows[i][1] for i in range(len(cw))):.1f}%  "
      f"Top3 {100*sum(cw[i]*ceil_rows[i][2] for i in range(len(cw))):.1f}%")

# --- McNemar : sim_cells vs pricing (exact top1) ---
def mcnemar(a, b):
    a = np.array(a); b = np.array(b)
    b01 = int(np.sum(a & ~b)); b10 = int(np.sum(~a & b))
    if b01 + b10 == 0: return 0.0, b01, b10
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return chi2, b01, b10

for arm in ("sim_dc", "sim_cells"):
    chi2, w, l = mcnemar(hit_top1[arm], hit_top1["pricing"])
    # p approx via chi2 1 dof survival
    from scipy.stats import chi2 as chi2dist
    p = 1 - chi2dist.cdf(chi2, 1)
    print(f"McNemar {arm} vs pricing (top1): chi2={chi2:.2f} p={p:.4f}  ({arm}+/pricing- ={w}, pricing+/{arm}- ={l})")

# --- stratification par bucket : ou Pass B aide le plus ---
print("\n=== exact_top1 par bucket de favori (pricing -> sim_cells) ===")
test["h_pricing"] = hit_top1["pricing"]; test["h_simcells"] = hit_top1["sim_cells"]
test["h_pricing3"] = hit_top3["pricing"]; test["h_simcells3"] = hit_top3["sim_cells"]
for b, g in test.groupby("bucket", observed=True):
    if len(g) < 80: continue
    print(f"  {str(b):<12} n={len(g):>4}  top1 {100*g.h_pricing.mean():>5.1f}->{100*g.h_simcells.mean():>5.1f}%   "
          f"top3 {100*g.h_pricing3.mean():>5.1f}->{100*g.h_simcells3.mean():>5.1f}%")
