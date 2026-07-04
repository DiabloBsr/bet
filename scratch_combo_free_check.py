# -*- coding: utf-8 -*-
# Stress-test du "survivant" (W, D, DOWN) side=HOME
import pandas as pd, numpy as np
from scipy import stats

CSV = "d:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv"
d = pd.read_csv(CSV)
d['ts'] = pd.to_datetime(d.ts)
h = d[d.venue == 'H'].copy()
a = d[d.venue == 'A'][['ts', 'team', 'p1_result', 'd_odds', 'odds', 'imp']].copy()
a = a.rename(columns={'team': 'away_team', 'p1_result': 'away_p1_result',
                      'd_odds': 'away_d_odds', 'odds': 'away_odds', 'imp': 'away_imp'})
h = h.drop_duplicates(subset=['ts', 'team'], keep='first')
a = a.drop_duplicates(subset=['ts', 'away_team'], keep='first')
m = h.merge(a, left_on=['opp', 'ts'], right_on=['away_team', 'ts'], how='inner')
m['home_win'] = (m.gf > m.ga).astype(int)

cell = m[(m.p1_result == 'W') & (m.away_p1_result == 'D') & (m.d_odds < 0)].sort_values('ts')
print("n cellule total:", len(cell))
print("resid global cellule:", round(float((cell.home_win - cell.imp).mean()), 4),
      "| ROI global:", round(float((cell.home_win * cell.odds - 1).mean()), 4))

med = m.ts.median()
tr = cell[cell.ts <= med]; te = cell[cell.ts > med]
print("\nTRAIN n=%d resid=%.4f ROI=%.4f" % (len(tr), (tr.home_win - tr.imp).mean(),
                                            (tr.home_win * tr.odds - 1).mean()))
print("TEST  n=%d resid=%.4f ROI=%.4f" % (len(te), (te.home_win - te.imp).mean(),
                                          (te.home_win * te.odds - 1).mean()))

# --- stabilite : 6 tranches chrono sur toute la periode ---
cell = cell.reset_index(drop=True)
cell['bin'] = pd.qcut(cell.index, 6, labels=False)
print("\n--- 6 tranches chrono (cellule complete) ---")
for b, g in cell.groupby('bin'):
    print("tranche %d  n=%3d  ts[%s -> %s]  resid=%+0.4f  ROI=%+0.4f" % (
        b, len(g), g.ts.min().date(), g.ts.max().date(),
        (g.home_win - g.imp).mean(), (g.home_win * g.odds - 1).mean()))

# --- TEST coupe en 2 ---
te = te.reset_index(drop=True)
half = len(te) // 2
for i, gg in enumerate([te.iloc[:half], te.iloc[half:]]):
    k = int(gg.home_win.sum()); n = len(gg); p0 = float(gg.imp.mean())
    pv = stats.binomtest(k, n, p0, alternative='greater').pvalue
    print("TEST moitie %d: n=%d resid=%+0.4f ROI=%+0.4f p(binom)=%.4f" % (
        i + 1, n, (gg.home_win - gg.imp).mean(), (gg.home_win * gg.odds - 1).mean(), pv))

# --- sensibilite a la definition de DOWN : seuil strict ---
for thr in [0.0, -0.05, -0.10, -0.20]:
    c2 = m[(m.p1_result == 'W') & (m.away_p1_result == 'D') & (m.d_odds < thr)]
    t2 = c2[c2.ts > med]
    if len(t2):
        print("d_odds<%.2f : TEST n=%4d resid=%+0.4f ROI=%+0.4f" % (
            thr, len(t2), (t2.home_win - t2.imp).mean(), (t2.home_win * t2.odds - 1).mean()))

# --- cellules soeurs (permutation des resultats precedents), TEST only ---
print("\n--- cellules soeurs en TEST (side HOME, dirH=DOWN) ---")
for r1 in ['W', 'D', 'L']:
    for r2 in ['W', 'D', 'L']:
        c2 = m[(m.p1_result == r1) & (m.away_p1_result == r2) & (m.d_odds < 0)]
        t2 = c2[c2.ts > med]
        if len(t2) > 50:
            print("(%s,%s,DOWN) TEST n=%4d resid=%+0.4f ROI=%+0.4f" % (
                r1, r2, len(t2), (t2.home_win - t2.imp).mean(), (t2.home_win * t2.odds - 1).mean()))
