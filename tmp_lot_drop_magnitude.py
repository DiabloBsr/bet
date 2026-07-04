import pandas as pd, numpy as np
from scipy import stats

CSV = 'D:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv'
d = pd.read_csv(CSV)
print("rows total:", len(d))

# Lot drop_magnitude: needs d_odds (requires p1) + venue + win/imp/odds
d = d.dropna(subset=['d_odds', 'venue', 'win', 'imp', 'odds']).copy()
print("rows with d_odds:", len(d))

# Chrono split on median ts
d['ts'] = pd.to_datetime(d['ts'])
med = d['ts'].median()
train = d[d['ts'] <= med]
test = d[d['ts'] > med]
print("median ts:", med, "| n_train:", len(train), "| n_test:", len(test))

# Fixed bands for d_odds
edges = [-np.inf, -0.5, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, np.inf]
labels = ['<=-0.5', '-0.5..-0.3', '-0.3..-0.15', '-0.15..-0.05', '-0.05..+0.05',
          '+0.05..+0.15', '+0.15..+0.3', '>+0.3']
for df in (train, test):
    df['band'] = pd.cut(df['d_odds'], bins=edges, labels=labels)

cells = []
for band in labels:
    for ven in ['H', 'A']:
        tr = train[(train['band'] == band) & (train['venue'] == ven)]
        te = test[(test['band'] == band) & (test['venue'] == ven)]
        n_tr = len(tr)
        resid_tr = tr['win'].mean() - tr['imp'].mean() if n_tr else np.nan
        cells.append(dict(cell=f"d_odds {band} x venue={ven}", band=band, venue=ven,
                          n_train=n_tr, resid_train=resid_tr,
                          n_test=len(te),
                          wins_test=int(te['win'].sum()) if len(te) else 0,
                          imp_test=te['imp'].mean() if len(te) else np.nan,
                          resid_test=(te['win'].mean() - te['imp'].mean()) if len(te) else np.nan,
                          roi_test=(te['win'] * te['odds'] - 1).mean() if len(te) else np.nan))

cells = pd.DataFrame(cells)
print("\n=== ALL 16 CELLS ===")
print(cells[['cell', 'n_train', 'resid_train', 'n_test', 'resid_test', 'roi_test']].to_string(index=False))

# Step 2: TRAIN filter
cand = cells[(cells['n_train'] >= 150) & (cells['resid_train'].abs() >= 0.02)].copy()
print("\ncandidates passing TRAIN filter (n>=150 & |resid|>=0.02):", len(cand))

# Step 3: OOS test + BH-FDR over all tested cells of the lot
if len(cand):
    pvals = []
    for _, r in cand.iterrows():
        if r['n_test'] > 0:
            p = stats.binomtest(int(r['wins_test']), int(r['n_test']), float(r['imp_test'])).pvalue
        else:
            p = 1.0
        pvals.append(p)
    cand['p_test'] = pvals
    cand['same_sign'] = np.sign(cand['resid_test']) == np.sign(cand['resid_train'])
    # BH-FDR on ALL tested cells of the lot
    m = len(cand)
    order = np.argsort(cand['p_test'].values)
    ranked = cand['p_test'].values[order]
    bh = ranked * m / (np.arange(m) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    padj = np.empty(m)
    padj[order] = np.clip(bh, 0, 1)
    cand['p_bh'] = padj
    cand['survivor'] = cand['same_sign'] & (cand['p_bh'] < 0.05)
    print("\n=== CANDIDATE CELLS (OOS) ===")
    print(cand[['cell', 'n_train', 'resid_train', 'n_test', 'resid_test',
                'same_sign', 'p_test', 'p_bh', 'survivor', 'roi_test']].to_string(index=False))
    surv = cand[cand['survivor']]
    print("\nn_survivors:", len(surv))
    if len(surv):
        print(surv[['cell', 'resid_test', 'p_bh', 'roi_test']].to_string(index=False))
else:
    print("no candidate -> 0 survivors")

# Best cell by ROI_test among candidates (or all cells if none)
pool = cand if len(cand) else cells
best = pool.loc[pool['roi_test'].idxmax()]
print("\nbest cell by ROI_test in pool:", best['cell'], "| ROI_test:", round(best['roi_test'] * 100, 2), "%",
      "| resid_train:", round(best['resid_train'], 4), "| resid_test:", round(best['resid_test'], 4))
