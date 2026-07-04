import pandas as pd, numpy as np
from scipy import stats

d = pd.read_csv('D:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/odds_mine.csv')
print("rows total:", len(d))
print("cols:", list(d.columns))

# Chrono split on median ts
d['ts'] = pd.to_datetime(d['ts']); d = d.sort_values('ts').reset_index(drop=True)
med = d['ts'].median()
train = d[d['ts'] <= med].copy()
test  = d[d['ts'] >  med].copy()
print("median ts:", med)
print("train n:", len(train), "test n:", len(test))

# LOT single_impd: imp_d seule en 10 quantiles
# Issue = draw vs imp_d
ISSUE = 'draw'
IMP   = 'imp_d'
ODDS  = 'od'

# Build quantile bins from TRAIN edges (no leakage)
nq = 10
# qcut edges on train
edges = np.quantile(train[IMP].values, np.linspace(0,1,nq+1))
edges = np.unique(edges)
# ensure coverage
edges[0]  = -np.inf
edges[-1] =  np.inf
print("bin edges (train quantiles):", edges)

train['cell'] = pd.cut(train[IMP], bins=edges, labels=False, include_lowest=True)
test['cell']  = pd.cut(test[IMP],  bins=edges, labels=False, include_lowest=True)

n_cells_total = train['cell'].nunique()
print("n cells (bins) total:", n_cells_total)

# Step 3: TRAIN filter
kept = []
for c, g in train.groupby('cell'):
    n = len(g)
    if n < 150:
        continue
    resid = g[ISSUE].mean() - g[IMP].mean()
    if abs(resid) >= 0.02:
        kept.append((c, n, resid, g[IMP].mean()))

print("\n=== TRAIN kept cells (n>=150 & |resid|>=0.02) ===")
for c,n,r,imp in kept:
    print(f"cell={c} n_train={n} resid_train={r:+.4f} imp_mean={imp:.4f}")
print("kept count:", len(kept))

# Step 3 TEST: binomtest on each kept cell, collect pvals for BH-FDR
results = []
for c, n_tr, resid_tr, imp_tr in kept:
    gt = test[test['cell']==c]
    n  = len(gt)
    if n == 0:
        continue
    succ = int(gt[ISSUE].sum())
    imp_test_mean = gt[IMP].mean()
    resid_test = gt[ISSUE].mean() - imp_test_mean
    # binomtest against mean implicit prob in test cell
    bt = stats.binomtest(succ, n, imp_test_mean)
    p  = bt.pvalue
    same_sign = (np.sign(resid_test) == np.sign(resid_tr)) and resid_test != 0
    # ROI on test
    roi = (gt[ISSUE]*gt[ODDS] - 1).mean()
    results.append({
        'cell': c, 'n_test': n, 'succ': succ,
        'imp_test': imp_test_mean,
        'resid_train': resid_tr, 'resid_test': resid_test,
        'same_sign': same_sign, 'p': p, 'roi_test': roi
    })

print("\n=== TEST results on kept cells ===")
for r in results:
    print(f"cell={r['cell']} n_test={r['n_test']} resid_test={r['resid_test']:+.4f} "
          f"same_sign={r['same_sign']} p={r['p']:.4g} roi_test={r['roi_test']:+.4f}")

# BH-FDR over ALL tested cells
m = len(results)
if m > 0:
    pvals = np.array([r['p'] for r in results])
    order = np.argsort(pvals)
    ranked = pvals[order]
    # BH threshold
    alpha = 0.05
    bh = ranked <= (np.arange(1,m+1)/m)*alpha
    # largest k where condition holds
    if bh.any():
        kmax = np.max(np.where(bh)[0])
        crit = ranked[kmax]
    else:
        crit = -1
    for i,r in enumerate(results):
        r['fdr_sig'] = r['p'] <= crit
    print(f"\nBH-FDR: m={m} alpha={alpha} crit_pvalue={crit:.4g}")
else:
    crit = -1

survivors = [r for r in results if r['same_sign'] and r.get('fdr_sig', False)]
print("\n=== SURVIVORS (same sign + FDR sig) ===")
for r in survivors:
    print(f"cell={r['cell']} n_test={r['n_test']} resid_test={r['resid_test']:+.4f} "
          f"p={r['p']:.4g} roi_test={r['roi_test']:+.4f}")
print("n survivors:", len(survivors))

monetizable = [r for r in survivors if r['roi_test'] > 0]
print("\n=== MONETIZABLE survivors (ROI_test>0) ===")
for r in monetizable:
    print(f"cell={r['cell']} roi_test={r['roi_test']:+.4f}")
print("n monetizable:", len(monetizable))

# best cell summary
if monetizable:
    best = max(monetizable, key=lambda r: r['roi_test'])
elif survivors:
    best = max(survivors, key=lambda r: r['roi_test'])
else:
    best = None
print("\nBEST:", best)
print("\nN_CELLS_TESTED:", m)
