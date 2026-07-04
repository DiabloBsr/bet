import pandas as pd, numpy as np
from scipy import stats

CSV = r'D:\AGENTOVA\SAMY\virtual-sports-scraper\data\vfoot_ml\odds_mine.csv'
d = pd.read_csv(CSV)

# ---- Chronological split on median ts ----
d['ts'] = pd.to_datetime(d['ts'])
d = d.sort_values('ts').reset_index(drop=True)
med = d['ts'].median()
train = d[d['ts'] <= med].copy()
test  = d[d['ts'] >  med].copy()
print(f"N total={len(d)}  median ts={med}")
print(f"N train={len(train)}  N test={len(test)}")

# ---- Binning: derived features fav, lam_tot, lam_diff -> 6 bins each (quantile-based, edges from TRAIN) ----
NBINS = 6
feats = ['fav', 'lam_tot', 'lam_diff']
edges = {}
for f in feats:
    # quantile edges computed on TRAIN, applied to both
    qs = np.linspace(0, 1, NBINS + 1)
    e = np.quantile(train[f].values, qs)
    e[0] = -np.inf
    e[-1] = np.inf
    # ensure strictly increasing
    e = np.unique(e)
    edges[f] = e

def assign_bins(df):
    df = df.copy()
    for f in feats:
        df[f + '_b'] = pd.cut(df[f], bins=edges[f], labels=False, include_lowest=True)
    df['cell'] = (df['fav_b'].astype(str) + '|' + df['lam_tot_b'].astype(str) + '|' + df['lam_diff_b'].astype(str))
    return df

train = assign_bins(train)
test  = assign_bins(test)

# ---- Three outcomes vs implied probabilities ----
# (issue_col, implied_col, odds_col)
outcomes = [
    ('home_win', 'imp_h', 'oh'),
    ('draw',     'imp_d', 'od'),
    ('away_win', 'imp_a', 'oa'),
]

N_TRAIN_MIN = 150
RESID_MIN   = 0.02

def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns boolean array of rejections."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresh
    if not below.any():
        return np.zeros(n, dtype=bool)
    kmax = np.max(np.where(below)[0])
    crit = ranked[kmax]
    rej = np.zeros(n, dtype=bool)
    rej[order[ranked <= crit]] = True
    return rej

# ---- Step 1: select candidate cells on TRAIN (per outcome) ----
candidates = []   # (outcome, cell, resid_train, n_train, imp_mean_train)
for issue, imp, odds in outcomes:
    g = train.groupby('cell')
    agg = g.agg(n=('cell', 'size'),
                mean_issue=(issue, 'mean'),
                mean_imp=(imp, 'mean'))
    agg = agg[agg['n'] >= N_TRAIN_MIN]
    agg['resid'] = agg['mean_issue'] - agg['mean_imp']
    sel = agg[agg['resid'].abs() >= RESID_MIN]
    for cell, row in sel.iterrows():
        candidates.append({
            'outcome': issue, 'imp': imp, 'odds': odds, 'cell': cell,
            'resid_train': row['resid'], 'n_train': int(row['n']),
            'imp_train': row['mean_imp'],
        })

print(f"\nTotal cells in TRAIN grid (per outcome avg): {train['cell'].nunique()}")
print(f"Candidate cells passing TRAIN filter (n>=150 & |resid|>=0.02): {len(candidates)}")

# ---- Step 2: OOS test on these candidates ----
# collect p-values for BH-FDR across all tested cells
rows = []
for c in candidates:
    issue, imp, odds, cell = c['outcome'], c['imp'], c['odds'], c['cell']
    sub = test[test['cell'] == cell]
    n_test = len(sub)
    if n_test == 0:
        continue
    succ = int(sub[issue].sum())
    imp_mean_test = sub[imp].mean()
    resid_test = sub[issue].mean() - imp_mean_test
    # binomtest against the mean implied prob in test cell
    p_imp = float(np.clip(imp_mean_test, 1e-9, 1 - 1e-9))
    bt = stats.binomtest(succ, n_test, p_imp, alternative='two-sided')
    pval = bt.pvalue
    same_sign = np.sign(resid_test) == np.sign(c['resid_train'])
    # ROI on TEST using offered odds
    roi_test = float((sub[issue] * sub[odds] - 1).mean())
    rows.append({
        **c,
        'n_test': n_test, 'succ': succ, 'imp_test': imp_mean_test,
        'resid_test': resid_test, 'pval': pval, 'same_sign': bool(same_sign),
        'roi_test': roi_test,
    })

res = pd.DataFrame(rows)
print(f"Candidate cells with n_test>0 (= cells actually OOS-tested): {len(res)}")

n_cells_tested = len(res)

if n_cells_tested == 0:
    print("\nNo testable cells.")
else:
    # BH-FDR across ALL tested cells
    res['fdr_reject'] = bh_fdr(res['pval'].values, alpha=0.05)
    # Survivor = same sign AND FDR-significant
    res['survivor'] = res['same_sign'] & res['fdr_reject']
    survivors = res[res['survivor']].copy()
    print(f"\nSurvivors (same sign + BH-FDR significant): {len(survivors)}")
    # monetizable = survivor AND roi_test > 0
    monet = survivors[survivors['roi_test'] > 0].copy()
    print(f"Monetizable survivors (ROI_test > 0): {len(monet)}")

    # show detail
    cols = ['outcome', 'cell', 'n_train', 'resid_train', 'n_test',
            'resid_test', 'pval', 'same_sign', 'fdr_reject', 'roi_test']
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    if len(survivors):
        print("\n--- SURVIVORS ---")
        print(survivors[cols].to_string(index=False))
    if len(monet):
        print("\n--- MONETIZABLE (ROI>0) ---")
        print(monet[cols].to_string(index=False))
        best = monet.sort_values('roi_test', ascending=False).iloc[0]
        print(f"\nBEST monetizable cell: {best['outcome']} cell={best['cell']} ROI_test={best['roi_test']*100:.2f}%")
    # Best ROI among all tested (context, not necessarily survivor)
    bestany = res.sort_values('roi_test', ascending=False).iloc[0]
    print(f"\nBest ROI_test among ALL tested cells (context): {bestany['outcome']} cell={bestany['cell']} ROI={bestany['roi_test']*100:.2f}% survivor={bestany['survivor']}")
    print(f"min pval among tested: {res['pval'].min():.4g}")
