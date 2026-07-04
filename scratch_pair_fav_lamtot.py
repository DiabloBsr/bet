import pandas as pd, numpy as np
from scipy import stats

CSV = 'D:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/odds_mine.csv'
d = pd.read_csv(CSV)

# ---- chrono split on median ts ----
med = d['ts'].median()
train = d[d['ts'] <= med].copy()
test  = d[d['ts'] >  med].copy()
print(f"N total={len(d)}  N train={len(train)}  N test={len(test)}  med_ts={med}")

# ---- build bins for fav and lam_tot as ~10 quantiles (defined on TRAIN to avoid leakage) ----
def qbins(s, q=10):
    edges = np.unique(np.quantile(s, np.linspace(0,1,q+1)))
    edges[0]  = -np.inf
    edges[-1] =  np.inf
    return edges

fav_edges = qbins(train['fav'], 10)
lam_edges = qbins(train['lam_tot'], 10)
print("fav edges:", np.round(fav_edges,4))
print("lam_tot edges:", np.round(lam_edges,4))

for df in (train, test):
    df['fav_bin'] = pd.cut(df['fav'], bins=fav_edges, labels=False, include_lowest=True)
    df['lam_bin'] = pd.cut(df['lam_tot'], bins=lam_edges, labels=False, include_lowest=True)
    df['cell'] = df['fav_bin'].astype(str) + '_' + df['lam_bin'].astype(str)

# issues: (outcome col, implied prob col, offered odds col)
issues = [
    ('home_win','imp_h','oh'),
    ('draw',    'imp_d','od'),
    ('away_win','imp_a','oa'),
]

MIN_N_TRAIN = 150
MIN_RESID   = 0.02

# Collect candidate cells (per issue) from TRAIN
candidates = []  # (issue, cell, resid_train, n_train)
for outcome, imp, odd in issues:
    g = train.groupby('cell')
    agg = g.agg(n=('cell','size'),
                mean_out=(outcome,'mean'),
                mean_imp=(imp,'mean'))
    agg['resid'] = agg['mean_out'] - agg['mean_imp']
    sel = agg[(agg['n']>=MIN_N_TRAIN) & (agg['resid'].abs()>=MIN_RESID)]
    for cell, row in sel.iterrows():
        candidates.append((outcome, imp, odd, cell, row['resid'], int(row['n'])))

print(f"\nTotal candidate cells kept on TRAIN (n>=150 & |resid|>=0.02): {len(candidates)}")

# Now evaluate ALL candidates on TEST, gather p-values for BH-FDR across the whole tested set
rows = []
for outcome, imp, odd, cell, resid_tr, n_tr in candidates:
    sub = test[test['cell']==cell]
    n_te = len(sub)
    if n_te == 0:
        continue
    succ = int(sub[outcome].sum())
    p_imp = sub[imp].mean()           # mean implied prob in cell on TEST
    mean_out_te = sub[outcome].mean()
    resid_te = mean_out_te - p_imp
    # binomtest succ ~ n_te under p_imp
    bt = stats.binomtest(succ, n_te, p_imp, alternative='two-sided')
    pval = bt.pvalue
    roi = (sub[outcome]*sub[odd] - 1).mean()
    rows.append(dict(outcome=outcome, cell=cell, n_tr=n_tr, resid_tr=resid_tr,
                     n_te=n_te, succ=succ, p_imp=p_imp, resid_te=resid_te,
                     same_sign=(np.sign(resid_te)==np.sign(resid_tr)),
                     pval=pval, roi_te=roi))

res = pd.DataFrame(rows)
print(f"Candidate cells with TEST data: {len(res)}")

# BH-FDR across ALL tested cells
res = res.sort_values('pval').reset_index(drop=True)
m = len(res)
alpha = 0.05
res['rank'] = np.arange(1, m+1)
res['bh_thresh'] = res['rank']/m*alpha
# BH: largest rank where pval <= bh_thresh; all with rank <= that are rejected
below = res[res['pval'] <= res['bh_thresh']]
if len(below):
    kmax = below['rank'].max()
else:
    kmax = 0
res['fdr_sig'] = res['rank'] <= kmax

# SURVIVOR = same_sign AND fdr_sig
res['survivor'] = res['same_sign'] & res['fdr_sig']
# MONETIZABLE survivor = survivor AND roi_te>0
res['monetizable'] = res['survivor'] & (res['roi_te']>0)

n_tested = m
n_surv = int(res['survivor'].sum())
n_money = int(res['monetizable'].sum())
print(f"\nn_cells_tested (OOS, with FDR) = {n_tested}")
print(f"same_sign cells = {int(res['same_sign'].sum())}")
print(f"fdr_sig cells   = {int(res['fdr_sig'].sum())}")
print(f"SURVIVORS (same_sign & FDR) = {n_surv}")
print(f"MONETIZABLE survivors (ROI_test>0) = {n_money}")

pd.set_option('display.width',200); pd.set_option('display.max_columns',30)
print("\n--- Top 15 by p-value ---")
print(res.head(15)[['outcome','cell','n_tr','resid_tr','n_te','succ','p_imp','resid_te','same_sign','pval','bh_thresh','fdr_sig','survivor','roi_te','monetizable']].round(4).to_string())

if n_surv:
    print("\n--- SURVIVORS ---")
    print(res[res['survivor']].round(4).to_string())
else:
    print("\nNo survivors.")

# Best cell by ROI among survivors (else among all tested)
if n_money:
    best = res[res['monetizable']].sort_values('roi_te',ascending=False).iloc[0]
elif n_surv:
    best = res[res['survivor']].sort_values('roi_te',ascending=False).iloc[0]
else:
    best = res.sort_values('roi_te',ascending=False).iloc[0]
print("\nBEST cell:", best['outcome'], best['cell'], "ROI_te=", round(best['roi_te'],4),
      "survivor=", bool(best['survivor']), "n_te=", int(best['n_te']))
