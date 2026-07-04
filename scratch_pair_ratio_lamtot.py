import pandas as pd, numpy as np
from scipy import stats
from scipy.stats import binomtest

CSV = r'D:\AGENTOVA\SAMY\virtual-sports-scraper\data\vfoot_ml\odds_mine.csv'
d = pd.read_csv(CSV)
d['ts'] = pd.to_datetime(d['ts'])
d = d.sort_values('ts').reset_index(drop=True)

# Chrono split on median ts
med = d['ts'].median()
train = d[d['ts'] <= med].copy()
test  = d[d['ts'] >  med].copy()
print(f"Total={len(d)}  TRAIN={len(train)}  TEST={len(test)}  median_ts={med}")

# --- Build quantile bins for odds_ratio and lam_tot on TRAIN, apply to both ---
NQ = 10
def qbins(col):
    # quantile edges from TRAIN
    edges = np.unique(np.quantile(train[col], np.linspace(0,1,NQ+1)))
    edges[0]  = -np.inf
    edges[-1] =  np.inf
    return edges

eb_ratio = qbins('odds_ratio')
eb_lam   = qbins('lam_tot')

for df in (train, test):
    df['b_ratio'] = pd.cut(df['odds_ratio'], bins=eb_ratio, labels=False, include_lowest=True)
    df['b_lam']   = pd.cut(df['lam_tot'],   bins=eb_lam,   labels=False, include_lowest=True)
    df['cell'] = df['b_ratio'].astype(str) + '|' + df['b_lam'].astype(str)

print(f"ratio edges (n={len(eb_ratio)-1}): {np.round(eb_ratio[1:-1],4)}")
print(f"lam_tot edges (n={len(eb_lam)-1}): {np.round(eb_lam[1:-1],4)}")
print(f"distinct cells TRAIN: {train['cell'].nunique()}")

# --- Outcomes: issue vs implicite, offered odds ---
outcomes = {
    'home_win': ('imp_h','oh'),
    'draw':     ('imp_d','od'),
    'away_win': ('imp_a','oa'),
}

N_MIN = 150
RESID_MIN = 0.02

# Step 1: gather candidate cells on TRAIN per outcome, then test ALL of them on TEST
# Collect all TEST p-values across every tested (cell,outcome) for global BH-FDR.
candidates = []  # dicts with train info per outcome+cell
for issue,(imp,odd) in outcomes.items():
    g = train.groupby('cell')
    agg = g.agg(n_train=('cell','size'),
                mean_issue=(issue,'mean'),
                mean_imp=(imp,'mean'))
    agg = agg.reset_index()
    keep = agg[(agg['n_train']>=N_MIN) & ((agg['mean_issue']-agg['mean_imp']).abs()>=RESID_MIN)].copy()
    keep['resid_train'] = keep['mean_issue'] - keep['mean_imp']
    keep['issue'] = issue
    candidates.append(keep)

cand = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
print(f"\nKept candidate cells (TRAIN gate n>=150 & |resid|>=0.02): {len(cand)}")
print(cand.groupby('issue').size().to_dict() if len(cand) else "none")

# Step 2: test each candidate on TEST
rows = []
for _,r in cand.iterrows():
    issue = r['issue']; imp,odd = outcomes[issue]
    cell = r['cell']
    sub = test[test['cell']==cell]
    n = len(sub)
    if n == 0:
        continue
    succ = int(sub[issue].sum())
    mean_issue_test = sub[issue].mean()
    p_imp = sub[imp].mean()  # implicite moyen sur TEST
    resid_test = mean_issue_test - p_imp
    # binomtest against implied prob
    p_imp_clip = min(max(p_imp,1e-9),1-1e-9)
    bt = binomtest(succ, n, p_imp_clip)
    pval = bt.pvalue
    roi = (sub[issue]*sub[odd] - 1).mean()
    rows.append(dict(issue=issue, cell=cell, n_train=int(r['n_train']),
                     resid_train=r['resid_train'], n_test=n, succ=succ,
                     resid_test=resid_test, p_imp_test=p_imp,
                     pval=pval, roi_test=roi))

res = pd.DataFrame(rows)
print(f"\nCandidates with TEST data: {len(res)}")

if len(res):
    # BH-FDR over ALL tested cells (across all outcomes)
    pv = res['pval'].values
    order = np.argsort(pv)
    m = len(pv)
    bh = np.empty(m)
    # BH adjusted p-values
    ranked = pv[order]
    adj = ranked * m / (np.arange(1,m+1))
    # enforce monotonic
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    bh_sorted = np.minimum(adj,1.0)
    res['p_bh'] = np.nan
    res.iloc[order, res.columns.get_loc('p_bh')] = bh_sorted

    res['same_sign'] = np.sign(res['resid_train']) == np.sign(res['resid_test'])
    res['survivor'] = res['same_sign'] & (res['p_bh'] < 0.05)
    res['monetizable'] = res['survivor'] & (res['roi_test'] > 0)

    res_sorted = res.sort_values('pval')
    pd.set_option('display.width',200); pd.set_option('display.max_columns',30)
    print("\n=== ALL TESTED CANDIDATES (sorted by raw pval) ===")
    print(res_sorted[['issue','cell','n_train','resid_train','n_test','succ',
                      'resid_test','pval','p_bh','same_sign','survivor','roi_test','monetizable']].round(4).to_string(index=False))

    surv = res[res['survivor']]
    mon  = res[res['monetizable']]
    print(f"\nN cells tested (TRAIN-gated, with TEST data): {len(res)}")
    print(f"Survivors (same-sign + BH-FDR<0.05): {len(surv)}")
    print(f"Monetizable survivors (ROI_test>0): {len(mon)}")
    if len(surv):
        print("\nSURVIVORS:")
        print(surv[['issue','cell','resid_train','resid_test','p_bh','roi_test']].round(4).to_string(index=False))
    if len(mon):
        best = mon.sort_values('roi_test',ascending=False).iloc[0]
        print(f"\nBEST monetizable: {best['issue']} cell={best['cell']} ROI={best['roi_test']:.4f}")
    else:
        # best ROI among survivors (even if <=0) for reporting
        if len(surv):
            b = surv.sort_values('roi_test',ascending=False).iloc[0]
            print(f"\nBest survivor ROI (not >0): {b['issue']} cell={b['cell']} ROI={b['roi_test']:.4f}")
else:
    print("No candidates with TEST data.")
