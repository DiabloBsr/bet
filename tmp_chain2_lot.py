import pandas as pd, numpy as np
from scipy import stats

CSV = 'D:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv'
d = pd.read_csv(CSV)
print('total rows:', len(d))

# lot chain2_results: need p1_result and p2_result + current odds/imp/win
d = d.dropna(subset=['p1_result', 'p2_result', 'odds', 'imp', 'win']).copy()
print('rows with 2-match history:', len(d))

# chrono split on median ts
d['ts'] = pd.to_datetime(d['ts'])
d = d.sort_values('ts')
med = d['ts'].median()
train = d[d['ts'] <= med].copy()
test = d[d['ts'] > med].copy()
print('median ts:', med, '| n_train:', len(train), '| n_test:', len(test))

# 3 odds bands: terciles fitted on TRAIN, applied to both
q1, q2 = train['odds'].quantile([1/3, 2/3]).values
print('odds band edges (train terciles): <=%.3f / <=%.3f / >%.3f' % (q1, q2, q2))
def band(o):
    return np.where(o <= q1, 'B1_low', np.where(o <= q2, 'B2_mid', 'B3_high'))
train['band'] = band(train['odds'].values)
test['band'] = band(test['odds'].values)
train['combo'] = train['p2_result'].astype(str) + train['p1_result'].astype(str)
test['combo'] = test['p2_result'].astype(str) + test['p1_result'].astype(str)

combos = ['WW','WD','WL','DW','DD','DL','LW','LD','LL']
bands = ['B1_low','B2_mid','B3_high']

rows = []
n_formed = 0
for c in combos:
    for b in bands:
        n_formed += 1
        tr = train[(train['combo'] == c) & (train['band'] == b)]
        n_tr = len(tr)
        if n_tr == 0:
            continue
        resid_tr = tr['win'].mean() - tr['imp'].mean()
        keep = (n_tr >= 150) and (abs(resid_tr) >= 0.02)
        te = test[(test['combo'] == c) & (test['band'] == b)]
        rows.append(dict(cell=f'{c}|{b}', n_train=n_tr, resid_train=resid_tr,
                         train_pass=keep, n_test=len(te),
                         k_test=int(te['win'].sum()) if len(te) else 0,
                         p0_test=te['imp'].mean() if len(te) else np.nan,
                         resid_test=(te['win'].mean() - te['imp'].mean()) if len(te) else np.nan,
                         roi_test=(te['win'] * te['odds'] - 1).mean() if len(te) else np.nan))

df = pd.DataFrame(rows)
print('\ncells formed:', n_formed, '| cells with train data:', len(df))
print('\n--- ALL cells (train) ---')
print(df[['cell','n_train','resid_train','train_pass']].to_string(index=False))

cand = df[df['train_pass']].copy()
print('\ncells passing TRAIN filter (n>=150 & |resid|>=0.02):', len(cand))

if len(cand):
    pvals = []
    for _, r in cand.iterrows():
        if r['n_test'] > 0 and not np.isnan(r['p0_test']):
            pv = stats.binomtest(r['k_test'], int(r['n_test']), float(np.clip(r['p0_test'], 1e-9, 1-1e-9))).pvalue
        else:
            pv = 1.0
        pvals.append(pv)
    cand['pval'] = pvals
    # BH-FDR over ALL tested cells of the lot
    m = len(cand)
    cand = cand.sort_values('pval').reset_index(drop=True)
    cand['rank'] = np.arange(1, m + 1)
    cand['bh_thresh'] = cand['rank'] / m * 0.05
    below = cand['pval'] <= cand['bh_thresh']
    kmax = below[below].index.max() if below.any() else -1
    cand['bh_reject'] = cand.index <= kmax
    cand['same_sign'] = np.sign(cand['resid_train']) == np.sign(cand['resid_test'])
    cand['survivor'] = cand['bh_reject'] & cand['same_sign']
    print('\n--- TEST (OOS) on train-passing cells ---')
    print(cand[['cell','n_train','resid_train','n_test','resid_test','pval','bh_thresh','bh_reject','same_sign','survivor','roi_test']].to_string(index=False))
    surv = cand[cand['survivor']]
    print('\nSURVIVORS (same sign + BH-FDR):', len(surv))
    if len(surv):
        print(surv[['cell','n_test','resid_train','resid_test','pval','roi_test']].to_string(index=False))
    # best candidate = lowest p-value among same-sign cells, else lowest overall
    ss = cand[cand['same_sign']]
    best = (ss if len(ss) else cand).sort_values('pval').iloc[0]
    print('\nBEST CELL:', best['cell'], '| pval=%.4f resid_tr=%.4f resid_te=%.4f n_te=%d roi_test=%.4f'
          % (best['pval'], best['resid_train'], best['resid_test'], best['n_test'], best['roi_test']))
else:
    print('No cell passed the TRAIN filter.')
