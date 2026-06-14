# -*- coding: utf-8 -*-
"""
WF4 — team totals & micro-markets — step 3: vectorized conditional scan.

Same protocol as step 2 but vectorized (groupby) so the full
(market, sel, odds-bucket, fav-context) grid is scanned in minutes.

Discovery on 8035-train (70%), validation on 8035-test (30%) AND pooled
8 new leagues. Candidate threshold on train: n>=80, ROI>=+4%, p<=0.05,
avg_odd>=3.5. A finding must then hold on BOTH validation sets
(ROI>0 on each) or be reported with the appropriate reduced scope.

Output: exports/wf4_teamtotals_scan_v.json
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_pickle('exports/_wf4_teamtotals_bets.pkl')
df['ts'] = pd.to_datetime(df['expected_start'])
e35 = df[df.league == '8035']
ev = e35[['event_id', 'ts']].drop_duplicates().sort_values('ts')
cut = ev.iloc[int(len(ev) * 0.7)].ts
df['split'] = np.where(df.league != '8035', 'newl',
                       np.where(df.ts < cut, 'train', 'test'))

oh, oa = df.oh.values, df.oa.values
ctx = np.full(len(df), 'balanced', dtype=object)
ctx[(oh < 1.6)] = 'home_strong'
ctx[(oh >= 1.6) & (oh < 2.2) & (oa >= 2.5)] = 'home_slight'
ctx[(oa < 1.6)] = 'away_strong'
ctx[(oa >= 1.6) & (oa < 2.2) & (oh >= 2.5)] = 'away_slight'
df['ctx'] = ctx

bins = [1, 1.3, 1.6, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 50, 100]
df['ob'] = pd.cut(df.odd, bins, right=False)


def agg(s):
    p0 = 1 / s['odd']
    mu, var = p0.sum(), (p0 * (1 - p0)).sum()
    z = (s['won'].sum() - mu) / np.sqrt(var) if var > 0 else 0.0
    return pd.Series({'n': len(s), 'wins': s['won'].sum(),
                      'wr': s['won'].mean(), 'avg_odd': s['odd'].mean(),
                      'roi': (s['won'] * s['odd'] - 1).mean() * 100,
                      'z': z, 'p': 1 - stats.norm.cdf(z)})


frames = []
# ctx-conditioned grid + ALL
for label, d in (('ctx', df), ):
    g1 = df.groupby(['market', 'sel', 'ob', 'ctx', 'split'], observed=True).apply(agg, include_groups=False)
    g1 = g1.unstack('split')
    g2 = df.groupby(['market', 'sel', 'ob', 'split'], observed=True).apply(agg, include_groups=False)
    g2 = g2.unstack('split')
    g2.index = pd.MultiIndex.from_tuples([(m, s, b, 'ALL') for m, s, b in g2.index],
                                         names=['market', 'sel', 'ob', 'ctx'])
    frames = [g1, g2]
g = pd.concat(frames)

n_tests = int((g[('n', 'train')].fillna(0) > 0).sum())
print(f"non-empty train cells scanned: {n_tests}  (grid rows incl. empty: {len(g)})")

cand = g[(g[('roi', 'train')] >= 4) & (g[('p', 'train')] <= 0.05)
         & (g[('n', 'train')] >= 80) & (g[('avg_odd', 'train')] >= 3.5)].copy()
# validation: positive on both holdouts
cand['valid'] = ((cand[('roi', 'test')].fillna(-100) > 0)
                 & (cand[('roi', 'newl')].fillna(-100) > 0))
pd.set_option('display.width', 300)
cols = [('n', 'train'), ('roi', 'train'), ('p', 'train'),
        ('n', 'test'), ('roi', 'test'), ('p', 'test'),
        ('n', 'newl'), ('roi', 'newl'), ('p', 'newl'), ('valid', '')]
cand = cand.sort_values(('p', 'train'))
print(f"train candidates: {len(cand)}   surviving both holdouts: {int(cand['valid'].sum())}")
print(cand[cols].round(3).to_string())

out = {'tests_scanned': n_tests, 'n_candidates': int(len(cand)),
       'n_validated': int(cand['valid'].sum()),
       'candidates': []}
for idx, row in cand.iterrows():
    rec = {'market': idx[0], 'sel': idx[1], 'bucket': str(idx[2]), 'ctx': idx[3], 'valid': bool(row['valid'].iloc[0]) if hasattr(row['valid'], 'iloc') else bool(row['valid'])}
    for sp in ('train', 'test', 'newl'):
        try:
            rec[sp] = {k: (None if pd.isna(row[(k, sp)]) else round(float(row[(k, sp)]), 4))
                       for k in ('n', 'wins', 'wr', 'avg_odd', 'roi', 'z', 'p')}
        except KeyError:
            rec[sp] = None
    out['candidates'].append(rec)
with open('exports/wf4_teamtotals_scan_v.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved exports/wf4_teamtotals_scan_v.json")
