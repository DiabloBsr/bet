# -*- coding: utf-8 -*-
import sys, json, math
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine
eng = create_engine(load_settings().db_url)
Q = """
SELECT e.id AS event_id, e.team_a, e.team_b, e.expected_start, os.id AS snap_id,
       os.extra_markets, r.score_a, r.score_b
FROM events e
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id = e.id
JOIN odds_snapshots os ON os.id = m.mid
JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
df = pd.read_sql(Q, eng).sort_values('snap_id').drop_duplicates(['team_a','team_b','expected_start'], keep='first')
df = df.sort_values('expected_start').reset_index(drop=True)
N = len(df); split = int(N*0.7)
rows = []
for i, em_raw in enumerate(df['extra_markets']):
    em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    o = em.get('Total de buts', {}).get('1')
    if o is None: continue
    tot = int(df['score_a'].iat[i] + df['score_b'].iat[i])
    rows.append((i < split, float(o), int(tot == 1)))
t1 = pd.DataFrame(rows, columns=['train','odds','won'])
for name, g in [('TRAIN', t1[t1['train']]), ('OOS', t1[~t1['train']]), ('ALL', t1)]:
    pf = g['won']*g['odds'] - 1
    k, n = int(g['won'].sum()), len(g)
    imp = (1/g['odds']).mean()
    bt = stats.binomtest(k, n, k_exp := imp if 0<imp<1 else 0.5)
    # test vs break-even prob mean(1/odds)
    print(f"{name}: n={n} wr={k/n*100:.2f}% cote_moy={g['odds'].mean():.2f} ROI={pf.mean()*100:+.2f}% "
          f"| mean(1/cote)={imp*100:.2f}% p_binom(wr=1/cote)={bt.pvalue:.3f}")
# CUSUM-ish: ROI per quintile of time
t1['q'] = pd.qcut(t1.index, 5, labels=False)
print(t1.assign(pf=t1['won']*t1['odds']-1).groupby('q')['pf'].agg(['mean','count']).to_string())
