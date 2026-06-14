# -*- coding: utf-8 -*-
"""WF3 - walk-forward HT/FT comebacks (2/1 et 1/2) + X/2."""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np, pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)
Q = """
SELECT e.id, e.team_a, e.team_b, e.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m ON m.mid = o.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
with eng.connect() as c:
    df = pd.read_sql(text(Q), c)
df = df.drop_duplicates(subset=['team_a','team_b','expected_start']).sort_values('expected_start').reset_index(drop=True)

def sgn(a,b): return '1' if a>b else ('2' if b>a else 'X')
rows = []
for _, row in df.iterrows():
    em = row['extra_markets']
    if em is None: continue
    em = json.loads(em) if isinstance(em, str) else em
    ht = em.get('HT/FT')
    ha = row['ht_score_a']
    if ht is None or ha is None or (isinstance(ha, float) and math.isnan(ha)): continue
    res = f"{sgn(int(row['ht_score_a']), int(row['ht_score_b']))}/{sgn(int(row['score_a']), int(row['score_b']))}"
    r = {'ts': row['expected_start'], 'res': res}
    for k, o in ht.items():
        r[k] = float(o)
    rows.append(r)
d = pd.DataFrame(rows)
cut = int(len(d)*0.7)
tr, te = d.iloc[:cut], d.iloc[cut:]
print(f"events: {len(d)} train={len(tr)} OOS={len(te)}")
for key in ['2/1','1/2','X/2','2/X','1/X']:
    for nm, dd in [('train', tr), ('OOS  ', te)]:
        sub = dd[dd[key].notna() & (dd[key] < 99.99)]
        w = (sub['res'] == key).astype(int)
        roi = (w*sub[key]).mean()-1
        print(f"{key} {nm}: n={len(sub)} freq={w.mean():.4f} avg_odds={sub[key].mean():.2f} ROI={roi:+.4f}")
    print()
