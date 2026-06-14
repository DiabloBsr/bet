# -*- coding: utf-8 -*-
"""
Table finale consolidee — protocole 70/30 strict :
signaux equipe calcules sur le train (70% premiers matchs tries par expected_start),
evaluation unique sur l'OOS (30% derniers). Memes definitions que _wf_value_buckets3.py.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

pd.set_option('display.width', 260)

eng = create_engine(load_settings().db_url)
df = pd.read_sql("""
SELECT e.id AS event_id, CAST(e.round_info AS INTEGER) AS round, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, r.score_a, r.score_b
FROM events e JOIN results r ON r.event_id=e.id
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots os WHERE os.event_id=e.id)
WHERE e.round_info != '0' AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL
  AND o.odds_away IS NOT NULL AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
ORDER BY e.expected_start, e.id""", eng)
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values(['expected_start', 'event_id']).reset_index(drop=True)

def segment(r):
    if r <= 3: return 'DS'
    if r <= 12: return 'MS_early'
    if r <= 25: return 'MS_mid'
    if r <= 33: return 'MS_late'
    return 'FS'
df['segment'] = df['round'].apply(segment)
df['outcome'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
df['home_win'] = (df.outcome == 'H').astype(int)
df['away_win'] = (df.outcome == 'A').astype(int)

prev_home = {}
vals = []
for _, r in df.iterrows():
    vals.append(prev_home.get(r.team_a))
    prev_home[r.team_a] = 'W' if r.outcome == 'H' else ('D' if r.outcome == 'D' else 'L')
df['home_prev_home_res'] = vals

cut = int(len(df) * 0.70)
train, oos = df.iloc[:cut], df.iloc[cut:]

th = train.groupby('team_a').agg(hw=('home_win', 'mean'), hn=('home_win', 'size'))
ta = train.groupby('team_b').agg(aw=('away_win', 'mean'), an=('away_win', 'size'))
ths = train.groupby(['team_a', 'segment']).agg(hw=('home_win', 'mean'), hn=('home_win', 'size'))
def hsegd(t, s):
    try:
        row = ths.loc[(t, s)]
        return row.hw - th.hw.get(t, np.nan) if row.hn >= 6 else np.nan
    except KeyError:
        return np.nan

def add_signals(d):
    d = d.copy()
    d['ta_home_wr'] = d.team_a.map(lambda t: th.hw.get(t, np.nan))
    d['ta_hseg_delta'] = d.apply(lambda r: hsegd(r.team_a, r.segment), axis=1)
    d['edge_home'] = d['ta_home_wr'] - 1.0 / d['odds_home']
    return d

trS, ooS = add_signals(train), add_signals(oos)

# golden pairs (negative control)
pgH = train.groupby(['team_a', 'team_b']).agg(w=('home_win', 'mean'), n=('home_win', 'size'))
goldH = set(pgH[(pgH.w >= 0.70) & (pgH.n >= 6)].index)

def ev(sub, side):
    if len(sub) == 0: return 0, np.nan, np.nan, np.nan
    odds = {'H': sub.odds_home, 'A': sub.odds_away, 'D': sub.odds_draw}[side]
    won = (sub.outcome == side).astype(float)
    return len(sub), won.mean(), odds.mean(), (won * (odds - 1) - (1 - won)).mean()

FINAL = [
    ("F1 HOME edge>=0.10 cote[2.5,3.5)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.10)),
    ("F2 HOME edge>=0.10 cote[2.0,3.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 3.0) & (d.edge_home >= 0.10)),
    ("F3 HOME edge>=0.10 cote[2.0,3.5) pooled", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 3.5) & (d.edge_home >= 0.10)),
    ("F4 HOME edge>=0.08 cote[2.5,3.5)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.08)),
    ("F5 HOME edge>=0.10 + prevHomeLoss cote[2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.10) & (d.home_prev_home_res == 'L')),
    ("F6 HOME segdelta>=0.10 cote[2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.ta_hseg_delta >= 0.10)),
    ("F7 HOME segdelta>=0.08 cote[2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.ta_hseg_delta >= 0.08)),
    ("F8 HOME edge>=0.10 cote[2.0,6.0) volume", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.10)),
    ("F9 HOME longshot cote[4.0,6.0) global", 'H', lambda d: (d.odds_home >= 4.0) & (d.odds_home < 6.0)),
    ("F10 HOME rebond prevHomeLoss cote[2.5,4.0)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 4.0) & (d.home_prev_home_res == 'L')),
    ("F11 AWAY favori cote<=1.40 (accuracy)", 'A', lambda d: d.odds_away <= 1.40),
    ("F12 NEGATIF golden pairs H wr>=70 n>=6", 'H', lambda d: d.apply(lambda r: (r.team_a, r.team_b) in goldH, axis=1)),
]
rows = []
for name, side, fn in FINAL:
    nt, wt, ot, rt = ev(trS[fn(trS)], side)
    no, wo, oo_, ro = ev(ooS[fn(ooS)], side)
    rows.append(dict(name=name, n_train=nt, wr_train=round(wt, 3), roi_train=round(rt, 3),
                     n_oos=no, wr_oos=round(wo, 3), avg_cote_oos=round(oo_, 3), roi_oos=round(ro, 3)))
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv('scripts/_wf_value_buckets_final.csv', index=False)
print("\nsaved scripts/_wf_value_buckets_final.csv")
