# -*- coding: utf-8 -*-
"""
WF4 — team totals & micro-markets — step 2: systematic scan.

Protocol:
  - bets from exports/_wf4_teamtotals_bets.pkl (opening odds, settled).
  - DISCOVERY set  = 8035 train (first 70% by expected_start).
  - VALIDATION 1   = 8035 test (last 30%).
  - VALIDATION 2   = pooled 8 new leagues.
  - Scan axes: (market, selection) x odds-bucket x context (favorite profile
    from opening 1X2). Only cells with avg odd >= 4 are candidate findings
    (mission scope), but full calibration table is exported for all.
  - Stats: n, wins, win rate, avg odd, flat 1u ROI, z & one-sided p-value
    of wins vs sum of implied probs (positive-EV test).

Outputs: exports/wf4_teamtotals_scan.json
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_pickle('exports/_wf4_teamtotals_bets.pkl')
df['ts'] = pd.to_datetime(df['expected_start'])

# exclude markets owned by other miners from the candidate scan (kept in calib)
OTHER_MINERS = {'HT/FT', 'FTTS', 'Score exact'}

# --- splits ---
e35 = df[df.league == '8035']
ev_order = e35[['event_id', 'ts']].drop_duplicates().sort_values('ts')
cut = ev_order.iloc[int(len(ev_order) * 0.7)].ts
print(f"8035 events: {len(ev_order)}, 70% cut at {cut}")
train = e35[e35.ts < cut]
test = e35[e35.ts >= cut]
newl = df[df.league != '8035']
print(f"bets: train={len(train)} test={len(test)} newleagues={len(newl)}")

ODDS_BUCKETS = [(1.0, 1.3), (1.3, 1.6), (1.6, 2.0), (2.0, 3.0), (3.0, 4.0),
                (4.0, 5.0), (5.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 12.0),
                (12.0, 15.0), (15.0, 20.0), (20.0, 30.0), (30.0, 50.0), (50.0, 100.0)]


def fav_profile(oh, oa):
    if oh is None or oa is None or not np.isfinite(oh) or not np.isfinite(oa):
        return 'na'
    if oh < 1.6:
        return 'home_strong'
    if 1.6 <= oh < 2.2 and oa >= 2.5:
        return 'home_slight'
    if oa < 1.6:
        return 'away_strong'
    if 1.6 <= oa < 2.2 and oh >= 2.5:
        return 'away_slight'
    return 'balanced'


df_all_ctx = {}
for name, d in (('train', train), ('test', test), ('newl', newl)):
    d = d.copy()
    d['ctx'] = [fav_profile(h, a) for h, a in zip(d.oh, d.oa)]
    df_all_ctx[name] = d


def cell_stats(s):
    n = len(s)
    if n == 0:
        return None
    wins = int(s.won.sum())
    p0 = (1.0 / s.odd).values  # implied probs at offered odds (incl. margin)
    mu, var = p0.sum(), (p0 * (1 - p0)).sum()
    z = (wins - mu) / np.sqrt(var) if var > 0 else 0.0
    pval = float(1 - stats.norm.cdf(z))  # one-sided: more wins than implied
    roi = float((s.won * s.odd - 1).mean())
    return dict(n=n, wins=wins, wr=round(wins / n, 4), avg_odd=round(float(s.odd.mean()), 3),
                roi=round(roi * 100, 2), z=round(float(z), 3), p=round(pval, 6))


results = {'tests_scanned': 0, 'cells': []}
mk_sels = df[['market', 'sel']].drop_duplicates().values.tolist()
print(f"(market,sel) pairs: {len(mk_sels)}")

for mk, sel in mk_sels:
    for lo, hi in ODDS_BUCKETS:
        for ctx in ('ALL', 'home_strong', 'home_slight', 'balanced', 'away_slight', 'away_strong'):
            tr = df_all_ctx['train']
            m = (tr.market == mk) & (tr.sel == sel) & (tr.odd >= lo) & (tr.odd < hi)
            if ctx != 'ALL':
                m &= (tr.ctx == ctx)
            s = tr[m]
            results['tests_scanned'] += 1
            st = cell_stats(s)
            if st is None or st['n'] < 80:
                continue
            rec = dict(market=mk, sel=sel, lo=lo, hi=hi, ctx=ctx, train=st)
            # always attach validations for cells that look promising on train
            if st['roi'] >= 4.0 and st['p'] <= 0.05 and st['n'] >= 80:
                for vname in ('test', 'newl'):
                    v = df_all_ctx[vname]
                    mv = (v.market == mk) & (v.sel == sel) & (v.odd >= lo) & (v.odd < hi)
                    if ctx != 'ALL':
                        mv &= (v.ctx == ctx)
                    rec[vname] = cell_stats(v[mv])
            results['cells'].append(rec)

print(f"tests scanned: {results['tests_scanned']}")
cands = [c for c in results['cells'] if 'test' in c]
print(f"train-promising cells: {len(cands)}")
cands.sort(key=lambda c: c['train']['p'])
for c in cands[:60]:
    mkr = '*' if c['market'] in OTHER_MINERS else ' '
    line = (f"{mkr}{c['market']!r:42s} {c['sel']!r:30s} [{c['lo']:>4},{c['hi']:>4}) {c['ctx']:12s} "
            f"TRAIN n={c['train']['n']:5d} roi={c['train']['roi']:+7.2f}% p={c['train']['p']:.4f}")
    t, nl = c.get('test'), c.get('newl')
    if t:
        line += f" | TEST n={t['n']:4d} roi={t['roi']:+7.2f}% p={t['p']:.4f}"
    else:
        line += " | TEST none"
    if nl:
        line += f" | NEW n={nl['n']:5d} roi={nl['roi']:+7.2f}% p={nl['p']:.4f}"
    else:
        line += " | NEW none"
    print(line)

with open('exports/wf4_teamtotals_scan.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)
print("saved exports/wf4_teamtotals_scan.json")
