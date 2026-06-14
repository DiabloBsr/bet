# -*- coding: utf-8 -*-
"""
Walk-forward analysis of DRAWS (X) — Bet261 virtual football.
Anti-leakage: temporal split 70/30 sorted by expected_start.
All signals (pairs, team rates, thresholds) computed on TRAIN only, evaluated on OOS.
Metrics reported: n_oos, wr_oos, avg_cote (actual X odds), roi_oos = mean(won*(cote-1) - (1-won)).
"""
import sys, json
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 200)

eng = create_engine(load_settings().db_url)

SQL = """
SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
WHERE e.round_info != '0'
  AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
  AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
ORDER BY e.expected_start
"""

with eng.connect() as c:
    df = pd.read_sql(text(SQL), c)

df['round'] = pd.to_numeric(df['round_info'], errors='coerce')
df = df[df['round'].between(1, 38)].copy()
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)

def seg(r):
    if r <= 3: return 'DS'
    if r <= 12: return 'MS_early'
    if r <= 25: return 'MS_mid'
    if r <= 33: return 'MS_late'
    return 'FS'
df['segment'] = df['round'].apply(seg)

df['is_draw'] = (df['score_a'] == df['score_b']).astype(int)
df['is_ht_draw'] = np.where(df['ht_score_a'].notna() & df['ht_score_b'].notna(),
                            (df['ht_score_a'] == df['ht_score_b']).astype(float), np.nan)
df['odds_gap'] = (df['odds_home'] - df['odds_away']).abs()
df['fav_odds'] = df[['odds_home', 'odds_away']].min(axis=1)

# Parse HT X odds and HT/FT X/X odds from extra_markets
def parse_em(em):
    if em is None: return (np.nan, np.nan)
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: return (np.nan, np.nan)
    if not isinstance(em, dict): return (np.nan, np.nan)
    htx = np.nan
    m = em.get('Mi-tps 1X2')
    if isinstance(m, dict):
        v = m.get('X')
        if v: htx = float(v)
    htft_xx = np.nan
    m = em.get('HT/FT')
    if isinstance(m, dict):
        for k in ('X/X', 'XX', 'X-X', 'Nul/Nul'):
            if k in m and m[k]:
                htft_xx = float(m[k]); break
    return (htx, htft_xx)

parsed = df['extra_markets'].apply(parse_em)
df['ht_x_odds'] = [p[0] for p in parsed]
df['htft_xx_odds'] = [p[1] for p in parsed]

n = len(df)
cut = int(n * 0.70)
train = df.iloc[:cut].copy()
oos = df.iloc[cut:].copy()
print(f"TOTAL n={n}  train={len(train)} ({train['expected_start'].min()} -> {train['expected_start'].max()})")
print(f"OOS n={len(oos)} ({oos['expected_start'].min()} -> {oos['expected_start'].max()})")
print(f"Base draw rate: train={train['is_draw'].mean():.3f}  oos={oos['is_draw'].mean():.3f}")
print(f"Base HT draw rate: train={train['is_ht_draw'].mean():.3f}  oos={oos['is_ht_draw'].mean():.3f}")
print(f"X odds: mean={df['odds_draw'].mean():.2f}  HT-X odds mean={df['ht_x_odds'].mean():.2f}")

def evaluate(sub, odds_col='odds_draw', won_col='is_draw', label=''):
    """ROI of betting X on every row of sub at odds_col."""
    sub = sub.dropna(subset=[odds_col])
    if won_col == 'is_ht_draw':
        sub = sub.dropna(subset=[won_col])
    nn = len(sub)
    if nn == 0:
        return dict(n=0, wr=np.nan, avg_cote=np.nan, roi=np.nan)
    won = sub[won_col].astype(float)
    cote = sub[odds_col].astype(float)
    roi = (won * (cote - 1) - (1 - won)).mean()
    return dict(n=nn, wr=won.mean(), avg_cote=cote.mean(), roi=roi)

def report(name, train_sub, oos_sub, odds_col='odds_draw', won_col='is_draw'):
    t = evaluate(train_sub, odds_col, won_col)
    o = evaluate(oos_sub, odds_col, won_col)
    print(f"{name:60s} | TRAIN n={t['n']:4d} wr={t['wr']*100 if t['n'] else 0:5.1f}% roi={t['roi']*100 if t['n'] else 0:+6.1f}% "
          f"| OOS n={o['n']:4d} wr={o['wr']*100 if o['n'] else 0:5.1f}% cote={o['avg_cote'] if o['n'] else 0:.2f} roi={o['roi']*100 if o['n'] else 0:+6.1f}%")
    return t, o

print("\n" + "=" * 120)
print("H0 — BASELINE: bet X on everything")
report("ALL matches, bet X", train, oos)

print("\n" + "=" * 120)
print("H1 — ODDS PROFILE (buckets defined a priori, scanned on train)")
for lo, hi in [(0, 0.2), (0, 0.3), (0, 0.4), (0.4, 0.8), (0.8, 1.5), (1.5, 99)]:
    report(f"  |oH-oA| in [{lo},{hi})", train[(train['odds_gap'] >= lo) & (train['odds_gap'] < hi)],
           oos[(oos['odds_gap'] >= lo) & (oos['odds_gap'] < hi)])
for lo, hi in [(0, 3.1), (3.1, 3.3), (3.3, 3.5), (3.5, 3.8), (3.8, 99)]:
    report(f"  X odds in [{lo},{hi})", train[(train['odds_draw'] >= lo) & (train['odds_draw'] < hi)],
           oos[(oos['odds_draw'] >= lo) & (oos['odds_draw'] < hi)])
for lo, hi in [(1.0, 1.6), (1.6, 2.0), (2.0, 2.4), (2.4, 99)]:
    report(f"  fav odds in [{lo},{hi})", train[(train['fav_odds'] >= lo) & (train['fav_odds'] < hi)],
           oos[(oos['fav_odds'] >= lo) & (oos['fav_odds'] < hi)])

print("\n" + "=" * 120)
print("H2 — SEGMENT")
for s in ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS']:
    report(f"  segment={s}", train[train['segment'] == s], oos[oos['segment'] == s])

print("\nH2b — SEGMENT x close odds (gap<0.4)")
for s in ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS']:
    report(f"  segment={s} & gap<0.4", train[(train['segment'] == s) & (train['odds_gap'] < 0.4)],
           oos[(oos['segment'] == s) & (oos['odds_gap'] < 0.4)])

print("\n" + "=" * 120)
print("H3 — DRAWISH PAIRS (train-only pair draw rates, applied OOS)")
pair_stats = train.groupby(['team_a', 'team_b'])['is_draw'].agg(['count', 'mean'])
for min_n, min_rate in [(5, 0.40), (5, 0.35), (8, 0.35), (8, 0.30), (10, 0.30)]:
    good_pairs = set(pair_stats[(pair_stats['count'] >= min_n) & (pair_stats['mean'] >= min_rate)].index)
    tm = train[[tuple(x) in good_pairs for x in zip(train['team_a'], train['team_b'])]]
    om = oos[[tuple(x) in good_pairs for x in zip(oos['team_a'], oos['team_b'])]]
    report(f"  pairs n>={min_n} train_rate>={min_rate} ({len(good_pairs)} pairs)", tm, om)

# unordered pairs (either venue)
train['ukey'] = [tuple(sorted(x)) for x in zip(train['team_a'], train['team_b'])]
oos['ukey'] = [tuple(sorted(x)) for x in zip(oos['team_a'], oos['team_b'])]
upair = train.groupby('ukey')['is_draw'].agg(['count', 'mean'])
for min_n, min_rate in [(10, 0.35), (10, 0.30), (14, 0.30)]:
    good = set(upair[(upair['count'] >= min_n) & (upair['mean'] >= min_rate)].index)
    report(f"  UNORDERED pairs n>={min_n} rate>={min_rate} ({len(good)} pairs)",
           train[train['ukey'].isin(good)], oos[oos['ukey'].isin(good)])

print("\n" + "=" * 120)
print("H4 — DRAWISH TEAMS (train per-team draw rate)")
home_rate = train.groupby('team_a')['is_draw'].agg(['count', 'mean']).rename(columns={'mean': 'h_rate'})
away_rate = train.groupby('team_b')['is_draw'].agg(['count', 'mean']).rename(columns={'mean': 'a_rate'})
all_team = pd.concat([
    train[['team_a', 'is_draw']].rename(columns={'team_a': 'team'}),
    train[['team_b', 'is_draw']].rename(columns={'team_b': 'team'})
]).groupby('team')['is_draw'].agg(['count', 'mean'])
print("  Train team draw rates (overall):")
for t_, r_ in all_team.sort_values('mean', ascending=False).head(8).iterrows():
    print(f"    {t_:20s} n={int(r_['count'])} rate={r_['mean']:.3f}")
for thr in [0.26, 0.28, 0.30]:
    drawish = set(all_team[all_team['mean'] >= thr].index)
    tm = train[train['team_a'].isin(drawish) & train['team_b'].isin(drawish)]
    om = oos[oos['team_a'].isin(drawish) & oos['team_b'].isin(drawish)]
    report(f"  BOTH teams drawish (rate>={thr}, {len(drawish)} teams)", tm, om)
    tm = train[train['team_a'].isin(drawish) | train['team_b'].isin(drawish)]
    om = oos[oos['team_a'].isin(drawish) | oos['team_b'].isin(drawish)]
    report(f"  EITHER team drawish (rate>={thr})", tm, om)

print("\n" + "=" * 120)
print("H5 — SEQUENCES (past-only features, no leakage by construction; evaluated on OOS rows)")
# Build per-team chronological draw history over the FULL df (feature uses only past matches)
events_long = pd.concat([
    df[['id', 'expected_start', 'team_a', 'is_draw']].rename(columns={'team_a': 'team'}),
    df[['id', 'expected_start', 'team_b', 'is_draw']].rename(columns={'team_b': 'team'})
]).sort_values('expected_start')
events_long['prev1'] = events_long.groupby('team')['is_draw'].shift(1)
events_long['prev2'] = events_long.groupby('team')['is_draw'].shift(2)
events_long['last5_draws'] = (events_long.groupby('team')['is_draw']
                              .transform(lambda s: s.shift(1).rolling(5).sum()))
feat = events_long.set_index(['id', 'team'])

def team_feat(row, team_col, col):
    try: return feat.loc[(row['id'], row[team_col]), col]
    except KeyError: return np.nan

for d in (train, oos):
    idx = pd.MultiIndex.from_arrays([d['id'], d['team_a']])
    d['h_prev1'] = feat.reindex(idx)['prev1'].values
    d['h_prev2'] = feat.reindex(idx)['prev2'].values
    d['h_last5'] = feat.reindex(idx)['last5_draws'].values
    idx = pd.MultiIndex.from_arrays([d['id'], d['team_b']])
    d['a_prev1'] = feat.reindex(idx)['prev1'].values
    d['a_prev2'] = feat.reindex(idx)['prev2'].values
    d['a_last5'] = feat.reindex(idx)['last5_draws'].values

def seq_mask(d, cond):
    return d[cond(d)]
conds = [
    ("home team 2 consecutive draws", lambda d: (d['h_prev1'] == 1) & (d['h_prev2'] == 1)),
    ("away team 2 consecutive draws", lambda d: (d['a_prev1'] == 1) & (d['a_prev2'] == 1)),
    ("either team 2 consecutive draws", lambda d: ((d['h_prev1'] == 1) & (d['h_prev2'] == 1)) | ((d['a_prev1'] == 1) & (d['a_prev2'] == 1))),
    ("both teams last match = draw", lambda d: (d['h_prev1'] == 1) & (d['a_prev1'] == 1)),
    ("home 0 draws in last 5 (due?)", lambda d: d['h_last5'] == 0),
    ("both 0 draws in last 5 (due?)", lambda d: (d['h_last5'] == 0) & (d['a_last5'] == 0)),
    ("home >=2 draws in last 5", lambda d: d['h_last5'] >= 2),
    ("both >=2 draws in last 5", lambda d: (d['h_last5'] >= 2) & (d['a_last5'] >= 2)),
]
for name, c in conds:
    report(f"  {name}", seq_mask(train, c), seq_mask(oos, c))

print("\n" + "=" * 120)
print("H6 — HALF-TIME X market ('Mi-tps 1X2' X)")
report("  ALL matches, bet HT-X", train, oos, odds_col='ht_x_odds', won_col='is_ht_draw')
for lo, hi in [(0, 0.3), (0, 0.4), (0.4, 1.0), (1.0, 99)]:
    report(f"  HT-X & gap in [{lo},{hi})", train[(train['odds_gap'] >= lo) & (train['odds_gap'] < hi)],
           oos[(oos['odds_gap'] >= lo) & (oos['odds_gap'] < hi)], odds_col='ht_x_odds', won_col='is_ht_draw')
for lo, hi in [(0, 2.15), (2.15, 2.3), (2.3, 2.45), (2.45, 99)]:
    report(f"  HT-X odds in [{lo},{hi})", train[(train['ht_x_odds'] >= lo) & (train['ht_x_odds'] < hi)],
           oos[(oos['ht_x_odds'] >= lo) & (oos['ht_x_odds'] < hi)], odds_col='ht_x_odds', won_col='is_ht_draw')
for s in ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS']:
    report(f"  HT-X & segment={s}", train[train['segment'] == s], oos[oos['segment'] == s],
           odds_col='ht_x_odds', won_col='is_ht_draw')

print("\nH6b — HT/FT X/X market")
df['is_xx'] = ((df['ht_score_a'] == df['ht_score_b']) & (df['score_a'] == df['score_b'])).astype(float)
train['is_xx'] = df.loc[train.index, 'is_xx']
oos['is_xx'] = df.loc[oos.index, 'is_xx']
report("  ALL matches, bet HT/FT X/X", train, oos, odds_col='htft_xx_odds', won_col='is_xx')
report("  X/X & gap<0.4", train[train['odds_gap'] < 0.4], oos[oos['odds_gap'] < 0.4],
       odds_col='htft_xx_odds', won_col='is_xx')

print("\nH6c — HT X -> FT X transition rate (info only)")
htx_t = train[train['is_ht_draw'] == 1]['is_draw'].mean()
htx_o = oos[oos['is_ht_draw'] == 1]['is_draw'].mean()
print(f"  P(FT X | HT X): train={htx_t:.3f} oos={htx_o:.3f}  (not bettable pre-match)")

print("\n" + "=" * 120)
print("H7 — COMBOS (defined from train signal, validated OOS)")
combos = [
    ("gap<0.4 & X<3.4", lambda d: (d['odds_gap'] < 0.4) & (d['odds_draw'] < 3.4)),
    ("gap<0.4 & FS", lambda d: (d['odds_gap'] < 0.4) & (d['segment'] == 'FS')),
    ("gap<0.3 & fav>=2.0", lambda d: (d['odds_gap'] < 0.3) & (d['fav_odds'] >= 2.0)),
    ("fav>=2.2 (no favorite)", lambda d: d['fav_odds'] >= 2.2),
    ("fav>=2.2 & X<3.4", lambda d: (d['fav_odds'] >= 2.2) & (d['odds_draw'] < 3.4)),
    ("gap<0.4 & either team 2 prev draws", lambda d: (d['odds_gap'] < 0.4) & (((d['h_prev1'] == 1) & (d['h_prev2'] == 1)) | ((d['a_prev1'] == 1) & (d['a_prev2'] == 1)))),
    ("X<3.3", lambda d: d['odds_draw'] < 3.3),
    ("X<3.2", lambda d: d['odds_draw'] < 3.2),
]
for name, c in combos:
    report(f"  {name}", train[c(train)], oos[c(oos)])

print("\n" + "=" * 120)
print("ITERATION 2 — focused refinement of surviving signals")

print("\nI2a — Fine X-odds threshold scan (train-selected, OOS-validated)")
for thr in [3.0, 3.05, 3.1, 3.15, 3.2, 3.25]:
    report(f"  X odds <= {thr}", train[train['odds_draw'] <= thr], oos[oos['odds_draw'] <= thr])

print("\nI2b — X cheaper than (or near) a side: structural tightness")
combos2 = [
    ("X < odds_home (X cheaper than home)", lambda d: d['odds_draw'] < d['odds_home']),
    ("X < odds_away", lambda d: d['odds_draw'] < d['odds_away']),
    ("X < min side +0.3", lambda d: d['odds_draw'] < d['fav_odds'] + 0.3),
    ("X < min side +0.6", lambda d: d['odds_draw'] < d['fav_odds'] + 0.6),
    ("X < min side +1.0", lambda d: d['odds_draw'] < d['fav_odds'] + 1.0),
]
for name, c in combos2:
    report(f"  {name}", train[c(train)], oos[c(oos)])

print("\nI2c — TOP-K drawish teams (rank from TRAIN overall draw rate)")
ranked = list(all_team[all_team['count'] >= 100].sort_values('mean', ascending=False).index)
for k in [1, 2, 3, 4]:
    top = set(ranked[:k])
    tm = train[train['team_a'].isin(top) | train['team_b'].isin(top)]
    om = oos[oos['team_a'].isin(top) | oos['team_b'].isin(top)]
    report(f"  top-{k} drawish involved ({sorted(top)})", tm, om)
for k in [3, 4, 5, 6, 8]:
    top = set(ranked[:k])
    tm = train[train['team_a'].isin(top) & train['team_b'].isin(top)]
    om = oos[oos['team_a'].isin(top) & oos['team_b'].isin(top)]
    report(f"  BOTH in top-{k} drawish", tm, om)

print("\nI2d — Burnley (top drawish) refined")
b = 'Burnley'
for name, c in [
    ("Burnley home", lambda d: d['team_a'] == b),
    ("Burnley away", lambda d: d['team_b'] == b),
    ("Burnley any & X<4.5", lambda d: ((d['team_a'] == b) | (d['team_b'] == b)) & (d['odds_draw'] < 4.5)),
    ("Burnley any & X<4.0", lambda d: ((d['team_a'] == b) | (d['team_b'] == b)) & (d['odds_draw'] < 4.0)),
    ("Burnley any & fav>=1.6", lambda d: ((d['team_a'] == b) | (d['team_b'] == b)) & (d['fav_odds'] >= 1.6)),
]:
    report(f"  {name}", train[c(train)], oos[c(oos)])

print("\nI2e — implied draw prob (overround-normalized) buckets")
for d in (train, oos):
    inv = 1 / d['odds_home'] + 1 / d['odds_draw'] + 1 / d['odds_away']
    d['p_x_imp'] = (1 / d['odds_draw']) / inv
for lo, hi in [(0.30, 1.0), (0.28, 1.0), (0.26, 0.28), (0.24, 0.26)]:
    report(f"  p_x_implied in [{lo},{hi})", train[(train['p_x_imp'] >= lo) & (train['p_x_imp'] < hi)],
           oos[(oos['p_x_imp'] >= lo) & (oos['p_x_imp'] < hi)])

print("\nI2f — multi-fold walk-forward for the X<=3.10 rule (rule is odds-only, no fitting)")
# 5 sequential folds on the last 50% of data, expanding-window style evaluation
df2 = df.dropna(subset=['odds_draw']).reset_index(drop=True)
half = int(len(df2) * 0.5)
test_all = df2.iloc[half:]
sel = test_all[test_all['odds_draw'] <= 3.10]
e = evaluate(sel)
print(f"  X<=3.10 on last 50% of history: n={e['n']} wr={e['wr']*100:.1f}% cote={e['avg_cote']:.2f} roi={e['roi']*100:+.1f}%")
sel_full = df2[df2['odds_draw'] <= 3.10]
e2 = evaluate(sel_full)
print(f"  X<=3.10 full sample (info):     n={e2['n']} wr={e2['wr']*100:.1f}% cote={e2['avg_cote']:.2f} roi={e2['roi']*100:+.1f}%")
# per-quintile stability
df2['fold'] = pd.qcut(np.arange(len(df2)), 5, labels=False)
for f_ in range(5):
    s = df2[(df2['fold'] == f_) & (df2['odds_draw'] <= 3.10)]
    e3 = evaluate(s)
    print(f"    fold {f_+1}/5: n={e3['n']:3d} wr={(e3['wr'] or 0)*100:5.1f}% roi={(e3['roi'] or 0)*100:+6.1f}%")

print("\nI2g — drawish team rate computed by venue (train), applied OOS")
h_rate2 = train.groupby('team_a')['is_draw'].agg(['count', 'mean'])
a_rate2 = train.groupby('team_b')['is_draw'].agg(['count', 'mean'])
top_home = set(h_rate2[h_rate2['count'] >= 50].sort_values('mean', ascending=False).head(3).index)
top_away = set(a_rate2[a_rate2['count'] >= 50].sort_values('mean', ascending=False).head(3).index)
print(f"  top3 drawish at home: {sorted(top_home)}  | top3 drawish away: {sorted(top_away)}")
report("  home team in top3-drawish-home", train[train['team_a'].isin(top_home)], oos[oos['team_a'].isin(top_home)])
report("  away team in top3-drawish-away", train[train['team_b'].isin(top_away)], oos[oos['team_b'].isin(top_away)])
report("  home top3-home OR away top3-away",
       train[train['team_a'].isin(top_home) | train['team_b'].isin(top_away)],
       oos[oos['team_a'].isin(top_home) | oos['team_b'].isin(top_away)])

print("\nI2h — combo: drawish-team x odds-cap")
top2 = set(ranked[:2])
for name, c in [
    ("top2 involved & X<4.6", lambda d: (d['team_a'].isin(top2) | d['team_b'].isin(top2)) & (d['odds_draw'] < 4.6)),
    ("top2 involved & X<4.2", lambda d: (d['team_a'].isin(top2) | d['team_b'].isin(top2)) & (d['odds_draw'] < 4.2)),
    ("top3 involved & X<4.2", lambda d: (d['team_a'].isin(set(ranked[:3])) | d['team_b'].isin(set(ranked[:3]))) & (d['odds_draw'] < 4.2)),
    ("top1 involved & X<4.6", lambda d: (d['team_a'].isin(set(ranked[:1])) | d['team_b'].isin(set(ranked[:1]))) & (d['odds_draw'] < 4.6)),
]:
    report(f"  {name}", train[c(train)], oos[c(oos)])

print("\n" + "=" * 120)
print("ITERATION 3 — stability checks & final portfolio")

print("\nI3a — team draw-rate RANK persistence (train 1st half vs train 2nd half)")
t1 = train.iloc[:len(train) // 2]
t2 = train.iloc[len(train) // 2:]
def team_rates(d):
    return pd.concat([
        d[['team_a', 'is_draw']].rename(columns={'team_a': 'team'}),
        d[['team_b', 'is_draw']].rename(columns={'team_b': 'team'})
    ]).groupby('team')['is_draw'].mean()
r1, r2 = team_rates(t1), team_rates(t2)
both = pd.concat([r1.rename('h1'), r2.rename('h2')], axis=1).dropna()
print(f"  Spearman corr of team draw rates (train h1 vs h2): {both['h1'].corr(both['h2'], method='spearman'):.3f}")
print(f"  Burnley rank: h1={both['h1'].rank(ascending=False)['Burnley']:.0f}/{len(both)}  h2={both['h2'].rank(ascending=False)['Burnley']:.0f}/{len(both)}")
print(f"  Everton rank: h1={both['h1'].rank(ascending=False)['Everton']:.0f}/{len(both)}  h2={both['h2'].rank(ascending=False)['Everton']:.0f}/{len(both)}")

print("\nI3b — OOS split in 2 halves (temporal) for surviving rules")
o1 = oos.iloc[:len(oos) // 2]
o2 = oos.iloc[len(oos) // 2:]
rules = [
    ("Burnley any", lambda d: (d['team_a'] == 'Burnley') | (d['team_b'] == 'Burnley')),
    ("Burnley any & X<4.5", lambda d: ((d['team_a'] == 'Burnley') | (d['team_b'] == 'Burnley')) & (d['odds_draw'] < 4.5)),
    ("Burnley away", lambda d: d['team_b'] == 'Burnley'),
    ("X<=3.10", lambda d: d['odds_draw'] <= 3.10),
]
for name, c in rules:
    for lab, part in [('OOS-h1', o1), ('OOS-h2', o2)]:
        e = evaluate(part[c(part)])
        print(f"  {name:30s} {lab}: n={e['n']:3d} wr={(e['wr'] or 0)*100:5.1f}% roi={(e['roi'] or 0)*100:+6.1f}%")

print("\nI3c — FINAL PORTFOLIO: (Burnley any & X<4.5) OR (X<=3.10), dedup")
def pf(d):
    return d[(((d['team_a'] == 'Burnley') | (d['team_b'] == 'Burnley')) & (d['odds_draw'] < 4.5)) | (d['odds_draw'] <= 3.10)]
report("  PORTFOLIO X", pf(train), pf(oos))
sel = pf(oos)
days = (oos['expected_start'].max() - oos['expected_start'].min()).total_seconds() / 86400
print(f"  pick frequency OOS: {len(sel)} picks / {days:.1f} days = {len(sel)/days:.1f} picks/day")
print(f"  cote distribution OOS: min={sel['odds_draw'].min():.2f} med={sel['odds_draw'].median():.2f} max={sel['odds_draw'].max():.2f}")
# cumulative PnL trace (flat 1u)
won = sel.sort_values('expected_start')['is_draw'].values
cotes = sel.sort_values('expected_start')['odds_draw'].values
pnl = np.cumsum(won * (cotes - 1) - (1 - won))
print(f"  cumulative PnL (1u flat): final={pnl[-1]:+.1f}u  maxDD={np.max(np.maximum.accumulate(pnl) - pnl):.1f}u")

print("\nI3d — Everton as 2nd team, same treatment (validation of generalization)")
for name, c in [
    ("Everton any", lambda d: (d['team_a'] == 'Everton') | (d['team_b'] == 'Everton')),
    ("Everton any & X<4.5", lambda d: ((d['team_a'] == 'Everton') | (d['team_b'] == 'Everton')) & (d['odds_draw'] < 4.5)),
]:
    report(f"  {name}", train[c(train)], oos[c(oos)])

print("\n" + "=" * 120)
print("ITERATION 4 — expected-goals proxy: '+/-' under-3.5 odds & 'Total de buts' implied xG")

def parse_totals(em):
    if em is None: return (np.nan, np.nan)
    if isinstance(em, str):
        try: em = json.loads(em)
        except Exception: return (np.nan, np.nan)
    if not isinstance(em, dict): return (np.nan, np.nan)
    under = np.nan
    m = em.get('+/-')
    if isinstance(m, dict):
        for k, v in m.items():
            if k.strip().startswith('<') and v:
                under = float(v)
    exg = np.nan
    m = em.get('Total de buts')
    if isinstance(m, dict):
        try:
            probs = {int(k): 1.0 / float(v) for k, v in m.items() if v}
            z = sum(probs.values())
            exg = sum(g * p / z for g, p in probs.items())
        except Exception:
            pass
    return (under, exg)

pt = df['extra_markets'].apply(parse_totals)
df['under35'] = [p[0] for p in pt]
df['exp_goals'] = [p[1] for p in pt]
train['under35'] = df.loc[train.index, 'under35']
train['exp_goals'] = df.loc[train.index, 'exp_goals']
oos['under35'] = df.loc[oos.index, 'under35']
oos['exp_goals'] = df.loc[oos.index, 'exp_goals']
print(f"  under35 coverage: {df['under35'].notna().mean()*100:.0f}%  exp_goals: {df['exp_goals'].notna().mean()*100:.0f}% "
      f"(mean exp_goals={df['exp_goals'].mean():.2f})")

print("\nI4a — under-3.5 odds buckets (low odds = low-scoring expected)")
for lo, hi in [(1.0, 1.40), (1.40, 1.50), (1.50, 1.60), (1.60, 1.75), (1.75, 9)]:
    report(f"  under3.5 odds in [{lo},{hi})", train[(train['under35'] >= lo) & (train['under35'] < hi)],
           oos[(oos['under35'] >= lo) & (oos['under35'] < hi)])

print("\nI4b — implied expected goals buckets")
for lo, hi in [(0, 2.6), (2.6, 2.9), (2.9, 3.2), (3.2, 9)]:
    report(f"  exp_goals in [{lo},{hi})", train[(train['exp_goals'] >= lo) & (train['exp_goals'] < hi)],
           oos[(oos['exp_goals'] >= lo) & (oos['exp_goals'] < hi)])

print("\nI4c — combos low-scoring x tight match")
for name, c in [
    ("exp_goals<2.9 & gap<0.8", lambda d: (d['exp_goals'] < 2.9) & (d['odds_gap'] < 0.8)),
    ("exp_goals<2.9 & X<3.6", lambda d: (d['exp_goals'] < 2.9) & (d['odds_draw'] < 3.6)),
    ("under35<1.5 & X<3.6", lambda d: (d['under35'] < 1.5) & (d['odds_draw'] < 3.6)),
    ("under35<1.5 & gap<0.8", lambda d: (d['under35'] < 1.5) & (d['odds_gap'] < 0.8)),
    ("exp_goals<2.7 & X<=3.5", lambda d: (d['exp_goals'] < 2.7) & (d['odds_draw'] <= 3.5)),
]:
    report(f"  {name}", train[c(train)], oos[c(oos)])

print("\nI4d — HT-X with low expected goals (fewer goals => more HT draws?)")
for name, c in [
    ("HT-X & exp_goals<2.7", lambda d: d['exp_goals'] < 2.7),
    ("HT-X & exp_goals<2.9", lambda d: d['exp_goals'] < 2.9),
    ("HT-X & under35<1.5", lambda d: d['under35'] < 1.5),
]:
    report(f"  {name}", train[c(train)], oos[c(oos)], odds_col='ht_x_odds', won_col='is_ht_draw')
