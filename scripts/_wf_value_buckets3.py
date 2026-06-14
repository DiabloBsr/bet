# -*- coding: utf-8 -*-
"""
Iteration 3 — walk-forward roulant (5 folds de 10% a partir de 50%) :
pour chaque fold, signaux equipe recalcules sur TOUT le passe, evaluation sur le fold.
Aggregat = pseudo-OOS ou chaque pari n'utilise que de l'info passee.
+ diagnostics composition equipes du signal edge_home.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

pd.set_option('display.width', 240)

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

def team_signals(base):
    th = base.groupby('team_a').agg(hw=('home_win', 'mean'), hn=('home_win', 'size'))
    ta = base.groupby('team_b').agg(aw=('away_win', 'mean'), an=('away_win', 'size'))
    ths = base.groupby(['team_a', 'segment']).agg(hw=('home_win', 'mean'), hn=('home_win', 'size'))
    def hwr(t): return th.hw.get(t, np.nan)
    def awr(t): return ta.aw.get(t, np.nan)
    def hsegd(t, s):
        try:
            row = ths.loc[(t, s)]
            return row.hw - th.hw.get(t, np.nan) if row.hn >= 6 else np.nan
        except KeyError:
            return np.nan
    return hwr, awr, hsegd

def add_signals(d, sig):
    hwr, awr, hsegd = sig
    d = d.copy()
    d['ta_home_wr'] = d.team_a.map(hwr)
    d['tb_away_wr'] = d.team_b.map(awr)
    d['ta_hseg_delta'] = d.apply(lambda r: hsegd(r.team_a, r.segment), axis=1)
    d['edge_home'] = d['ta_home_wr'] - 1.0 / d['odds_home']
    d['edge_away'] = d['tb_away_wr'] - 1.0 / d['odds_away']
    return d

def ev(sub, side):
    if len(sub) == 0: return 0, np.nan, np.nan, np.nan
    odds = {'H': sub.odds_home, 'A': sub.odds_away, 'D': sub.odds_draw}[side]
    won = (sub.outcome == side).astype(float)
    return len(sub), won.mean(), odds.mean(), (won * (odds - 1) - (1 - won)).mean()

RULES = [
    ("R1 edge>=0.10 [2.5,3.5)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.10)),
    ("R2 edge>=0.10 [2.0,3.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 3.0) & (d.edge_home >= 0.10)),
    ("R3 edge>=0.10 [2.0,3.5) pooled", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 3.5) & (d.edge_home >= 0.10)),
    ("R4 edge>=0.05 [2.5,3.5)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.05)),
    ("R5 edge>=0.10 [2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.10)),
    ("R6 edge>=0.10 + prevL [2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.10) & (d.home_prev_home_res == 'L')),
    ("R7 segdelta>=0.08 [2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.ta_hseg_delta >= 0.08)),
    ("R8 segdelta>=0.10 [2.0,6.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.ta_hseg_delta >= 0.10)),
    ("R9 home longshot [4.0,6.0)", 'H', lambda d: (d.odds_home >= 4.0) & (d.odds_home < 6.0)),
    ("R10 away fav <=1.40", 'A', lambda d: d.odds_away <= 1.40),
    ("R11 rebond prevL [2.5,4.0)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 4.0) & (d.home_prev_home_res == 'L')),
    ("R12 edge>=0.08 [2.5,3.5)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.08)),
]

# ---------------- rolling walk-forward: folds [50-60), [60-70), ... [90-100)
n = len(df)
fold_edges = [int(n * x) for x in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
print(f"total={n}, folds at {fold_edges}")
fold_frames = {name: [] for name, _, _ in RULES}
fold_rois = {name: [] for name, _, _ in RULES}
for k in range(5):
    lo_i, hi_i = fold_edges[k], fold_edges[k + 1]
    past = df.iloc[:lo_i]
    fold = df.iloc[lo_i:hi_i]
    sig = team_signals(past)
    foldS = add_signals(fold, sig)
    for name, side, fn in RULES:
        sub = foldS[fn(foldS)]
        fold_frames[name].append(sub.assign(_side=side))
        nf, wf, of, rf = ev(sub, side)
        fold_rois[name].append((nf, rf))

print("\n===== ROLLING WALK-FORWARD (signal = tout le passe, 5 folds de 10%) =====")
rows = []
for name, side, fn in RULES:
    allb = pd.concat(fold_frames[name])
    na, wa, oa, ra = ev(allb, side)
    per_fold = "  ".join(f"f{k+1}:n={nf},roi={rf:+.2f}" if nf else f"f{k+1}:n=0" for k, (nf, rf) in enumerate(fold_rois[name]))
    pos_folds = sum(1 for nf, rf in fold_rois[name] if nf > 0 and rf > 0)
    rows.append(dict(name=name, n_agg=na, wr_agg=wa, cote_agg=oa, roi_agg=ra, pos_folds=f"{pos_folds}/5"))
    print(f"{name:35s} AGG: n={na:4d} wr={wa:.3f} cote={oa:.2f} roi={ra:+.3f}  folds+:{pos_folds}/5   [{per_fold}]")

# ---------------- diagnostics: team composition of edge rule (signals on 70% train)
print("\n===== DIAGNOSTIC: equipes du filtre R3 (signaux train 70%) sur l'OOS =====")
cut = int(n * 0.7)
train, oos = df.iloc[:cut], df.iloc[cut:]
sig = team_signals(train)
ooS = add_signals(oos, sig)
trS = add_signals(train, sig)
sub = ooS[(ooS.odds_home >= 2.0) & (ooS.odds_home < 3.5) & (ooS.edge_home >= 0.10)]
comp = sub.groupby('team_a').apply(
    lambda g: pd.Series(dict(n=len(g), wr=(g.outcome == 'H').mean(),
                             roi=((g.outcome == 'H') * (g.odds_home - 1) - (g.outcome != 'H') * 1.0).mean())),
    include_groups=False)
print(comp.sort_values('n', ascending=False).to_string())
hwr, _, _ = sig
print("\ntrain home WR par equipe:")
print(pd.Series({t: hwr(t) for t in sorted(df.team_a.unique())}).sort_values(ascending=False).to_string())

# distribution mensuelle/journee des picks R3 oos
print(f"\nR3 OOS: {len(sub)} picks, repartition segment: {sub.segment.value_counts().to_dict()}")
print(f"cotes: min={sub.odds_home.min():.2f} max={sub.odds_home.max():.2f} mean={sub.odds_home.mean():.2f}")
