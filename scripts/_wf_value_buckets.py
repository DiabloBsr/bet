# -*- coding: utf-8 -*-
"""
Walk-forward value buckets — recherche de filtres 1X2 cotes 2.0-6.0, ROI_oos >= +15%.
Methodo anti-leakage :
  - tri par expected_start, train = premiers 70%, OOS = derniers 30%
  - tout signal (paires, deltas equipes, brackets) calcule UNIQUEMENT sur le train
  - selection des candidats sur le TRAIN, evaluation unique sur l'OOS
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

pd.set_option('display.width', 220)
pd.set_option('display.max_rows', 300)

# ---------------------------------------------------------------- load
eng = create_engine(load_settings().db_url)
Q = """
SELECT e.id AS event_id, CAST(e.round_info AS INTEGER) AS round, e.team_a, e.team_b,
       e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away,
       r.score_a, r.score_b
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o
  ON o.id = (SELECT MIN(id) FROM odds_snapshots os WHERE os.event_id = e.id)
WHERE e.round_info != '0' AND e.round_info IS NOT NULL
  AND o.odds_home IS NOT NULL AND o.odds_draw IS NOT NULL AND o.odds_away IS NOT NULL
  AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
ORDER BY e.expected_start, e.id
"""
df = pd.read_sql(Q, eng)
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values(['expected_start', 'event_id']).reset_index(drop=True)
print(f"loaded {len(df)} matches  {df.expected_start.min()} -> {df.expected_start.max()}")

# ---------------------------------------------------------------- derive
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
df['draw']     = (df.outcome == 'D').astype(int)
df['round_parity'] = np.where(df['round'] % 2 == 0, 'even', 'odd')

SEG_BOUNDS = {'DS': (1, 3), 'MS_early': (4, 12), 'MS_mid': (13, 25), 'MS_late': (26, 33), 'FS': (34, 38)}
def seg_pos(row):
    lo, hi = SEG_BOUNDS[row['segment']]
    if hi == lo: return 'mid'
    p = (row['round'] - lo) / (hi - lo)
    return 'start' if p < 0.34 else ('end' if p > 0.66 else 'mid')
df['seg_pos'] = df.apply(seg_pos, axis=1)

# ---------------------------------------------------------------- causal history features (chronological, past-only)
prev_home_res = {}     # team -> result of its LAST home match ('W','D','L')
last3 = {}             # team -> list of last 3 points (any venue)
fa_prev, fa_pts3, fb_pts3 = [], [], []
for _, r in df.iterrows():
    ta, tb = r.team_a, r.team_b
    fa_prev.append(prev_home_res.get(ta))
    fa_pts3.append(sum(last3.get(ta, [])) if last3.get(ta) else None)
    fb_pts3.append(sum(last3.get(tb, [])) if last3.get(tb) else None)
    # update AFTER recording features
    res_a = 'W' if r.outcome == 'H' else ('D' if r.outcome == 'D' else 'L')
    prev_home_res[ta] = res_a
    pa = 3 if r.outcome == 'H' else (1 if r.outcome == 'D' else 0)
    pb = 3 if r.outcome == 'A' else (1 if r.outcome == 'D' else 0)
    for t, p in ((ta, pa), (tb, pb)):
        h = last3.setdefault(t, [])
        h.append(p)
        if len(h) > 3: h.pop(0)
df['home_prev_home_res'] = fa_prev
df['ta_pts3'] = fa_pts3
df['tb_pts3'] = fb_pts3

# ---------------------------------------------------------------- temporal split
cut = int(len(df) * 0.70)
train, oos = df.iloc[:cut].copy(), df.iloc[cut:].copy()
print(f"train={len(train)} ({train.expected_start.min().date()}->{train.expected_start.max().date()})  "
      f"oos={len(oos)} ({oos.expected_start.min().date()}->{oos.expected_start.max().date()})")

# ---------------------------------------------------------------- eval helper
def evaluate(sub, side):
    """side in {'H','A','D'} ; returns n, wr, avg_odds, roi"""
    if len(sub) == 0: return 0, np.nan, np.nan, np.nan
    odds = {'H': sub.odds_home, 'A': sub.odds_away, 'D': sub.odds_draw}[side]
    won = (sub.outcome == side).astype(float)
    roi = (won * (odds - 1) - (1 - won)).mean()
    return len(sub), won.mean(), odds.mean(), roi

def report(name, mask_train, mask_oos, side, definition):
    nt, wt, at_, rt = evaluate(train[mask_train], side)
    no, wo, ao, ro = evaluate(oos[mask_oos], side)
    return dict(name=name, side=side, definition=definition,
                n_train=nt, wr_train=wt, roi_train=rt,
                n_oos=no, wr_oos=wo, avg_cote_oos=ao, roi_oos=ro)

results = []

# ================================================================ H1: segment x side x odds bucket
BUCKETS = [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0)]
LOW_BUCKETS = [(1.0, 1.3), (1.3, 1.5), (1.5, 1.8), (1.8, 2.0)]
SEGS = ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS', 'ALL']

print("\n========== H1: segment x odds-bucket (train -> oos) ==========")
h1 = []
for side, col in (('H', 'odds_home'), ('A', 'odds_away')):
    for lo, hi in BUCKETS + LOW_BUCKETS:
        for seg in SEGS:
            mt = (train[col] >= lo) & (train[col] < hi)
            mo = (oos[col] >= lo) & (oos[col] < hi)
            if seg != 'ALL':
                mt &= train.segment == seg
                mo &= oos.segment == seg
            r = report(f"H1 {side} {seg} cote[{lo},{hi})", mt, mo, side,
                       f"bet {side}: {col} in [{lo},{hi})" + (f" AND segment={seg}" if seg != 'ALL' else ""))
            h1.append(r)
# draws
for lo, hi in [(2.9, 3.2), (3.2, 3.6), (3.6, 4.5), (4.5, 9.5)]:
    for seg in SEGS:
        mt = (train.odds_draw >= lo) & (train.odds_draw < hi)
        mo = (oos.odds_draw >= lo) & (oos.odds_draw < hi)
        if seg != 'ALL':
            mt &= train.segment == seg
            mo &= oos.segment == seg
        r = report(f"H1 D {seg} cote[{lo},{hi})", mt, mo, 'D',
                   f"bet DRAW: odds_draw in [{lo},{hi})" + (f" AND segment={seg}" if seg != 'ALL' else ""))
        h1.append(r)

h1df = pd.DataFrame(h1)
sel = h1df[(h1df.n_train >= 40) & (h1df.roi_train >= 0.08)]
print(sel.sort_values('roi_train', ascending=False)[
    ['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))
results += h1.copy()

# ================================================================ H2: team delta forme (train-only) x odds bucket
# per-team train stats
th = train.groupby('team_a').agg(hw=('home_win', 'mean'), hn=('home_win', 'size'))
ta_ = train.groupby('team_b').agg(aw=('away_win', 'mean'), an=('away_win', 'size'))
ths = train.groupby(['team_a', 'segment']).agg(hw=('home_win', 'mean'), hn=('home_win', 'size'))
tas = train.groupby(['team_b', 'segment']).agg(aw=('away_win', 'mean'), an=('away_win', 'size'))

def team_home_wr(t): return th.hw.get(t, np.nan)
def team_away_wr(t): return ta_.aw.get(t, np.nan)
def team_home_seg_delta(t, s):
    try:
        row = ths.loc[(t, s)]
        if row.hn < 8: return np.nan
        return row.hw - th.hw.get(t, np.nan)
    except KeyError:
        return np.nan
def team_away_seg_delta(t, s):
    try:
        row = tas.loc[(t, s)]
        if row.an < 8: return np.nan
        return row.aw - ta_.aw.get(t, np.nan)
    except KeyError:
        return np.nan

for d in (train, oos):
    d['ta_home_wr'] = d.team_a.map(team_home_wr)
    d['tb_away_wr'] = d.team_b.map(team_away_wr)
    d['ta_hseg_delta'] = d.apply(lambda r: team_home_seg_delta(r.team_a, r.segment), axis=1)
    d['tb_aseg_delta'] = d.apply(lambda r: team_away_seg_delta(r.team_b, r.segment), axis=1)
    d['edge_home'] = d['ta_home_wr'] - 1.0 / d['odds_home']   # value vs implied
    d['edge_away'] = d['tb_away_wr'] - 1.0 / d['odds_away']

print("\n========== H2: team value edge / segment delta (train-only signals) ==========")
h2 = []
for lo, hi in [(2.0, 6.0), (2.0, 3.0), (3.0, 6.0), (2.0, 2.5), (2.5, 3.5)]:
    for thr in (0.05, 0.10, 0.15):
        mt = (train.odds_home >= lo) & (train.odds_home < hi) & (train.edge_home >= thr)
        mo = (oos.odds_home >= lo) & (oos.odds_home < hi) & (oos.edge_home >= thr)
        h2.append(report(f"H2 H value edge>={thr} cote[{lo},{hi})", mt, mo, 'H',
                  f"bet HOME: odds_home in [{lo},{hi}) AND (train_home_WR(team_a) - 1/odds_home) >= {thr}"))
        mt = (train.odds_away >= lo) & (train.odds_away < hi) & (train.edge_away >= thr)
        mo = (oos.odds_away >= lo) & (oos.odds_away < hi) & (oos.edge_away >= thr)
        h2.append(report(f"H2 A value edge>={thr} cote[{lo},{hi})", mt, mo, 'A',
                  f"bet AWAY: odds_away in [{lo},{hi}) AND (train_away_WR(team_b) - 1/odds_away) >= {thr}"))
    for dthr in (0.08, 0.12):
        mt = (train.odds_home >= lo) & (train.odds_home < hi) & (train.ta_hseg_delta >= dthr)
        mo = (oos.odds_home >= lo) & (oos.odds_home < hi) & (oos.ta_hseg_delta >= dthr)
        h2.append(report(f"H2 H segdelta>={dthr} cote[{lo},{hi})", mt, mo, 'H',
                  f"bet HOME: odds_home in [{lo},{hi}) AND train (home_WR_segment - home_WR_global)(team_a) >= {dthr} (n_seg>=8)"))
        mt = (train.odds_away >= lo) & (train.odds_away < hi) & (train.tb_aseg_delta >= dthr)
        mo = (oos.odds_away >= lo) & (oos.odds_away < hi) & (oos.tb_aseg_delta >= dthr)
        h2.append(report(f"H2 A segdelta>={dthr} cote[{lo},{hi})", mt, mo, 'A',
                  f"bet AWAY: odds_away in [{lo},{hi}) AND train (away_WR_segment - away_WR_global)(team_b) >= {dthr} (n_seg>=8)"))
h2df = pd.DataFrame(h2)
print(h2df[(h2df.n_train >= 40)].sort_values('roi_train', ascending=False).head(25)[
    ['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))
results += h2

# ================================================================ H3: cycle position x odds bucket
print("\n========== H3: parity / seg position x bucket ==========")
h3 = []
for side, col in (('H', 'odds_home'), ('A', 'odds_away')):
    for lo, hi in BUCKETS:
        for par in ('even', 'odd'):
            mt = (train[col] >= lo) & (train[col] < hi) & (train.round_parity == par)
            mo = (oos[col] >= lo) & (oos[col] < hi) & (oos.round_parity == par)
            h3.append(report(f"H3 {side} {par} cote[{lo},{hi})", mt, mo, side,
                      f"bet {side}: {col} in [{lo},{hi}) AND round {par}"))
        for sp in ('start', 'mid', 'end'):
            mt = (train[col] >= lo) & (train[col] < hi) & (train.seg_pos == sp)
            mo = (oos[col] >= lo) & (oos[col] < hi) & (oos.seg_pos == sp)
            h3.append(report(f"H3 {side} segpos={sp} cote[{lo},{hi})", mt, mo, side,
                      f"bet {side}: {col} in [{lo},{hi}) AND position-in-segment={sp}"))
h3df = pd.DataFrame(h3)
print(h3df[(h3df.n_train >= 40) & (h3df.roi_train >= 0.10)].sort_values('roi_train', ascending=False).head(20)[
    ['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))
results += h3

# ================================================================ H4: golden pairs walk-forward (train-only)
print("\n========== H4: golden pairs (defined on train, evaluated oos) ==========")
h4 = []
for side, wcol in (('H', 'home_win'), ('A', 'away_win')):
    for wr_thr, n_thr in [(0.70, 6), (0.70, 8), (0.60, 6), (0.75, 6)]:
        pg = train.groupby(['team_a', 'team_b']).agg(w=(wcol, 'mean'), n=(wcol, 'size'))
        gold = set(pg[(pg.w >= wr_thr) & (pg.n >= n_thr)].index)
        mt = train.apply(lambda r: (r.team_a, r.team_b) in gold, axis=1)
        mo = oos.apply(lambda r: (r.team_a, r.team_b) in gold, axis=1)
        h4.append(report(f"H4 pairs {side} wr>={wr_thr} n>={n_thr} (k={len(gold)})", mt, mo, side,
                  f"bet {side}: ordered pair (team_a,team_b) had train {side}-WR>={wr_thr} with n>={n_thr}"))
        # high odds subset
        col = 'odds_home' if side == 'H' else 'odds_away'
        mt2 = mt & (train[col] >= 2.0)
        mo2 = mo & (oos[col] >= 2.0)
        h4.append(report(f"H4 pairs {side} wr>={wr_thr} n>={n_thr} cote>=2.0", mt2, mo2, side,
                  f"bet {side}: golden pair (train WR>={wr_thr}, n>={n_thr}) AND {col} >= 2.0"))
h4df = pd.DataFrame(h4)
print(h4df[['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))
results += h4

# ================================================================ H5: favoris trahis / rebond
print("\n========== H5: favori trahi (lost last home match) ==========")
h5 = []
# home rebond after home loss, by odds bucket
for lo, hi in [(1.0, 1.8), (1.8, 2.5), (2.0, 3.0), (2.5, 4.0), (2.0, 6.0)]:
    mt = (train.odds_home >= lo) & (train.odds_home < hi) & (train.home_prev_home_res == 'L')
    mo = (oos.odds_home >= lo) & (oos.odds_home < hi) & (oos.home_prev_home_res == 'L')
    h5.append(report(f"H5 H rebond cote[{lo},{hi}) prevhome=L", mt, mo, 'H',
              f"bet HOME: odds_home in [{lo},{hi}) AND team_a LOST its previous home match"))
    # spiral: fade them -> bet away
    h5.append(report(f"H5 A fade cote_home[{lo},{hi}) prevhome=L", mt, mo, 'A',
              f"bet AWAY: odds_home in [{lo},{hi}) AND team_a LOST its previous home match"))
    # control: won last home
    mtW = (train.odds_home >= lo) & (train.odds_home < hi) & (train.home_prev_home_res == 'W')
    moW = (oos.odds_home >= lo) & (oos.odds_home < hi) & (oos.home_prev_home_res == 'W')
    h5.append(report(f"H5 H confirm cote[{lo},{hi}) prevhome=W", mtW, moW, 'H',
              f"bet HOME: odds_home in [{lo},{hi}) AND team_a WON its previous home match"))
h5df = pd.DataFrame(h5)
print(h5df[['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))
results += h5

# ================================================================ H6: forme courte (pts last3) x bucket
print("\n========== H6: forme courte (points 3 derniers matchs) ==========")
h6 = []
for side, col in (('H', 'odds_home'), ('A', 'odds_away')):
    fcol = 'ta_pts3' if side == 'H' else 'tb_pts3'
    for lo, hi in [(2.0, 3.0), (3.0, 6.0), (2.0, 6.0)]:
        for fl, fh, lab in [(7, 10, 'hot(7-9)'), (0, 3, 'cold(0-2)'), (4, 7, 'mid(4-6)')]:
            mt = (train[col] >= lo) & (train[col] < hi) & (train[fcol] >= fl) & (train[fcol] < fh)
            mo = (oos[col] >= lo) & (oos[col] < hi) & (oos[fcol] >= fl) & (oos[fcol] < fh)
            h6.append(report(f"H6 {side} {lab} cote[{lo},{hi})", mt, mo, side,
                      f"bet {side}: {col} in [{lo},{hi}) AND pts last 3 matches of bet team in [{fl},{fh})"))
h6df = pd.DataFrame(h6)
print(h6df[h6df.n_train >= 40].sort_values('roi_train', ascending=False).head(15)[
    ['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))
results += h6

# ================================================================ summary: train-selected candidates, oos verdict
print("\n\n################ TRAIN-SELECTED (roi_train>=0.10, n_train>=40, avg cote 2-6) -> OOS ################")
rdf = pd.DataFrame(results)
cand = rdf[(rdf.roi_train >= 0.10) & (rdf.n_train >= 40)]
print(cand.sort_values('roi_oos', ascending=False)[
    ['name', 'definition', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'avg_cote_oos', 'roi_oos']].to_string(index=False))

rdf.to_csv('scripts/_wf_value_buckets_results.csv', index=False)
print("\nsaved scripts/_wf_value_buckets_results.csv")
