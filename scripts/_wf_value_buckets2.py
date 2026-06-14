# -*- coding: utf-8 -*-
"""
Iteration 2 — validation imbriquee + stabilite.
Protocole nested : A = 0-50% (calcul des signaux equipe), B = 50-70% (selection des regles),
C = 70-100% (OOS final, jamais utilise pour selectionner).
Puis pour les regles retenues : version deployable (signaux recalcules sur 70%) evaluee sur OOS,
avec stabilite par demi-OOS.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

pd.set_option('display.width', 240)
pd.set_option('display.max_rows', 400)

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
SEG_BOUNDS = {'DS': (1, 3), 'MS_early': (4, 12), 'MS_mid': (13, 25), 'MS_late': (26, 33), 'FS': (34, 38)}
def seg_pos(row):
    lo, hi = SEG_BOUNDS[row['segment']]
    p = (row['round'] - lo) / max(hi - lo, 1)
    return 'start' if p < 0.34 else ('end' if p > 0.66 else 'mid')
df['seg_pos'] = df.apply(seg_pos, axis=1)

# causal prev home result
prev_home = {}
vals = []
for _, r in df.iterrows():
    vals.append(prev_home.get(r.team_a))
    prev_home[r.team_a] = 'W' if r.outcome == 'H' else ('D' if r.outcome == 'D' else 'L')
df['home_prev_home_res'] = vals

n = len(df)
iA, iB = int(n * 0.50), int(n * 0.70)
A, B, C = df.iloc[:iA], df.iloc[iA:iB], df.iloc[iB:]
print(f"A(signal)={len(A)}  B(select)={len(B)}  C(OOS)={len(C)}")

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

def add_signals(d, hwr, awr, hsegd):
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

# ============================================================ NESTED: signals on A, select on B, final on C
hwrA, awrA, hsegdA = team_signals(A)
Bs = add_signals(B, hwrA, awrA, hsegdA)
Cs = add_signals(C, hwrA, awrA, hsegdA)

BUCKETS = [(2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 6.0), (2.0, 3.0), (3.0, 6.0), (2.0, 6.0), (2.5, 3.5), (2.5, 6.0)]
rules = []
for lo, hi in BUCKETS:
    for thr in (0.03, 0.05, 0.08, 0.10, 0.12, 0.15):
        rules.append((f"H edge>={thr} [{lo},{hi})", 'H',
                      lambda d, lo=lo, hi=hi, thr=thr: (d.odds_home >= lo) & (d.odds_home < hi) & (d.edge_home >= thr),
                      f"bet HOME: odds_home in [{lo},{hi}) AND (signal_home_WR(team_a) - 1/odds_home) >= {thr}"))
        rules.append((f"A edge>={thr} [{lo},{hi})", 'A',
                      lambda d, lo=lo, hi=hi, thr=thr: (d.odds_away >= lo) & (d.odds_away < hi) & (d.edge_away >= thr),
                      f"bet AWAY: odds_away in [{lo},{hi}) AND (signal_away_WR(team_b) - 1/odds_away) >= {thr}"))
    for thr in (0.06, 0.08, 0.10, 0.12):
        rules.append((f"H segdelta>={thr} [{lo},{hi})", 'H',
                      lambda d, lo=lo, hi=hi, thr=thr: (d.odds_home >= lo) & (d.odds_home < hi) & (d.ta_hseg_delta >= thr),
                      f"bet HOME: odds_home in [{lo},{hi}) AND signal (home_WR_seg - home_WR_glob)(team_a) >= {thr}"))
# pure buckets and contextual longshots (no team signal -> selection still on B only)
for lo, hi in BUCKETS:
    rules.append((f"H bucket [{lo},{hi})", 'H',
                  lambda d, lo=lo, hi=hi: (d.odds_home >= lo) & (d.odds_home < hi),
                  f"bet HOME: odds_home in [{lo},{hi})"))
for seg in ['DS', 'MS_early', 'MS_mid', 'MS_late', 'FS']:
    rules.append((f"H [4.0,6.0) seg={seg}", 'H',
                  lambda d, seg=seg: (d.odds_home >= 4.0) & (d.odds_home < 6.0) & (d.segment == seg),
                  f"bet HOME: odds_home in [4.0,6.0) AND segment={seg}"))
for sp in ['start', 'mid', 'end']:
    rules.append((f"H [4.0,6.0) segpos={sp}", 'H',
                  lambda d, sp=sp: (d.odds_home >= 4.0) & (d.odds_home < 6.0) & (d.seg_pos == sp),
                  f"bet HOME: odds_home in [4.0,6.0) AND seg_pos={sp}"))
# rebond after home loss
for lo, hi in [(2.0, 3.0), (2.5, 4.0), (2.0, 6.0), (3.0, 6.0)]:
    rules.append((f"H rebond prevL [{lo},{hi})", 'H',
                  lambda d, lo=lo, hi=hi: (d.odds_home >= lo) & (d.odds_home < hi) & (d.home_prev_home_res == 'L'),
                  f"bet HOME: odds_home in [{lo},{hi}) AND team_a lost previous home match"))
# combos edge x rebond / edge x segment
for thr in (0.05, 0.10):
    rules.append((f"H edge>={thr} & prevL [2.0,6.0)", 'H',
                  lambda d, thr=thr: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= thr) & (d.home_prev_home_res == 'L'),
                  f"bet HOME: odds_home in [2.0,6.0) AND edge_home>={thr} AND lost previous home match"))
    for seg in ['MS_mid', 'MS_late', 'FS']:
        rules.append((f"H edge>={thr} seg={seg} [2.0,6.0)", 'H',
                      lambda d, thr=thr, seg=seg: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= thr) & (d.segment == seg),
                      f"bet HOME: odds_home in [2.0,6.0) AND edge_home>={thr} AND segment={seg}"))
# union edge | segdelta
rules.append(("H edge>=0.08|segdelta>=0.08 [2.0,6.0)", 'H',
              lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & ((d.edge_home >= 0.08) | (d.ta_hseg_delta >= 0.08)),
              "bet HOME: odds_home in [2.0,6.0) AND (edge_home>=0.08 OR home segdelta>=0.08)"))

rows = []
for name, side, fn, definition in rules:
    nB, wB, oB, rB = ev(Bs[fn(Bs)], side)
    nC, wC, oC, rC = ev(Cs[fn(Cs)], side)
    rows.append(dict(name=name, side=side, definition=definition,
                     n_B=nB, wr_B=wB, roi_B=rB, n_C=nC, wr_C=wC, cote_C=oC, roi_C=rC))
res = pd.DataFrame(rows)
selected = res[(res.n_B >= 25) & (res.roi_B >= 0.08)].sort_values('roi_B', ascending=False)
print("\n===== NESTED: selected on B (roi_B>=0.08, n_B>=25) -> verdict on C =====")
print(selected[['name', 'n_B', 'wr_B', 'roi_B', 'n_C', 'wr_C', 'cote_C', 'roi_C']].to_string(index=False))
nsel = len(selected)
nwin = (selected.roi_C >= 0.15).sum()
print(f"\nselected={nsel}  survived C with roi>=15%: {nwin}")

# ============================================================ DEPLOYABLE: signals on full 70% train, eval on OOS + stability halves
print("\n===== DEPLOYABLE rules (signals on 70% train) -> OOS with half-split stability =====")
train, oos = df.iloc[:iB], df.iloc[iB:]
hwrT, awrT, hsegdT = team_signals(train)
trS = add_signals(train, hwrT, awrT, hsegdT)
ooS = add_signals(oos, hwrT, awrT, hsegdT)
half = int(len(ooS) * 0.5)
oo1, oo2 = ooS.iloc[:half], ooS.iloc[half:]

final_rules = [
    ("VB-1 home value edge 10 large", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.10),
     "bet HOME: odds_home in [2.0,6.0) AND (train_home_WR(team_a) - 1/odds_home) >= 0.10"),
    ("VB-2 home value edge 10 mid-odds", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.10),
     "bet HOME: odds_home in [2.5,3.5) AND (train_home_WR(team_a) - 1/odds_home) >= 0.10"),
    ("VB-3 home value edge 05 [2.5,3.5)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 3.5) & (d.edge_home >= 0.05),
     "bet HOME: odds_home in [2.5,3.5) AND edge_home >= 0.05"),
    ("VB-4 home segdelta 08", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.ta_hseg_delta >= 0.08),
     "bet HOME: odds_home in [2.0,6.0) AND train (home_WR_segment - home_WR_global)(team_a) >= 0.08 (n_seg>=6)"),
    ("VB-5 union edge|segdelta", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & ((d.edge_home >= 0.10) | (d.ta_hseg_delta >= 0.08)),
     "bet HOME: odds_home in [2.0,6.0) AND (edge_home>=0.10 OR home segdelta>=0.08)"),
    ("VB-6 home longshot global", 'H', lambda d: (d.odds_home >= 4.0) & (d.odds_home < 6.0),
     "bet HOME: odds_home in [4.0,6.0)"),
    ("VB-7 home longshot edge>=0", 'H', lambda d: (d.odds_home >= 4.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.0),
     "bet HOME: odds_home in [4.0,6.0) AND edge_home >= 0"),
    ("VB-8 home edge 10 [2.0,3.0)", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 3.0) & (d.edge_home >= 0.10),
     "bet HOME: odds_home in [2.0,3.0) AND edge_home >= 0.10"),
    ("VB-9 rebond prevL [2.5,4.0)", 'H', lambda d: (d.odds_home >= 2.5) & (d.odds_home < 4.0) & (d.home_prev_home_res == 'L'),
     "bet HOME: odds_home in [2.5,4.0) AND team_a lost previous home match"),
    ("VB-10 edge 10 + prevL", 'H', lambda d: (d.odds_home >= 2.0) & (d.odds_home < 6.0) & (d.edge_home >= 0.10) & (d.home_prev_home_res == 'L'),
     "bet HOME: odds_home in [2.0,6.0) AND edge_home>=0.10 AND lost previous home match"),
]
out = []
for name, side, fn, definition in final_rules:
    nt, wt, ot, rt = ev(trS[fn(trS)], side)
    no, wo, oo_, ro = ev(ooS[fn(ooS)], side)
    n1, w1, _, r1 = ev(oo1[fn(oo1)], side)
    n2, w2, _, r2 = ev(oo2[fn(oo2)], side)
    out.append(dict(name=name, definition=definition, n_train=nt, wr_train=wt, roi_train=rt,
                    n_oos=no, wr_oos=wo, cote_oos=oo_, roi_oos=ro,
                    n_oos1=n1, roi_oos1=r1, n_oos2=n2, roi_oos2=r2))
odf = pd.DataFrame(out)
print(odf[['name', 'n_train', 'wr_train', 'roi_train', 'n_oos', 'wr_oos', 'cote_oos', 'roi_oos', 'roi_oos1', 'roi_oos2']].to_string(index=False))

# ============================================================ ACCURACY MAX (objectif a): WR>=75% OOS
print("\n===== ACCURACY MAX: filtres bas-cote WR (train -> oos) =====")
acc_rules = [
    ("ACC-1 home fav <=1.30", 'H', lambda d: d.odds_home <= 1.30, "bet HOME: odds_home <= 1.30"),
    ("ACC-2 home fav <=1.25", 'H', lambda d: d.odds_home <= 1.25, "bet HOME: odds_home <= 1.25"),
    ("ACC-3 home fav <=1.30 + prevW", 'H', lambda d: (d.odds_home <= 1.30) & (d.home_prev_home_res == 'W'),
     "bet HOME: odds_home <= 1.30 AND won previous home match"),
    ("ACC-4 home fav <=1.40 edge>=0.05", 'H', lambda d: (d.odds_home <= 1.40) & (d.edge_home >= 0.05),
     "bet HOME: odds_home <= 1.40 AND edge_home >= 0.05"),
    ("ACC-5 away fav <=1.40", 'A', lambda d: d.odds_away <= 1.40, "bet AWAY: odds_away <= 1.40"),
    ("ACC-6 home fav <=1.50 edge>=0.08", 'H', lambda d: (d.odds_home <= 1.50) & (d.edge_home >= 0.08),
     "bet HOME: odds_home <= 1.50 AND edge_home >= 0.08"),
]
out2 = []
for name, side, fn, definition in acc_rules:
    nt, wt, ot, rt = ev(trS[fn(trS)], side)
    no, wo, oo_, ro = ev(ooS[fn(ooS)], side)
    out2.append(dict(name=name, definition=definition, n_train=nt, wr_train=wt,
                     n_oos=no, wr_oos=wo, cote_oos=oo_, roi_oos=ro))
print(pd.DataFrame(out2)[['name', 'n_train', 'wr_train', 'n_oos', 'wr_oos', 'cote_oos', 'roi_oos']].to_string(index=False))

odf.to_csv('scripts/_wf_value_buckets2_results.csv', index=False)
print("\nsaved scripts/_wf_value_buckets2_results.csv")
