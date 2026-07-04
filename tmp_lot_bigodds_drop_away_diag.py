import pandas as pd, numpy as np

CSV = r"d:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv"
d = pd.read_csv(CSV)
d['ts'] = pd.to_datetime(d['ts'])

h = d[d.venue == 'H'].copy()
a = d[d.venue == 'A'][['ts', 'team', 'opp', 'd_odds', 'odds', 'imp']].rename(
    columns={'team': 'away_team', 'opp': 'away_opp', 'd_odds': 'away_d_odds',
             'odds': 'away_odds', 'imp': 'away_imp'})
m = h.merge(a, left_on=['opp', 'ts'], right_on=['away_team', 'ts'], how='inner')
m = m[m.away_opp == m.team]
m = m[~m.duplicated(['ts', 'team', 'opp'], keep=False)]
m['away_win'] = (m.gf < m.ga).astype(int)
med = m.ts.median()
m['split'] = np.where(m.ts <= med, 'TRAIN', 'TEST')

sub = m[(m.away_odds >= 2.6) & m.away_d_odds.notna() & (m.away_d_odds < 0)].copy()
odds_bins = [2.6, 3.0, 3.5, 4.25, 5.5, 8.0, np.inf]
drop_bins = [-np.inf, -1.0, -0.6, -0.35, -0.2, -0.1, 0.0]
sub['odds_band'] = pd.cut(sub.away_odds, odds_bins, right=False)
sub['drop_band'] = pd.cut(sub.away_d_odds, drop_bins, right=False)
tr = sub[sub.split == 'TRAIN']

rows = []
for (ob, db), g in tr.groupby(['odds_band', 'drop_band'], observed=False):
    n = len(g)
    resid = (g.away_win.mean() - g.away_imp.mean()) if n else np.nan
    rows.append(dict(odds_band=str(ob), drop_band=str(db), n_train=n,
                     resid_train=resid,
                     gate=('PASS' if (n >= 150 and abs(resid) >= 0.02) else
                           ('n<150' if n < 150 else '|resid|<0.02'))))
res = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(res.to_string(index=False, formatters={'resid_train': lambda v: '' if pd.isna(v) else f'{v:+.4f}'}))
print(f"\ncells with n_train>=150: {int((res.n_train>=150).sum())} / {len(res)}")
big = res[res.n_train >= 150]
if len(big):
    print(f"max |resid_train| among n>=150 cells: {big.resid_train.abs().max():.4f}")
