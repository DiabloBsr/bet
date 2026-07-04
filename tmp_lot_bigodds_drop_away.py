import pandas as pd, numpy as np
from scipy import stats

CSV = r"d:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv"
d = pd.read_csv(CSV)
d['ts'] = pd.to_datetime(d['ts'])

# --- construction face-a-face (1 ligne = 1 match, perspective domicile) ---
h = d[d.venue == 'H'].copy()
a = d[d.venue == 'A'][['ts', 'team', 'opp', 'p1_result', 'p1_margin', 'd_odds', 'odds', 'imp', 'p2_result']].rename(
    columns={'team': 'away_team', 'opp': 'away_opp', 'p1_result': 'away_p1_result', 'p1_margin': 'away_p1_margin',
             'd_odds': 'away_d_odds', 'odds': 'away_odds', 'imp': 'away_imp', 'p2_result': 'away_p2_result'})
m = h.merge(a, left_on=['opp', 'ts'], right_on=['away_team', 'ts'], how='inner')
# reciprocite : la ligne away doit pointer vers la meme equipe domicile (anti double-comptage)
m = m[m.away_opp == m.team]
# purge des cles encore ambigues (2 matchs identiques (ts,team,opp))
m = m[~m.duplicated(['ts', 'team', 'opp'], keep=False)]
print(f"rows_total={len(d)} home_rows={len(h)} matches_merged={len(m)}")

m['away_win'] = (m.gf < m.ga).astype(int)

# --- split chrono sur mediane ts ---
med = m.ts.median()
m['split'] = np.where(m.ts <= med, 'TRAIN', 'TEST')
print(f"median_ts={med} n_train={(m.split=='TRAIN').sum()} n_test={(m.split=='TEST').sum()}")

# --- lot bigodds_drop_away : away_odds >= 2.6, away_d_odds < 0 (chute), en bandes ---
sub = m[(m.away_odds >= 2.6) & m.away_d_odds.notna() & (m.away_d_odds < 0)].copy()
print(f"lot_rows={len(sub)} (train={(sub.split=='TRAIN').sum()}, test={(sub.split=='TEST').sum()})")

odds_bins = [2.6, 3.0, 3.5, 4.25, 5.5, 8.0, np.inf]
drop_bins = [-np.inf, -1.0, -0.6, -0.35, -0.2, -0.1, 0.0]
sub['odds_band'] = pd.cut(sub.away_odds, odds_bins, right=False)
sub['drop_band'] = pd.cut(sub.away_d_odds, drop_bins, right=False)

tr = sub[sub.split == 'TRAIN']
te = sub[sub.split == 'TEST']

rows = []
for (ob, db), g in tr.groupby(['odds_band', 'drop_band'], observed=True):
    n_tr = len(g)
    if n_tr < 150:
        continue
    wr_tr = g.away_win.mean()
    imp_tr = g.away_imp.mean()
    resid_tr = wr_tr - imp_tr
    if abs(resid_tr) < 0.02:
        continue
    gt = te[(te.odds_band == ob) & (te.drop_band == db)]
    n_te = len(gt)
    if n_te == 0:
        continue
    wr_te = gt.away_win.mean()
    imp_te = gt.away_imp.mean()
    resid_te = wr_te - imp_te
    same_sign = np.sign(resid_te) == np.sign(resid_tr)
    alt = 'greater' if resid_tr > 0 else 'less'
    pval = stats.binomtest(int(gt.away_win.sum()), n_te, imp_te, alternative=alt).pvalue
    roi_te = (gt.away_win * gt.away_odds - 1).mean()
    rows.append(dict(cell=f"odds{ob}|drop{db}", n_tr=n_tr, resid_tr=resid_tr,
                     n_te=n_te, resid_te=resid_te, same_sign=same_sign,
                     pval=pval, roi_te=roi_te))

res = pd.DataFrame(rows)
print(f"\ncells_tested={len(res)}")
if len(res):
    # BH-FDR sur toutes les cellules du lot
    res = res.sort_values('pval').reset_index(drop=True)
    k = len(res)
    res['bh_crit'] = 0.05 * (res.index + 1) / k
    passed = res[res.pval <= res.bh_crit]
    max_i = passed.index.max() if len(passed) else -1
    res['fdr_pass'] = res.index <= max_i
    res['survivor'] = res.fdr_pass & res.same_sign & (res.roi_te > 0)
    pd.set_option('display.width', 250)
    print(res.to_string(index=False,
          formatters={'resid_tr': '{:+.4f}'.format, 'resid_te': '{:+.4f}'.format,
                      'pval': '{:.4f}'.format, 'roi_te': '{:+.4f}'.format, 'bh_crit': '{:.5f}'.format}))
    print(f"\nn_survivors={int(res.survivor.sum())}")
    best = res.iloc[0]
    print(f"best_by_pval: {best.cell} p={best.pval:.4f} resid_tr={best.resid_tr:+.4f} resid_te={best.resid_te:+.4f} roi_te={best.roi_te:+.4f} n_te={best.n_te}")
else:
    print("no cell passed the TRAIN filter")

# sanity: calibration globale du lot
for name, g in [('TRAIN', tr), ('TEST', te)]:
    print(f"{name} lot: n={len(g)} win={g.away_win.mean():.4f} imp={g.away_imp.mean():.4f} resid={g.away_win.mean()-g.away_imp.mean():+.4f} roi={(g.away_win*g.away_odds-1).mean():+.4f}")
