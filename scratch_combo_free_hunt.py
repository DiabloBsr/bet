# -*- coding: utf-8 -*-
# LOT combo_free — combos libres 3 dimensions, methode obligatoire (TRAIN n>=150 & |resid|>=0.02,
# TEST meme signe + binomtest + BH-FDR sur tout le lot, survivant -> ROI_test > 0 requis)
import pandas as pd, numpy as np
from scipy import stats

CSV = "d:/AGENTOVA/SAMY/virtual-sports-scraper/data/vfoot_ml/trajectory.csv"
d = pd.read_csv(CSV)
d['ts'] = pd.to_datetime(d.ts)

# ---- construction face-a-face (anti double-comptage) ----
h = d[d.venue == 'H'].copy()
a = d[d.venue == 'A'][['ts', 'team', 'p1_result', 'p1_margin', 'd_odds', 'odds', 'p2_result', 'imp']].copy()
a = a.rename(columns={'team': 'away_team', 'p1_result': 'away_p1_result', 'p1_margin': 'away_p1_margin',
                      'd_odds': 'away_d_odds', 'odds': 'away_odds', 'p2_result': 'away_p2_result',
                      'imp': 'away_imp'})
dup_h = int(h.duplicated(subset=['ts', 'team']).sum())
dup_a = int(a.duplicated(subset=['ts', 'away_team']).sum())
h = h.drop_duplicates(subset=['ts', 'team'], keep='first')
a = a.drop_duplicates(subset=['ts', 'away_team'], keep='first')
m = h.merge(a, left_on=['opp', 'ts'], right_on=['away_team', 'ts'], how='inner')
print("lignes CSV:", len(d), "| lignes H:", len(h), "| dup(ts,team) H/A:", dup_h, dup_a,
      "| matches merges:", len(m))
assert len(m) <= len(h), "double-comptage residuel"

m['home_win'] = (m.gf > m.ga).astype(int)
m['away_win'] = (m.gf < m.ga).astype(int)
print("coherence win==(gf>ga):", float((m.home_win == m.win).mean()))
print("calibration globale: mean(resid home) =", round(float((m.home_win - m.imp).mean()), 5),
      "| mean(away_win - away_imp) =", round(float((m.away_win - m.away_imp).mean()), 5))
print("imp*odds moyen (marge?):", round(float((m.imp * m.odds).mean()), 4))

# ---- dimensions ----
def dirn(x):
    if pd.isna(x):
        return np.nan
    return 'DOWN' if x < 0 else ('UP' if x > 0 else 'FLAT')

m['dir_h'] = m.d_odds.map(dirn)
m['dir_a'] = m.away_d_odds.map(dirn)
bins = [1.0, 1.8, 2.4, 3.2, 4.5, 1000.0]
labels = ['1.0-1.8', '1.8-2.4', '2.4-3.2', '3.2-4.5', '4.5+']
m['oband'] = pd.cut(m.odds, bins=bins, labels=labels, right=False)

# ---- split chrono sur mediane ts ----
med = m.ts.median()
tr = m[m.ts <= med]
te = m[m.ts > med]
print("TRAIN:", len(tr), "| TEST:", len(te), "| mediane ts:", med)

FAMILIES = [
    ('A[p1H x p1A x dirH]', ['p1_result', 'away_p1_result', 'dir_h']),
    ('B[dirH x dirA x bandeH]', ['dir_h', 'dir_a', 'oband']),
]

cands = []
total_cells = 0
for fam, keys in FAMILIES:
    trg = tr.dropna(subset=keys).groupby(keys, observed=True)
    teg = {k: g for k, g in te.dropna(subset=keys).groupby(keys, observed=True)}
    for cell, gtr in trg:
        for side in ('HOME', 'AWAY'):
            total_cells += 1
            if side == 'HOME':
                w_tr, imp_tr = gtr.home_win, gtr.imp
            else:
                w_tr, imp_tr = gtr.away_win, gtr.away_imp
            n_tr = len(gtr)
            resid_tr = float((w_tr - imp_tr).mean())
            if n_tr < 150 or abs(resid_tr) < 0.02:
                continue
            gte = teg.get(cell)
            rec = dict(fam=fam, cell=str(cell), side=side, n_tr=n_tr, resid_tr=resid_tr,
                       n_te=0, resid_te=np.nan, pval=1.0, same_sign=False, roi_te=np.nan,
                       roi_tr=np.nan)
            if side == 'HOME':
                rec['roi_tr'] = float((gtr.home_win * gtr.odds - 1).mean())
            else:
                rec['roi_tr'] = float((gtr.away_win * gtr.away_odds - 1).mean())
            if gte is not None and len(gte) > 0:
                if side == 'HOME':
                    w_te, imp_te, odds_te = gte.home_win, gte.imp, gte.odds
                else:
                    w_te, imp_te, odds_te = gte.away_win, gte.away_odds, gte.away_odds
                    imp_te = gte.away_imp
                n_te = len(gte)
                resid_te = float((w_te - imp_te).mean())
                p0 = float(np.clip(imp_te.mean(), 1e-9, 1 - 1e-9))
                alt = 'greater' if resid_tr > 0 else 'less'
                pval = stats.binomtest(int(w_te.sum()), n_te, p0, alternative=alt).pvalue
                rec.update(n_te=n_te, resid_te=resid_te, pval=float(pval),
                           same_sign=bool(np.sign(resid_te) == np.sign(resid_tr) and resid_te != 0),
                           roi_te=float((w_te * odds_te - 1).mean()))
            cands.append(rec)

print("\ncellules(x cotes) formees TRAIN:", total_cells, "| candidates (n>=150 & |resid|>=0.02):", len(cands))

if cands:
    c = pd.DataFrame(cands).sort_values('pval').reset_index(drop=True)
    # BH-FDR sur toutes les cellules candidates du lot
    nn = len(c)
    order = c.pval.values.argsort()
    ranked = c.pval.values[order]
    q = ranked * nn / (np.arange(nn) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qvals = np.empty(nn)
    qvals[order] = np.clip(q, 0, 1)
    c['qval'] = qvals
    c['fdr_pass'] = (c.qval <= 0.05) & c.same_sign
    c['survivor'] = c.fdr_pass & (c.roi_te > 0)

    pd.set_option('display.width', 250)
    print("\n--- TOP 12 candidates par p-value TEST ---")
    print(c[['fam', 'cell', 'side', 'n_tr', 'resid_tr', 'roi_tr', 'n_te', 'resid_te', 'pval', 'qval',
             'same_sign', 'roi_te', 'fdr_pass', 'survivor']].head(12).to_string(index=False))

    best_train = c.loc[c.resid_tr.abs().idxmax()]
    print("\n--- MEILLEURE CELLULE TRAIN (|resid| max) ---")
    print(best_train.to_string())

    n_fdr = int(c.fdr_pass.sum())
    n_surv = int(c.survivor.sum())
    print("\nFDR-pass (meme signe + q<=0.05):", n_fdr, "| survivants finaux (ROI_test>0):", n_surv)
    if n_surv:
        print(c[c.survivor].to_string(index=False))
    # meilleure cellule train : perf OOS
    print("\nBEST_TRAIN OOS -> n_te=%d resid_te=%s pval=%.4g qval=%.4g roi_te=%s" % (
        best_train.n_te, best_train.resid_te, best_train.pval, best_train.qval, best_train.roi_te))
else:
    print("aucune candidate")
