# -*- coding: utf-8 -*-
"""
WF3 cross-market — pass 2 : robustesse
A. Les sur-cotees naives impliquent-elles des cellules cappees ? (artefact)
B. Sweep de seuil par marche (train) -> OOS, variante calib
C. Vig ladder : ratio calib median par (marche, cle) = marge par selection, stabilite train/oos
D. ROI de TOUT parier par marche (sanity : = -vig)
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine

rng = np.random.default_rng(7)
eng = create_engine(load_settings().db_url)

Q = """
SELECT e.id AS event_id, e.team_a, e.team_b, e.expected_start,
       os.id AS snap_id, os.odds_home, os.odds_draw, os.odds_away, os.extra_markets,
       r.score_a, r.score_b
FROM events e
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id = e.id
JOIN odds_snapshots os ON os.id = m.mid
JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
df = pd.read_sql(Q, eng)
df = df.sort_values('snap_id').drop_duplicates(['team_a', 'team_b', 'expected_start'], keep='first')
df = df.sort_values('expected_start').reset_index(drop=True)
N = len(df)
CELLS = [(h, a) for h in range(7) for a in range(7) if h + a <= 6]
score_idx = {c: i for i, c in enumerate(CELLS)}

from importlib import import_module
sys.path.insert(0, 'scripts')
cm = import_module('_wf3_crossmarket') if False else None

# redefine predicates (same as main script)
def preds_for(market, key):
    if market == '1X2':
        return {'1': lambda h, a: h > a, 'X': lambda h, a: h == a, '2': lambda h, a: h < a}[key]
    if market == 'Double Chance':
        return {'1X': lambda h, a: h >= a, 'X2': lambda h, a: h <= a, '12': lambda h, a: h != a}[key]
    if market == '+/-':
        line = float(key.split()[-1])
        return (lambda h, a: h + a > line) if key.startswith('>') else (lambda h, a: h + a < line)
    if market == 'G/NG':
        return (lambda h, a: h > 0 and a > 0) if key == 'Oui' else (lambda h, a: not (h > 0 and a > 0))
    if market == 'Total de buts':
        k = int(key); return lambda h, a: h + a == k
    if market == '1X2 & Total':
        side, tot = [s.strip() for s in key.split('/')]
        line = float(tot.split()[-1]); over = tot.startswith('>')
        s = {'1': lambda h, a: h > a, 'X': lambda h, a: h == a, '2': lambda h, a: h < a}[side]
        if over: return lambda h, a: s(h, a) and h + a > line
        return lambda h, a: s(h, a) and h + a < line
    if market == 'Total equipe domicile':
        line = float(key.split()[-1])
        return (lambda h, a: h > line) if key.startswith('>') else (lambda h, a: h < line)
    if market == 'Total equipe extérieur':
        line = float(key.split()[-1])
        return (lambda h, a: a > line) if key.startswith('>') else (lambda h, a: a < line)
    if market == 'Pair/Impair':
        return (lambda h, a: (h + a) % 2 == 0) if key == 'Pair' else (lambda h, a: (h + a) % 2 == 1)
    if market == 'Multi-Buts':
        m = {'Le total de buts est de 0, 1 ou 2': {0, 1, 2},
             'Le total de buts est de 1, 2 ou 3': {1, 2, 3},
             'Le total de buts est de 2, 3 ou 4': {2, 3, 4}}
        if key in m:
            s = m[key]; return lambda h, a: h + a in s
        return lambda h, a: h + a > 4
    if market == 'G/NG equipe domicile':
        return (lambda h, a: h > 0) if key == 'Oui' else (lambda h, a: h == 0)
    if market == 'G/NG equipe extérieur':
        return (lambda h, a: a > 0) if key == 'Oui' else (lambda h, a: a == 0)
    if market == '1X2 & G/NG':
        m = {'1 gagne et les deux équipes marquent': lambda h, a: h > a and a > 0,
             '1 gagne et seulement  1  marque':      lambda h, a: h > a and a == 0,
             'X et les deux équipes marquent':       lambda h, a: h == a and h > 0,
             'X et aucun but':                       lambda h, a: h == 0 and a == 0,
             '2 gagne et les deux équipes marquent': lambda h, a: a > h and h > 0,
             '2 gagne et seulement 2 marque':        lambda h, a: a > h and h == 0}
        return m[key]
    raise KeyError(market)

DERIVED = ['Double Chance', '+/-', 'G/NG', 'Total de buts', '1X2 & Total',
           'Total equipe domicile', 'Total equipe extérieur', 'Pair/Impair',
           'Multi-Buts', 'G/NG equipe domicile', 'G/NG equipe extérieur', '1X2 & G/NG']

grids_raw, grids_cap, ems = [], [], []
for em_raw in df['extra_markets']:
    em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    ems.append(em)
    cs = em['Score exact']
    grids_raw.append(np.array([1.0 / cs.get(f"{h}-{a}", 100.0) for h, a in CELLS]))
    grids_cap.append(np.array([cs.get(f"{h}-{a}", 100.0) >= 100.0 for h, a in CELLS]))
grids_raw = np.vstack(grids_raw); grids_cap = np.vstack(grids_cap)
p_naive = grids_raw / grids_raw.sum(axis=1, keepdims=True)
y_cell = np.array([score_idx[(h, a)] for h, a in zip(df['score_a'], df['score_b'])])

split = int(N * 0.70)
tr = np.arange(N) < split

corr = np.ones((28, 2))
for ci in range(28):
    for capped in (0, 1):
        msk = tr & (grids_cap[:, ci] == bool(capped))
        if msk.sum() < 30: continue
        imp = p_naive[msk, ci].mean()
        emp = (y_cell[msk] == ci).mean()
        if imp > 0: corr[ci, capped] = emp / imp
p_adj = p_naive * corr[np.arange(28)[None, :], grids_cap.astype(int)]
p_calib = p_adj / p_adj.sum(axis=1, keepdims=True)

recs = []
for i in range(N):
    em = ems[i]; h_res, a_res = int(df['score_a'].iat[i]), int(df['score_b'].iat[i])
    pn, pc, gc = p_naive[i], p_calib[i], grids_cap[i]

    def add(market, key, odds):
        if odds is None or odds <= 1.0: return
        f = preds_for(market, key)
        mask = np.array([f(h, a) for h, a in CELLS])
        ncap = int((mask & gc).sum())
        capshare = float(pn[mask & gc].sum() / max(pn[mask].sum(), 1e-12))
        recs.append((i, market, key, float(odds), pn[mask].sum(), pc[mask].sum(),
                     int(f(h_res, a_res)), ncap, capshare))

    add('1X2', '1', df['odds_home'].iat[i]); add('1X2', 'X', df['odds_draw'].iat[i]); add('1X2', '2', df['odds_away'].iat[i])
    for mk in DERIVED:
        sels = em.get(mk)
        if not sels: continue
        for k, o in sels.items():
            add(mk, k, o)
sel = pd.DataFrame(recs, columns=['ev', 'market', 'key', 'odds', 'p_naive', 'p_calib', 'won', 'ncap', 'capshare'])
sel['ratio_n'] = sel['odds'] * sel['p_naive']
sel['ratio_c'] = sel['odds'] * sel['p_calib']
sel['train'] = sel['ev'] < split

# ---------- A. cap involvement in naive over-priced ----------
print("=== A. Sur-cotees naives (ratio_n>=1.10, odds<=20) : implication des cellules cappees ===")
op = sel[(sel['ratio_n'] >= 1.10) & (sel['odds'] <= 20) & (sel['odds'] < 100)]
print(f"n={len(op)}  avec >=1 cellule cappee: {(op['ncap']>0).mean()*100:.1f}%  "
      f"part mediane de proba issue de cellules cappees: {op['capshare'].median()*100:.1f}%")
print(f"(reference toutes selections odds<=20: cap-involved {(sel[sel['odds']<=20]['ncap']>0).mean()*100:.1f}%, "
      f"capshare med {sel[sel['odds']<=20]['capshare'].median()*100:.2f}%)")
op_nocap = sel[(sel['ratio_n'] >= 1.10) & (sel['odds'] <= 20) & (sel['capshare'] < 0.05)]
print(f"sur-cotees naives avec capshare<5%: n={len(op_nocap)}")

# ---------- B. threshold sweep (calib) train->OOS ----------
print("\n=== B. Sweep de seuil (variante calib), selection sur train, eval OOS ===")
results = []
for thr in [1.00, 1.02, 1.04, 1.06, 1.08, 1.10]:
    b = sel[(sel['ratio_c'] >= thr) & (sel['odds'] <= 20)]
    btr, boo = b[b['train']], b[~b['train']]
    if len(btr) == 0: continue
    g_tr = btr.assign(pf=btr['won']*btr['odds']-1).groupby('market')['pf'].agg(['mean', 'count'])
    good = g_tr[(g_tr['mean'] > 0.02) & (g_tr['count'] >= 200)].index.tolist()
    roi_tr = (btr['won']*btr['odds']-1).mean()
    roi_oo = (boo['won']*boo['odds']-1).mean() if len(boo) else np.nan
    sub = boo[boo['market'].isin(good)] if good else boo.iloc[0:0]
    roi_sub = (sub['won']*sub['odds']-1).mean() if len(sub) else np.nan
    print(f"thr={thr:.2f}: train n={len(btr)} ROI={roi_tr*100:+.1f}% | OOS n={len(boo)} ROI={roi_oo*100:+.1f}% | "
          f"marches train-pos {good} -> OOS n={len(sub)} ROI={(roi_sub*100 if len(sub) else float('nan')):+.1f}%")

# ---------- C. vig ladder stability ----------
print("\n=== C. VIG LADDER : ratio_c median par (marche, cle), train vs OOS (stabilite) ===")
lad = sel[sel['odds'] < 100].groupby(['market', 'key']).apply(
    lambda g: pd.Series({
        'n': len(g),
        'med_train': g.loc[g['train'], 'ratio_c'].median(),
        'med_oos': g.loc[~g['train'], 'ratio_c'].median(),
        'iqr': g['ratio_c'].quantile(0.75) - g['ratio_c'].quantile(0.25),
    }), include_groups=False)
lad['delta'] = lad['med_oos'] - lad['med_train']
lad = lad.sort_values('med_oos', ascending=False)
print(lad.to_string(float_format=lambda x: f"{x:.4f}"))

# ---------- D. ROI tout parier par marche (OOS) ----------
print("\n=== D. ROI 'tout parier' par marche, OOS, odds<=20 (= -vig effectif realise) ===")
allb = sel[(~sel['train']) & (sel['odds'] <= 20)]
gd = allb.assign(pf=allb['won']*allb['odds']-1).groupby('market').agg(
    n=('pf', 'size'), roi=('pf', 'mean'), cote=('odds', 'mean'))
print(gd.sort_values('roi', ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

# best single selections by OOS ROI with n
print("\n--- top (marche,cle) par ROI OOS (n>=300) ---")
gk = allb.assign(pf=allb['won']*allb['odds']-1).groupby(['market', 'key']).agg(
    n=('pf', 'size'), roi=('pf', 'mean'), wr=('won', 'mean'), cote=('odds', 'mean'))
gk = gk[gk['n'] >= 300].sort_values('roi', ascending=False)
print(gk.head(12).to_string(float_format=lambda x: f"{x:.4f}"))
