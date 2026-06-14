# -*- coding: utf-8 -*-
"""
WF3 — ARBITRAGES INTERNES ENTRE MARCHES (cross-market consistency vs grille Score exact)
1. Devig grille CS -> probas fair de tous les marches derivables -> selections sur-cotees (real >= 1.10*fair)
2. Distribution des ecarts par marche (systematiques ?)
3. Walk-forward 70/30 : ROI des sur-cotees (cote <= 20)
4. Sens inverse : les sous-cotees confirment-elles la grille ?
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine

rng = np.random.default_rng(42)
eng = create_engine(load_settings().db_url)

Q = """
SELECT e.id AS event_id, e.round_info, e.team_a, e.team_b, e.expected_start,
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
print(f"events dedup: {N}")

CELLS = [(h, a) for h in range(7) for a in range(7) if h + a <= 6]  # 28 cells


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
        k = int(key)
        return lambda h, a: h + a == k
    if market == '1X2 & Total':
        side, tot = [s.strip() for s in key.split('/')]
        line = float(tot.split()[-1]); over = tot.startswith('>')
        s = {'1': lambda h, a: h > a, 'X': lambda h, a: h == a, '2': lambda h, a: h < a}[side]
        if over:
            return lambda h, a: s(h, a) and h + a > line
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
            s = m[key]
            return lambda h, a: h + a in s
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

# ---------- pass 1: grids ----------
grids_raw, grids_cap, ems = [], [], []
for em_raw in df['extra_markets']:
    em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    ems.append(em)
    cs = em['Score exact']
    raw = np.array([1.0 / cs.get(f"{h}-{a}", 100.0) for h, a in CELLS])
    cap = np.array([cs.get(f"{h}-{a}", 100.0) >= 100.0 for h, a in CELLS])
    grids_raw.append(raw); grids_cap.append(cap)
grids_raw = np.vstack(grids_raw); grids_cap = np.vstack(grids_cap)
p_naive = grids_raw / grids_raw.sum(axis=1, keepdims=True)

score_idx = {c: i for i, c in enumerate(CELLS)}
oog = [(h, a) for h, a in zip(df['score_a'], df['score_b']) if (h, a) not in score_idx]
print(f"results hors grille (total>6): {len(oog)}")
y_cell = np.array([score_idx[(h, a)] for h, a in zip(df['score_a'], df['score_b'])])

# ---------- chronological split 70/30 ----------
split = int(N * 0.70)
tr = np.arange(N) < split
print(f"train {split} events (jusqu'a {df['expected_start'].iloc[split-1]}), oos {N - split}")

# ---------- per-(cell,capped) calibration on TRAIN -> p_calib ----------
print("\n=== CALIBRATION GRILLE (train) : implied naive vs frequence empirique par cellule ===")
corr = np.ones((28, 2))
rows_cal = []
for ci, (h, a) in enumerate(CELLS):
    for capped in (0, 1):
        msk = tr & (grids_cap[:, ci] == bool(capped))
        n = int(msk.sum())
        if n < 30:
            continue
        imp = p_naive[msk, ci].mean()
        emp = (y_cell[msk] == ci).mean()
        c = emp / imp if imp > 0 else 1.0
        corr[ci, capped] = c
        k = int((y_cell[msk] == ci).sum())
        pv = stats.binomtest(k, n, imp).pvalue if 0 < imp < 1 else np.nan
        rows_cal.append((f"{h}-{a}", capped, n, imp, emp, c, pv))
cal = pd.DataFrame(rows_cal, columns=['cell', 'capped', 'n', 'implied', 'empirique', 'corr', 'p_binom'])
print(cal.sort_values(['capped', 'cell']).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

p_adj = p_naive * corr[np.arange(28)[None, :], grids_cap.astype(int)]
p_calib = p_adj / p_adj.sum(axis=1, keepdims=True)

ll_naive = -np.log(np.clip(p_naive[np.arange(N), y_cell], 1e-9, 1)).mean()
ll_calib = -np.log(np.clip(p_calib[np.arange(N), y_cell], 1e-9, 1)).mean()
print(f"\nlogloss score-exact  naive={ll_naive:.4f}  calib(train-fit)={ll_calib:.4f}  (uniforme={math.log(28):.4f})")

# ---------- pass 2: selections ----------
recs = []
for i in range(N):
    em = ems[i]; h_res, a_res = int(df['score_a'].iat[i]), int(df['score_b'].iat[i])
    pn, pc = p_naive[i], p_calib[i]

    def add(market, key, odds):
        if odds is None or odds <= 1.0:
            return
        f = preds_for(market, key)
        mask = np.array([f(h, a) for h, a in CELLS])
        recs.append((i, market, key, float(odds), pn[mask].sum(), pc[mask].sum(), int(f(h_res, a_res))))

    add('1X2', '1', df['odds_home'].iat[i]); add('1X2', 'X', df['odds_draw'].iat[i]); add('1X2', '2', df['odds_away'].iat[i])
    for mk in DERIVED:
        sels = em.get(mk)
        if not sels:
            continue
        for k, o in sels.items():
            add(mk, k, o)
sel = pd.DataFrame(recs, columns=['ev', 'market', 'key', 'odds', 'p_naive', 'p_calib', 'won'])
sel['ratio_n'] = sel['odds'] * sel['p_naive']
sel['ratio_c'] = sel['odds'] * sel['p_calib']
sel['train'] = sel['ev'] < split
print(f"\nselections totales: {len(sel)}")

# ---------- PART 2 ----------
print("\n=== PART 2 — ECARTS real/fair PAR MARCHE (odds<100) ===")
s2 = sel[sel['odds'] < 100].copy()
g = s2.groupby('market').agg(n=('ratio_n', 'size'),
                             ratio_n_mean=('ratio_n', 'mean'), ratio_n_med=('ratio_n', 'median'),
                             ratio_n_std=('ratio_n', 'std'),
                             ratio_c_mean=('ratio_c', 'mean'), ratio_c_med=('ratio_c', 'median'),
                             pct_over110_n=('ratio_n', lambda x: (x >= 1.10).mean() * 100),
                             pct_over110_c=('ratio_c', lambda x: (x >= 1.10).mean() * 100))
print(g.sort_values('ratio_n_med', ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))

print("\n--- biais systematiques par (marche, selection), tri par median ratio calib ---")
gk = s2.groupby(['market', 'key']).agg(n=('ratio_n', 'size'), med_n=('ratio_n', 'median'),
                                       med_c=('ratio_c', 'median'), std_n=('ratio_n', 'std'))
gk = gk.sort_values('med_c', ascending=False)
print(gk.head(25).to_string(float_format=lambda x: f"{x:.4f}"))
print("   ... bottom 10:")
print(gk.tail(10).to_string(float_format=lambda x: f"{x:.4f}"))

print("\n--- marche = fonction deterministe de la grille ? (R2 implied~fair_naive, sd residus) ---")
for mk, gg in s2.groupby('market'):
    if len(gg) < 50:
        continue
    x = gg['p_naive'].values; y = 1.0 / gg['odds'].values
    r = np.corrcoef(x, y)[0, 1]
    resid = y - np.poly1d(np.polyfit(x, y, 1))(x)
    print(f"{mk:28s} R2={r*r:.5f}  sd(resid)={resid.std():.5f}")

# ---------- PART 3 : walk-forward ----------
print("\n=== PART 3 — WALK-FORWARD : sur-cotees (ratio >= 1.10, cote <= 20) ===")


def roi_report(b, label):
    if len(b) == 0:
        print(f"{label}: 0 bets")
        return None
    profit = b['won'] * b['odds'] - 1.0
    roi = profit.mean()
    pb = b.assign(pf=profit).groupby('ev')['pf'].agg(['sum', 'count'])
    boots = []
    sv, cv = pb['sum'].values, pb['count'].values
    for _ in range(2000):
        idx = rng.integers(0, len(pb), len(pb))
        boots.append(sv[idx].sum() / cv[idx].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p_neg = (np.array(boots) <= 0).mean()
    print(f"{label}: n={len(b)} wr={b['won'].mean()*100:.1f}% cote_moy={b['odds'].mean():.2f} "
          f"ROI={roi*100:+.1f}% CI95=[{lo*100:+.1f}%,{hi*100:+.1f}%] P(ROI<=0)={p_neg:.4f}")
    return roi


for variant in ['ratio_n', 'ratio_c']:
    print(f"\n--- variante fair = {'naive devig' if variant == 'ratio_n' else 'calib (corrections train)'} ---")
    bets = sel[(sel[variant] >= 1.10) & (sel['odds'] <= 20)]
    btr, boo = bets[bets['train']], bets[~bets['train']]
    print("TRAIN:"); roi_report(btr, "  global")
    per_mk_tr = btr.assign(pf=btr['won'] * btr['odds'] - 1).groupby('market')['pf'].agg(['mean', 'count'])
    print(per_mk_tr.rename(columns={'mean': 'ROI_train', 'count': 'n'}).to_string(float_format=lambda x: f"{x:+.3f}"))
    print("OOS:"); roi_report(boo, "  global")
    per_mk_oo = boo.assign(pf=boo['won'] * boo['odds'] - 1).groupby('market')['pf'].agg(['mean', 'count'])
    print(per_mk_oo.rename(columns={'mean': 'ROI_oos', 'count': 'n'}).to_string(float_format=lambda x: f"{x:+.3f}"))
    good = per_mk_tr[(per_mk_tr['mean'] > 0) & (per_mk_tr['count'] >= 100)].index.tolist()
    print(f"marches retenus sur train (ROI>0, n>=100): {good}")
    if good:
        roi_report(boo[boo['market'].isin(good)], "  OOS (marches train-positifs)")

# ---------- PART 4 : sous-cotees ----------
print("\n=== PART 4 — SOUS-COTEES : la grille a-t-elle raison contre le marche ? ===")
imp = 1.0 / s2['odds']
fair_sum = s2.groupby(['ev', 'market'])['p_naive'].transform('sum')
imp_sum = imp.groupby([s2['ev'], s2['market']]).transform('sum')
s2['p_mkt'] = imp / imp_sum * fair_sum

for label, msk in [("SOUS-cotees ratio_c<=0.909 odds<=20", (s2['ratio_c'] <= 1 / 1.10) & (s2['odds'] <= 20)),
                   ("SUR-cotees  ratio_c>=1.10  odds<=20", (s2['ratio_c'] >= 1.10) & (s2['odds'] <= 20)),
                   ("toutes odds<=20", s2['odds'] <= 20)]:
    b = s2[msk]
    for name, bb in [("ALL", b), ("OOS", b[~b['train']])]:
        if len(bb) < 30:
            continue
        W = int(bb['won'].sum())
        line = f"{label} [{name}] n={len(bb)} W={W}"
        for pcol in ['p_naive', 'p_calib', 'p_mkt']:
            E = bb[pcol].sum(); V = (bb[pcol] * (1 - bb[pcol])).sum()
            z = (W - E) / math.sqrt(V)
            line += f" | {pcol}: E={E:.0f} z={z:+.2f}"
        print(line)

print()
for name, bb in [("ALL", s2), ("OOS", s2[~s2['train']])]:
    out = {}
    for pcol in ['p_naive', 'p_calib', 'p_mkt']:
        p = np.clip(bb[pcol], 1e-6, 1 - 1e-6)
        out[pcol] = float(-(bb['won'] * np.log(p) + (1 - bb['won']) * np.log(1 - p)).mean())
    print(f"logloss selections [{name}]: " + "  ".join(f"{k}={v:.5f}" for k, v in out.items()))
