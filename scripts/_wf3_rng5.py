# -*- coding: utf-8 -*-
"""
WF3 — RNG & TEMPS, partie 5 (clôture)
N) Artefact 0-0/HT manquant -> la corrélation négative MT1/MT2 est-elle réelle ?
O) FTTS '1' : edge exploitable ? walk-forward strict (train 70% / OOS 30%) + conditionnements
P) Minutes par nb de buts de la mi-temps : signature de troncature (cap 3/MT)
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(17)
eng = create_engine(load_settings().db_url)

SQL = """
SELECT e.id AS event_id, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e
JOIN (SELECT event_id, MIN(id) AS first_snap FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots o ON o.id = f.first_snap
JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0'
"""
with eng.connect() as c:
    df = pd.read_sql(text(SQL), c)
df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df['ts'] = pd.to_datetime(df['expected_start'])
df = df.sort_values(['ts', 'event_id']).reset_index(drop=True)
df['em'] = [json.loads(x) if isinstance(x, str) else x for x in df['extra_markets']]
df['total'] = df['score_a'] + df['score_b']
def parse_goals(gj):
    if gj is None or (isinstance(gj, float) and pd.isna(gj)): return None
    try: g = json.loads(gj) if isinstance(gj, str) else gj
    except Exception: return None
    return g if isinstance(g, list) else None
def devig(d):
    items = {k: 1.0 / float(v) for k, v in d.items()}
    t = sum(items.values())
    return {k: v / t for k, v in items.items()}
df['goals'] = [parse_goals(g) for g in df['goals_json']]

# ================================================================
print("=" * 80)
print("N — ARTEFACT : COUVERTURE HT SELON LE SCORE FT")
print("=" * 80)
df['has_ht'] = df['ht_score_a'].notna()
cov = df.groupby(df['total'].clip(0, 6))['has_ht'].agg(['mean', 'size'])
print("Couverture HT par total FT:\n", cov.round(4).to_string())
z00 = df[df['total'] == 0]
print(f"FT 0-0: {len(z00)} matchs, HT renseigné: {z00['has_ht'].sum()} ({z00['has_ht'].mean():.2%})")
# Corr MT1/MT2 en INTÉGRANT les 0-0 manquants (ht=0,sh=0 imputables sans risque)
ht = df[df['has_ht'] | (df['total'] == 0)].copy()
ht['ht_total'] = np.where(ht['has_ht'], ht['ht_score_a'] + ht['ht_score_b'], 0).astype(int)
ht['sh_total'] = (ht['total'] - ht['ht_total']).astype(int)
ok = (ht['sh_total'] >= 0) & (ht['sh_total'] <= 3) & (ht['ht_total'] <= 3)
ht = ht[ok]
ct = pd.crosstab(ht['ht_total'], ht['sh_total'])
chi2, p, dof, _ = stats.chi2_contingency(ct)
r_h, p_h = stats.pearsonr(ht['ht_total'], ht['sh_total'])
print(f"\nAvec 0-0 imputés (n={len(ht)}): chi2={chi2:.2f} p={p:.4f} | corr={r_h:+.4f} (p={p_h:.4f})")
print(ct.to_string())
print("E[sh|ht]:", ht.groupby('ht_total')['sh_total'].mean().round(3).to_dict())

# ================================================================
print("\n" + "=" * 80)
print("O — FTTS '1' (le domicile marque en premier) : WALK-FORWARD")
print("=" * 80)
recs = []
for _, row in df.iterrows():
    m = (row['em'] or {}).get('FTTS')
    if not m or '1' not in m: continue
    if row['total'] > 0 and not row['goals']: continue
    if row['total'] == 0: win = 0
    else:
        first = sorted(row['goals'], key=lambda g: int(g['minute']))[0]
        win = 1 if first['team'] == 'Home' else 0
    recs.append((row['ts'], float(m['1']), win, row['odds_home'], row['odds_away'],
                 float(m.get('2', np.nan))))
ft = pd.DataFrame(recs, columns=['ts', 'odds', 'win', 'oh', 'oa', 'odds2'])
print(f"n={len(ft)} | P(home first)={ft['win'].mean():.4f} | cote moyenne FTTS1={ft['odds'].mean():.3f} "
      f"| implied brute={np.mean(1 / ft['odds']):.4f}")
pnl = ft['win'] * ft['odds'] - 1
print(f"ROI flat global FTTS '1': {pnl.mean() * 100:+.2f}% (SE {pnl.std(ddof=1) / math.sqrt(len(ft)) * 100:.2f}%)")
cut = ft['ts'].quantile(0.7)
tr, te = ft[ft['ts'] <= cut], ft[ft['ts'] > cut]
print(f"train n={len(tr)} ROI={(tr['win'] * tr['odds'] - 1).mean() * 100:+.2f}% | "
      f"OOS n={len(te)} ROI={(te['win'] * te['odds'] - 1).mean() * 100:+.2f}%")
# conditionnements sur train
print("\nConditionnements (train) :")
conds = {
    'oh<1.6 (gros favori dom.)': lambda d: d['oh'] < 1.6,
    'oh<2.0': lambda d: d['oh'] < 2.0,
    'oh>=2.0': lambda d: d['oh'] >= 2.0,
    'odds FTTS1 <1.75': lambda d: d['odds'] < 1.75,
    'odds FTTS1 >=1.75 <2.0': lambda d: (d['odds'] >= 1.75) & (d['odds'] < 2.0),
    'odds FTTS1 >=2.0': lambda d: d['odds'] >= 2.0,
}
best = None
for name, f in conds.items():
    g = tr[f(tr)]
    if len(g) < 200: continue
    roi = (g['win'] * g['odds'] - 1).mean()
    print(f"  {name:>26}: n={len(g)} P={g['win'].mean():.4f} ROI={roi * 100:+.2f}%")
    if best is None or roi > best[1]: best = (name, roi, f)
if best and best[1] > 0.02:
    name, _, f = best
    g = te[f(te)]
    wins = int(g['win'].sum())
    bt = stats.binomtest(wins, len(g), float(np.mean(1 / g['odds'])))
    roi_oos = (g['win'] * g['odds'] - 1).mean()
    print(f"\nOOS [{name}]: n={len(g)} WR={g['win'].mean():.4f} ROI={roi_oos * 100:+.2f}% "
          f"binom_p={bt.pvalue:.4f} cote moy={g['odds'].mean():.3f}")
else:
    print("Aucune condition train avec ROI>+2% -> pas d'edge FTTS robuste.")
# calibration de la cote FTTS1 par tranche (le pricing varie-t-il assez ?)
ft['bucket'] = pd.qcut(ft['odds'], 5, duplicates='drop')
print("\nCalibration par quintile de cote FTTS1:")
print(ft.groupby('bucket', observed=True).apply(
    lambda g: pd.Series({'n': len(g), 'P_obs': g['win'].mean(), 'implied': np.mean(1 / g['odds']),
                         'ROI%': (g['win'] * g['odds'] - 1).mean() * 100}), include_groups=False)
    .round(4).to_string())

# ================================================================
print("\n" + "=" * 80)
print("P — MINUTES PAR NB DE BUTS DE LA MI-TEMPS (signature de cap)")
print("=" * 80)
h1_by_k, h2_by_k = {1: [], 2: [], 3: []}, {1: [], 2: [], 3: []}
for _, row in df.iterrows():
    if row['goals'] is None or not row['has_ht']: continue
    m1 = [int(g['minute']) for g in row['goals'] if int(g['minute']) <= 45]
    m2 = [int(g['minute']) - 45 for g in row['goals'] if int(g['minute']) > 45]
    if len(m1) == row['ht_score_a'] + row['ht_score_b'] and len(m1) in h1_by_k:
        h1_by_k[len(m1)].extend(m1)
    if len(m2) in h2_by_k:
        h2_by_k[len(m2)].extend(m2)
for half, d in [('MT1', h1_by_k), ('MT2', h2_by_k)]:
    print(f"{half}: " + " | ".join(f"k={k}: mean={np.mean(v):.2f} n={len(v)}" for k, v in d.items()))
    ks12 = stats.ks_2samp(d[1], d[2]); ks13 = stats.ks_2samp(d[1], d[3])
    print(f"   KS k=1 vs k=2: D={ks12.statistic:.3f} p={ks12.pvalue:.2e} | "
          f"k=1 vs k=3: D={ks13.statistic:.3f} p={ks13.pvalue:.2e}")
# position du DERNIER but quand k=3 (si cap: dernier but loin de 45 ?)
last3 = []
for _, row in df.iterrows():
    if row['goals'] is None or not row['has_ht']: continue
    m1 = [int(g['minute']) for g in row['goals'] if int(g['minute']) <= 45]
    if len(m1) == 3 and row['ht_score_a'] + row['ht_score_b'] == 3:
        last3.append(max(m1))
print(f"Dernier but des MT1 à 3 buts: mean={np.mean(last3):.1f} (si iid depuis hazard: ~max de 3 tirages)")

print("\nFIN _wf3_rng5.py")
