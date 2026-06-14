# -*- coding: utf-8 -*-
"""
WF3 — RNG & TEMPS, partie 4 (vérifications finales)
J) Corrélation MT1/MT2 sur données PROPRES + partielle (contrôle force du match)
K) FT = convolution de deux mi-temps indépendantes ? (vs obs et vs marché Score exact)
L) 'Minute du premier but' & FTTS recalibrés avec les 0-0 (artefact goals_json)
M) Hazard MT1 : step à la minute 16 ? spike 44-45 ? MT2 plat + spike fin ?
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(13)
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
    if gj is None or (isinstance(gj, float) and np.isnan(gj)): return None
    try: g = json.loads(gj) if isinstance(gj, str) else gj
    except Exception: return None
    return g if isinstance(g, list) else None
df['goals'] = [parse_goals(g) for g in df['goals_json']]

def devig(d, drop_capped=False):
    items = {}
    for k, v in d.items():
        v = float(v)
        if drop_capped and v >= 99.99: continue
        items[k] = 1.0 / v
    t = sum(items.values())
    return {k: vv / t for k, vv in items.items()}

# ================================================================
print("=" * 80)
print("J — MT1 vs MT2 SUR DONNÉES PROPRES")
print("=" * 80)
ht = df[df['ht_score_a'].notna()].copy()
ht['ht_total'] = (ht['ht_score_a'] + ht['ht_score_b']).astype(int)
ht['sh_a'] = ht['score_a'] - ht['ht_score_a']
ht['sh_b'] = ht['score_b'] - ht['ht_score_b']
ht['sh_total'] = (ht['sh_a'] + ht['sh_b']).astype(int)
# cohérence avec goals_json quand dispo
def gj_ok(row):
    if row['goals'] is None: return True
    n1 = sum(1 for g in row['goals'] if int(g['minute']) <= 45)
    return n1 == row['ht_score_a'] + row['ht_score_b'] and len(row['goals']) == row['total']
clean = ht[(ht['sh_total'] >= 0) & (ht['sh_total'] <= 3) & (ht['ht_total'] <= 3)
           & (ht['sh_a'] >= 0) & (ht['sh_b'] >= 0)].copy()
clean = clean[clean.apply(gj_ok, axis=1)]
print(f"HT rows: {len(ht)} -> propres: {len(clean)}")
ct = pd.crosstab(clean['ht_total'], clean['sh_total'])
chi2, p, dof, expd = stats.chi2_contingency(ct)
r_h, p_h = stats.pearsonr(clean['ht_total'], clean['sh_total'])
print(f"chi2 indépendance: {chi2:.2f} dof={dof} p={p:.4f} | corr={r_h:+.4f} (p={p_h:.4f})")
print(ct.to_string())
# partielle: contrôle force attendue (mu HT et mu 2H des marchés devig)
def mu_cs_market(em, name):
    m = em.get(name) if em else None
    if not m: return np.nan
    dv = devig(m)
    mu = 0.0
    for k, pdv in dv.items():
        try:
            a, b = k.split('-'); mu += (int(a) + int(b)) * pdv
        except Exception: pass
    return mu
clean['mu1'] = [mu_cs_market(em, 'Mi-tps CS') for em in clean['em']]
clean['mu2'] = [mu_cs_market(em, '2ème mi-tps - CS') for em in clean['em']]
cc = clean.dropna(subset=['mu1', 'mu2'])
res1 = cc['ht_total'] - cc['mu1']; res2 = cc['sh_total'] - cc['mu2']
r_p, p_p = stats.pearsonr(res1, res2)
print(f"Corr partielle (résidus vs mu marchés): r={r_p:+.4f} (p={p_p:.4f}) n={len(cc)}")
print(f"Calibration: mu1={cc['mu1'].mean():.3f} vs obs={cc['ht_total'].mean():.3f} | "
      f"mu2={cc['mu2'].mean():.3f} vs obs={cc['sh_total'].mean():.3f}")
# E[buts MT2 | buts MT1] propre
print("E[sh_total | ht_total]:", cc.groupby('ht_total')['sh_total'].mean().round(3).to_dict(),
      " (n:", cc.groupby('ht_total')['sh_total'].size().to_dict(), ")")

# ================================================================
print("\n" + "=" * 80)
print("K — SCORE FT = CONVOLUTION DE DEUX MI-TEMPS INDÉPENDANTES ?")
print("=" * 80)
# Par match: P_conv(s) = somme_{h1+h2=s} P1(h1) P2(h2) ; aggrégé vs obs et vs Score exact devig
conv_exp, ft_exp, obs_cnt = {}, {}, {}
n_used = 0
for _, row in df.iterrows():
    em = row['em'] or {}
    if 'Mi-tps CS' not in em or '2ème mi-tps - CS' not in em or 'Score exact' not in em: continue
    d1 = devig(em['Mi-tps CS']); d2 = devig(em['2ème mi-tps - CS'])
    dft = devig(em['Score exact'])
    n_used += 1
    sc = (int(row['score_a']), int(row['score_b']))
    obs_cnt[sc] = obs_cnt.get(sc, 0) + 1
    for k1, p1 in d1.items():
        a1, b1 = map(int, k1.split('-'))
        for k2, p2 in d2.items():
            a2, b2 = map(int, k2.split('-'))
            s = (a1 + a2, b1 + b2)
            conv_exp[s] = conv_exp.get(s, 0.0) + p1 * p2
    for k, pp in dft.items():
        a, b = map(int, k.split('-'))
        ft_exp[(a, b)] = ft_exp.get((a, b), 0.0) + pp
chi_conv = chi_ft = 0.0; n_conv = n_ft = 0
print(f"n={n_used}")
print(f"{'score':>6} {'obs':>5} {'conv':>8} {'z_conv':>7} {'mktFT':>8} {'z_mkt':>7}")
for s in sorted(conv_exp, key=lambda x: -conv_exp[x]):
    o = obs_cnt.get(s, 0); ec = conv_exp[s]; ef = ft_exp.get(s, 0.0)
    if ec >= 5:
        chi_conv += (o - ec) ** 2 / ec; n_conv += 1
    if ef >= 5:
        chi_ft += (o - ef) ** 2 / ef; n_ft += 1
    if ec >= 30:
        zc = (o - ec) / math.sqrt(ec); zf = (o - ef) / math.sqrt(max(ef, 1e-9))
        print(f"{s[0]}-{s[1]:>4} {o:>5} {ec:>8.1f} {zc:>+7.2f} {ef:>8.1f} {zf:>+7.2f}")
print(f"chi2 obs vs CONVOLUTION: {chi_conv:.1f} (cellules e>=5: {n_conv}) "
      f"p={1 - stats.chi2.cdf(chi_conv, n_conv - 1):.3f}")
print(f"chi2 obs vs MARCHÉ FT  : {chi_ft:.1f} (cellules e>=5: {n_ft}) "
      f"p={1 - stats.chi2.cdf(chi_ft, n_ft - 1):.2e}")
mu_conv = sum((a + b) * v for (a, b), v in conv_exp.items()) / n_used
print(f"E[buts] convolution={mu_conv:.3f} vs obs={df['total'].mean():.3f}")

# ================================================================
print("\n" + "=" * 80)
print("L — 'MINUTE DU PREMIER BUT' & FTTS RECALIBRÉS (0-0 inclus)")
print("=" * 80)
# goals_json manquant <=> 0-0 ?
miss = df[df['goals'].isna()]
print(f"goals_json manquant: {len(miss)} matchs, dont total=0: {(miss['total'] == 0).sum()} "
      f"| matchs 0-0 au global: {(df['total'] == 0).sum()}")
sub = df[(df['goals'].notna()) | (df['total'] == 0)]
b_obs, b_exp, nfb = {}, {}, 0
for _, row in sub.iterrows():
    m = (row['em'] or {}).get('Minute du premier but')
    if not m: continue
    nfb += 1
    dv = devig(m)
    fmin = None if row['total'] == 0 else (min((int(g['minute']) for g in row['goals']), default=None)
                                           if row['goals'] else None)
    if fmin is None and row['total'] > 0: nfb -= 1; continue
    for k, pdv in dv.items(): b_exp[k] = b_exp.get(k, 0.0) + pdv
    for k in dv:
        kk = k.strip()
        hit = (fmin is None) if 'pas de but' in kk.lower() else False
        if not hit and fmin is not None and '-' in kk:
            lo_, hi_ = kk.split('-'); hit = int(lo_) <= fmin <= int(hi_)
        if hit: b_obs[k] = b_obs.get(k, 0) + 1
chi_fb = 0.0
for k in sorted(b_exp, key=lambda x: -b_exp[x]):
    o, e = b_obs.get(k, 0), b_exp[k]
    z = (o - e) / math.sqrt(e * (1 - e / nfb))
    chi_fb += (o - e) ** 2 / e
    print(f"  {k:>12}: obs={o:>5} exp={e:>7.1f} z={z:+6.2f}")
print(f"n={nfb} chi2={chi_fb:.1f} dof~6 p={1 - stats.chi2.cdf(chi_fb, 6):.4f}")
# ROI flat par bucket (cotes réelles)
print("ROI flat par bucket (cotes réelles):")
for key in list(b_exp):
    pnl, n = 0.0, 0
    for _, row in sub.iterrows():
        m = (row['em'] or {}).get('Minute du premier but')
        if not m or key not in m: continue
        fmin = None if row['total'] == 0 else (min((int(g['minute']) for g in row['goals']), default=None)
                                               if row['goals'] else None)
        if fmin is None and row['total'] > 0: continue
        kk = key.strip()
        hit = (fmin is None) if 'pas de but' in kk.lower() else \
              (fmin is not None and '-' in kk and int(kk.split('-')[0]) <= fmin <= int(kk.split('-')[1]))
        pnl += (float(m[key]) if hit else 0.0) - 1.0; n += 1
    print(f"  {key:>12}: n={n} ROI={pnl / n * 100:+.2f}%")

# FTTS
f_obs, f_exp, nft = {}, {}, 0
for _, row in sub.iterrows():
    m = (row['em'] or {}).get('FTTS')
    if not m: continue
    if row['total'] > 0 and not row['goals']: continue
    nft += 1
    dv = devig(m)
    if row['total'] == 0: res = 'Pas de but'
    else:
        first = sorted(row['goals'], key=lambda g: int(g['minute']))[0]
        res = '1' if first['team'] == 'Home' else '2'
    for k, pdv in dv.items(): f_exp[k] = f_exp.get(k, 0.0) + pdv
    f_obs[res] = f_obs.get(res, 0) + 1
print(f"\nFTTS (n={nft}):")
for k in f_exp:
    o, e = f_obs.get(k, 0), f_exp[k]
    z = (o - e) / math.sqrt(e * (1 - e / nft))
    print(f"  {k:>12}: obs={o:>5} exp={e:>7.1f} z={z:+6.2f}")

# ================================================================
print("\n" + "=" * 80)
print("M — FORME FINE DU HAZARD")
print("=" * 80)
allmin = np.array([int(g['minute']) for _, row in df.iterrows() if row['goals']
                   for g in row['goals']])
h = np.bincount(allmin, minlength=91)
seg = {'MT1 min 1-15': h[1:16].mean(), 'MT1 min 16-40': h[16:41].mean(),
       'MT1 min 41-45': h[41:46].mean(), 'MT2 min 46-85': h[46:86].mean(),
       'MT2 min 86-90': h[86:91].mean()}
for k, v in seg.items(): print(f"  {k:>15}: {v:.1f} buts/min")
# step à 16 significatif ?
a, b = h[1:16], h[16:41]
t, pt = stats.ttest_ind(a, b)
print(f"t-test 1-15 vs 16-40: t={t:.2f} p={pt:.2e}")
a2, b2 = h[16:41], h[41:46]
print(f"t-test 16-40 vs 41-45: t={stats.ttest_ind(a2, b2).statistic:.2f} p={stats.ttest_ind(a2, b2).pvalue:.4f}")
a3, b3 = h[46:86], h[86:91]
print(f"t-test 46-85 vs 86-90: t={stats.ttest_ind(a3, b3).statistic:.2f} p={stats.ttest_ind(a3, b3).pvalue:.4f}")
# minutes intra-MT1 selon nb de buts MT1 (iid au sein de la MT ?)
ht2 = df[df['ht_score_a'].notna() & df['goals'].notna()].copy()
ht2['ht_total'] = (ht2['ht_score_a'] + ht2['ht_score_b']).astype(int)
g1 = [int(g['minute']) for _, row in ht2[ht2['ht_total'] == 1].iterrows()
      for g in row['goals'] if int(g['minute']) <= 45]
g3 = [int(g['minute']) for _, row in ht2[ht2['ht_total'] == 3].iterrows()
      for g in row['goals'] if int(g['minute']) <= 45]
ks = stats.ks_2samp(g1, g3)
print(f"KS minutes MT1 (matchs 1 but MT1) vs (3 buts MT1): D={ks.statistic:.4f} p={ks.pvalue:.4f} "
      f"(n={len(g1)},{len(g3)})")

print("\nFIN _wf3_rng4.py")
