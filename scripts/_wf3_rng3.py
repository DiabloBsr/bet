# -*- coding: utf-8 -*-
"""
WF3 — RNG & TEMPS, partie 3
G) Structure par mi-temps : cap 3 buts/MT ? indépendance MT1 vs MT2 ? calibration CS MT1/MT2
H) Minutes : hazard par MT, collisions même minute vs iid, KS selon total, home vs away, 1er but
I) Scan global de TOUS les marchés : calibration devig + ROI flat (cotes réelles) + walk-forward
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(11)
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
def parse_goals(gj):
    if not gj: return None
    try: g = json.loads(gj) if isinstance(gj, str) else gj
    except Exception: return None
    return g if isinstance(g, list) else None
df['goals'] = [parse_goals(g) for g in df['goals_json']]
df['total'] = df['score_a'] + df['score_b']
print(f"Matchs: {len(df)} | avec HT: {df['ht_score_a'].notna().sum()} | avec goals_json: {df['goals'].notna().sum()}")

# ================================================================
print("\n" + "=" * 80)
print("G — STRUCTURE PAR MI-TEMPS")
print("=" * 80)
ht = df[df['ht_score_a'].notna()].copy()
ht['ht_total'] = ht['ht_score_a'] + ht['ht_score_b']
ht['sh_a'] = ht['score_a'] - ht['ht_score_a']
ht['sh_b'] = ht['score_b'] - ht['ht_score_b']
ht['sh_total'] = ht['sh_a'] + ht['sh_b']
print(f"n avec HT: {len(ht)}")
print(f"max buts MT1: {ht['ht_total'].max()} | max buts MT2: {ht['sh_total'].max()} "
      f"| négatifs MT2 (incohérences): {(ht['sh_total'] < 0).sum()}")
print("Distribution buts MT1:", np.bincount(ht['ht_total'].astype(int).clip(0, 9)).tolist())
print("Distribution buts MT2:", np.bincount(ht['sh_total'].astype(int).clip(0, 9)).tolist())

# Indépendance MT1 vs MT2 (totaux 0..3) — attendu = produit des marges empiriques
ct = pd.crosstab(ht['ht_total'].clip(0, 3), ht['sh_total'].clip(0, 3))
chi2, p, dof, expd = stats.chi2_contingency(ct)
print(f"\nIndépendance totaux MT1 x MT2 (table 4x4): chi2={chi2:.2f} dof={dof} p={p:.4f}")
print(ct.to_string())
# corrélation
r_h, p_h = stats.pearsonr(ht['ht_total'], ht['sh_total'])
print(f"corr(buts MT1, buts MT2) = {r_h:+.4f} (p={p_h:.4f})")

# Calibration des marchés CS MT1 et CS MT2
def devig(d, drop_capped=False):
    items = {}
    for k, v in d.items():
        v = float(v)
        if drop_capped and v >= 99.99: continue
        items[k] = 1.0 / v
    t = sum(items.values())
    return {k: v / t for k, v in items.items()}

for mname, sa, sb in [('Mi-tps CS', 'ht_score_a', 'ht_score_b'), ('2ème mi-tps - CS', 'sh_a', 'sh_b')]:
    obs_c, exp_c = {}, {}
    n = 0
    for _, row in ht.iterrows():
        m = row['em'].get(mname) if row['em'] else None
        if not m: continue
        n += 1
        dv = devig(m)
        key = f"{int(row[sa])}-{int(row[sb])}"
        obs_c[key] = obs_c.get(key, 0) + 1
        for k, pdv in dv.items(): exp_c[k] = exp_c.get(k, 0.0) + pdv
    print(f"\n{mname} (n={n}) — obs vs devig:")
    chi = 0.0; cnt = 0
    for k in sorted(exp_c, key=lambda x: -exp_c[x]):
        o, e = obs_c.get(k, 0), exp_c[k]
        if e >= 5: chi += (o - e) ** 2 / e; cnt += 1
        z = (o - e) / math.sqrt(max(e, 1e-9))
        print(f"  {k:>4}: obs={o:>5} exp={e:>7.1f} z={z:+6.2f}")
    print(f"  chi2={chi:.1f} dof~{cnt - 1} p={1 - stats.chi2.cdf(chi, cnt - 1):.2e}")

# MT2 conditionnel à MT1 : le marché pré-match '2ème mi-tps - CS' reste-t-il calibré après MT1 chargée ?
# (si moteur = 2 tirages indépendants, oui)

# ================================================================
print("\n" + "=" * 80)
print("H — MINUTES : MODÈLE TEMPOREL")
print("=" * 80)
gdf = df[df['goals'].notna()].copy()
allmin, per_match = [], []
for _, row in gdf.iterrows():
    ms = sorted(int(g['minute']) for g in row['goals'])
    per_match.append(ms); allmin.extend(ms)
allmin = np.array(allmin)
m1 = allmin[allmin <= 45]; m2 = allmin[allmin > 45]
print(f"buts MT1={len(m1)} MT2={len(m2)} ratio={len(m2) / len(m1):.3f}")
h1 = np.bincount(m1, minlength=46)[1:46]
h2 = np.bincount(m2 - 45, minlength=46)[1:46]
print("MT1 par minute (1-45):", h1.tolist())
print("MT2 par minute (46-90):", h2.tolist())
# formes des deux mi-temps comparées (KS sur minute-within-half)
ks_h = stats.ks_2samp(m1, m2 - 45)
print(f"KS forme MT1 vs MT2 (minute intra-MT): D={ks_h.statistic:.4f} p={ks_h.pvalue:.2e}")
sl1 = stats.linregress(np.arange(1, 46), h1)
sl2 = stats.linregress(np.arange(1, 46), h2)
print(f"Pente MT1: {sl1.slope:+.2f}/min (p={sl1.pvalue:.1e}) | Pente MT2: {sl2.slope:+.2f}/min (p={sl2.pvalue:.1e})")

# Collisions même minute vs tirage iid (nul simulé depuis la distribution empirique des minutes)
n_coll_obs = sum(len(ms) - len(set(ms)) for ms in per_match)
pmin = np.bincount(allmin, minlength=91)[1:91].astype(float); pmin /= pmin.sum()
counts = np.array([len(ms) for ms in per_match])
NS = 400
coll_sim = np.zeros(NS)
for s in range(NS):
    tot = 0
    for k in counts[counts >= 2]:
        d = rng.choice(90, size=k, p=pmin)
        tot += k - len(np.unique(d))
    coll_sim[s] = tot
pv_coll = 2 * min((coll_sim >= n_coll_obs).mean(), (coll_sim <= n_coll_obs).mean())
print(f"Collisions même minute: obs={n_coll_obs} | iid null: {coll_sim.mean():.1f} "
      f"[{np.percentile(coll_sim, 2.5):.0f},{np.percentile(coll_sim, 97.5):.0f}] p={pv_coll:.4f}")

# Minutes selon le total du match : iid => même distribution
lo = [m for ms, c in zip(per_match, counts) if 1 <= c <= 2 for m in ms]
hi = [m for ms, c in zip(per_match, counts) if c >= 5 for m in ms]
ks_t = stats.ks_2samp(lo, hi)
print(f"KS minutes (matchs 1-2 buts) vs (matchs 5+ buts): D={ks_t.statistic:.4f} p={ks_t.pvalue:.4f} "
      f"(mean {np.mean(lo):.1f} vs {np.mean(hi):.1f})")

# Home vs away
hm = [int(g['minute']) for _, row in gdf.iterrows() for g in row['goals'] if g['team'] == 'Home']
am = [int(g['minute']) for _, row in gdf.iterrows() for g in row['goals'] if g['team'] == 'Away']
ks_ha = stats.ks_2samp(hm, am)
print(f"KS minutes Home vs Away: D={ks_ha.statistic:.4f} p={ks_ha.pvalue:.4f}")

# Marché 'Minute du premier but'
print("\nCalibration 'Minute du premier but':")
b_obs, b_exp, nfb = {}, {}, 0
for _, row in gdf.iterrows():
    m = row['em'].get('Minute du premier but') if row['em'] else None
    if not m: continue
    nfb += 1
    dv = devig(m)
    fmin = min((int(g['minute']) for g in row['goals']), default=None)
    for k, pdv in dv.items(): b_exp[k] = b_exp.get(k, 0.0) + pdv
    for k in dv:
        kk = k.strip()
        hit = (fmin is None) if 'pas de but' in kk.lower() else False
        if not hit and fmin is not None and '-' in kk:
            try:
                lo_, hi_ = kk.split('-'); hit = int(lo_) <= fmin <= int(hi_)
            except Exception: hit = False
        if hit: b_obs[k] = b_obs.get(k, 0) + 1
for k in sorted(b_exp, key=lambda x: -b_exp[x]):
    o, e = b_obs.get(k, 0), b_exp[k]
    z = (o - e) / math.sqrt(e * (1 - e / nfb))
    print(f"  {k:>12}: obs={o:>5} exp={e:>7.1f} z={z:+6.2f}")

# ================================================================
print("\n" + "=" * 80)
print("I — SCAN GLOBAL DES MARCHÉS : CALIBRATION + ROI FLAT (cotes réelles)")
print("=" * 80)

def settle(market, key, row):
    """True si l'outcome gagne, False sinon, None si non-réglable."""
    sa, sb = int(row['score_a']), int(row['score_b'])
    tot = sa + sb
    ha = row['ht_score_a']; hb = row['ht_score_b']
    has_ht = pd.notna(ha)
    if has_ht: ha, hb = int(ha), int(hb)
    goals = row['goals']
    k = key.strip()
    def sign(a, b): return '1' if a > b else ('2' if a < b else 'X')
    if market == '1X2': return {'1': sa > sb, 'X': sa == sb, '2': sa < sb}[k]
    if market == 'Mi-tps 1X2':
        if not has_ht: return None
        return {'1': ha > hb, 'X': ha == hb, '2': ha < hb}[k]
    if market == 'Double Chance':
        return {'1X': sa >= sb, '12': sa != sb, 'X2': sa <= sb}[k]
    if market == 'Mi-tps DC':
        if not has_ht: return None
        return {'1X': ha >= hb, '12': ha != hb, 'X2': ha <= hb}[k]
    if market == 'Score exact': return k == f"{sa}-{sb}"
    if market == 'Mi-tps CS':
        if not has_ht: return None
        return k == f"{ha}-{hb}"
    if market == '2ème mi-tps - CS':
        if not has_ht: return None
        return k == f"{sa - ha}-{sb - hb}"
    if market == '+/-':
        line = float(k.replace('>', '').replace('<', '').strip())
        return tot > line if k.startswith('>') else tot < line
    if market == 'Total de buts': return tot == int(k) or (int(k) == 6 and tot >= 6)
    if market == 'HT/FT':
        if not has_ht: return None
        p1, p2 = k.split('/')
        return sign(ha, hb) == p1 and sign(sa, sb) == p2
    if market == 'G/NG': return (sa > 0 and sb > 0) == (k == 'Oui')
    if market == 'Les deux équipes marquent / 1ère mi temps':
        if not has_ht: return None
        return (ha > 0 and hb > 0) == (k == 'Oui')
    if market == '1X2 & Total':
        r_, t_ = [x.strip() for x in k.split('/')]
        line = float(t_.replace('>', '').replace('<', '').strip())
        okt = tot > line if t_.startswith('>') else tot < line
        return sign(sa, sb) == r_ and okt
    if market == '1X2 & G/NG':
        kl = k.lower()
        if kl.startswith('1 gagne et les deux'): return sa > sb and sb > 0
        if kl.startswith('1 gagne et seulement'): return sa > sb and sb == 0
        if kl.startswith('2 gagne et les deux'): return sb > sa and sa > 0
        if kl.startswith('2 gagne et seulement'): return sb > sa and sa == 0
        if kl.startswith('x et aucun'): return sa == 0 and sb == 0
        if kl.startswith('x et les deux'): return sa == sb and sa > 0
        return None
    if market in ('Total equipe domicile', 'Total equipe extérieur'):
        v = sa if 'domicile' in market else sb
        line = float(k.replace('>', '').replace('<', '').strip())
        return v > line if k.startswith('>') else v < line
    if market == 'G/NG equipe domicile': return (sa > 0) == (k == 'Oui')
    if market == 'G/NG equipe extérieur': return (sb > 0) == (k == 'Oui')
    if market == 'Pair/Impair': return (tot % 2 == 0) == (k == 'Pair')
    if market == 'Minute du premier but':
        if goals is None: return None
        fmin = min((int(g['minute']) for g in goals), default=None)
        if 'pas de but' in k.lower(): return fmin is None
        try:
            lo_, hi_ = k.split('-'); return fmin is not None and int(lo_) <= fmin <= int(hi_)
        except Exception: return None
    if market == 'FTTS':
        if goals is None: return None
        if not goals: return k.lower().startswith('pas')
        first = sorted(goals, key=lambda g: (int(g['minute']), g.get('homeScore', 0) + g.get('awayScore', 0)))[0]
        if k == '1': return first['team'] == 'Home'
        if k == '2': return first['team'] == 'Away'
        return False
    if market == 'Multi-Buts':
        kl = k.lower()
        if '0, 1 ou 2' in kl: return tot <= 2
        if '1, 2 ou 3' in kl: return 1 <= tot <= 3
        if '2, 3 ou 4' in kl: return 2 <= tot <= 4
        if 'supérieur à 4' in kl: return tot > 4
        return None
    return None

# Construit le grand tableau (market, key) -> liste (win, odds, ts)
from collections import defaultdict
book = defaultdict(list)
for _, row in df.iterrows():
    em = dict(row['em'] or {})
    em['1X2'] = {'1': row['odds_home'], 'X': row['odds_draw'], '2': row['odds_away']}
    for mname, mdict in em.items():
        if not isinstance(mdict, dict): continue
        for k, v in mdict.items():
            try: odds = float(v)
            except Exception: continue
            w = settle(mname, k, row)
            if w is None: continue
            book[(mname, k)].append((int(w), odds, row['ts']))

cut = df['ts'].quantile(0.7)
rows_out = []
for (mname, k), recs in book.items():
    arr = pd.DataFrame(recs, columns=['win', 'odds', 'ts'])
    n = len(arr)
    if n < 300: continue
    p_obs = arr['win'].mean()
    p_imp = (1.0 / arr['odds']).mean()      # implicite brute (avec marge)
    roi = (arr['win'] * arr['odds'] - 1).mean()
    se = (arr['win'] * arr['odds'] - 1).std(ddof=1) / math.sqrt(n)
    z = roi / se if se > 0 else 0
    tr = arr[arr['ts'] <= cut]; te = arr[arr['ts'] > cut]
    roi_tr = (tr['win'] * tr['odds'] - 1).mean() if len(tr) else np.nan
    roi_te = (te['win'] * te['odds'] - 1).mean() if len(te) else np.nan
    rows_out.append((mname, k, n, p_obs, p_imp, arr['odds'].mean(), roi, z, roi_tr, roi_te, len(te)))
out = pd.DataFrame(rows_out, columns=['market', 'key', 'n', 'p_obs', 'p_implied', 'avg_odds',
                                      'roi', 'z', 'roi_train', 'roi_oos', 'n_oos'])
out = out.sort_values('z', ascending=False)
pd.set_option('display.width', 200)
print(f"Outcomes évalués (n>=300): {len(out)} | split walk-forward au {cut}")
print("\nTOP 20 par z(ROI):")
print(out.head(20).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nBOTTOM 10:")
print(out.tail(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

pos = out[(out['roi_train'] > 0.02) & (out['n'] >= 500)]
print(f"\nCandidats train ROI>+2%: {len(pos)}")
if len(pos):
    print(pos.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for _, r in pos.iterrows():
        recs = pd.DataFrame(book[(r['market'], r['key'])], columns=['win', 'odds', 'ts'])
        te = recs[recs['ts'] > cut]
        wins = int(te['win'].sum())
        bt = stats.binomtest(wins, len(te), (1.0 / te['odds']).mean())
        print(f"  OOS {r['market']} / {r['key']}: n={len(te)} WR={te['win'].mean():.4f} "
              f"ROI={((te['win'] * te['odds'] - 1).mean()) * 100:+.2f}% binom_p={bt.pvalue:.4f}")

# Marge moyenne par marché
print("\nOverround par marché (somme 1/odds, médiane):")
ov = defaultdict(list)
for _, row in df.iterrows():
    em = dict(row['em'] or {})
    em['1X2'] = {'1': row['odds_home'], 'X': row['odds_draw'], '2': row['odds_away']}
    for mname, mdict in em.items():
        if isinstance(mdict, dict) and mdict:
            try: ov[mname].append(sum(1.0 / float(v) for v in mdict.values()))
            except Exception: pass
for mname, v in sorted(ov.items(), key=lambda kv: np.median(kv[1])):
    print(f"  {mname:45s} {np.median(v):.4f} (n={len(v)})")

print("\nFIN _wf3_rng3.py")
