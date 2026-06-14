# -*- coding: utf-8 -*-
"""WF3 iter 2 - architecture des marges + deep-dive FTTS + score exact cap-aware.

A. Architecture du vig : marge effective par selection et par marche (hors caps/floors),
   test "memes cotes pour le meme outcome dans des marches differents".
B. Caps : % de cotes a 100.0 / <1.01 par marche ; total de buts max du generateur.
C. Score exact cap-aware : p_model = (1/cote)/(1+v) avec v ancre par cross-marche ;
   chi2 global et par score (binomial).
D. FTTS : p1 implicite vs q_grid (P(home marque 1er) derive de la grille Score exact),
   regression logit, test d'echangeabilite de l'ordre des buts par cellule de score,
   walk-forward 70/30 de regles raffinees + stabilite par folds.
E. Tables par cle : FTTS, Minute du premier but, Multi-Buts, HT/FT.
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)

eng = create_engine(load_settings().db_url)
Q = """
SELECT e.id, e.team_a, e.team_b, e.expected_start, e.round_info,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m ON m.mid = o.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
with eng.connect() as c:
    df = pd.read_sql(text(Q), c)
df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df = df.sort_values('expected_start').reset_index(drop=True)
print(f"events: {len(df)}")

CAP, FLOOR = 99.99, 1.01

def parse_em(row):
    em = row['extra_markets']
    if em is None: return None
    em = json.loads(em) if isinstance(em, str) else em
    em = dict(em)
    em['1X2'] = {'1': row['odds_home'], 'X': row['odds_draw'], '2': row['odds_away']}
    return em

def sgn(a, b): return '1' if a > b else ('2' if b > a else 'X')

def first_team(row):
    """'Home'/'Away'/None(no goal)/'NA'(inconnu)"""
    a, b = int(row['score_a']), int(row['score_b'])
    if a == 0 and b == 0: return None
    if a > 0 and b == 0: return 'Home'
    if b > 0 and a == 0: return 'Away'
    gj = row['goals_json']
    if gj is None or (isinstance(gj, float) and math.isnan(gj)): return 'NA'
    g = json.loads(gj) if isinstance(gj, str) else gj
    if not g: return 'NA'
    g = sorted(g, key=lambda x: int(x['minute']))
    return g[0]['team']

# ============================================================ B. caps & grid support
print("\n" + "=" * 90)
print("B. CAPS / FLOORS / SUPPORT DU GENERATEUR")
print("=" * 90)
tot = df['score_a'].astype(int) + df['score_b'].astype(int)
print(f"total de buts: max={tot.max()}  distribution={dict(Counter(tot))}")
print(f"score_a max={df['score_a'].astype(int).max()}  score_b max={df['score_b'].astype(int).max()}")

cap_cnt, floor_cnt, all_cnt = Counter(), Counter(), Counter()
for _, row in df.iterrows():
    em = parse_em(row)
    if em is None: continue
    for mk, sels in em.items():
        for k, o in sels.items():
            try: o = float(o)
            except (TypeError, ValueError): continue
            all_cnt[mk] += 1
            if o > CAP: cap_cnt[mk] += 1
            if o < FLOOR: floor_cnt[mk] += 1
print("\n% selections cappees a 100.0 / floorees <1.01 par marche:")
for mk in sorted(all_cnt):
    print(f"  {mk:48s} cap100={cap_cnt[mk]/all_cnt[mk]*100:6.2f}%  floor={floor_cnt[mk]/all_cnt[mk]*100:6.2f}%")

# ============================================================ A. architecture du vig
print("\n" + "=" * 90)
print("A. ARCHITECTURE DU VIG")
print("=" * 90)
# A1: memes cotes pour le meme outcome "0 but dans le match" dans 5 marches ?
same, diff, ncheck = 0, 0, 0
for _, row in df.iterrows():
    em = parse_em(row)
    if em is None: continue
    vals = []
    try: vals.append(float(em['Score exact']['0-0']))
    except (KeyError, TypeError): pass
    try: vals.append(float(em['Total de buts']['0']))
    except (KeyError, TypeError): pass
    try: vals.append(float(em['FTTS']['Pas de but']))
    except (KeyError, TypeError): pass
    try: vals.append(float(em['Minute du premier but']['Pas de but']))
    except (KeyError, TypeError): pass
    try: vals.append(float(em['1X2 & G/NG']['X et aucun but']))
    except (KeyError, TypeError): pass
    if len(vals) >= 2:
        ncheck += 1
        if max(vals) - min(vals) < 0.005: same += 1
        else: diff += 1
print(f"A1. cote '0 but' identique a travers 5 marches: identique={same}/{ncheck} ({same/ncheck*100:.2f}%)")

# A2: marge effective par selection par marche = 1/payout - 1 (hors caps/floors)
# settlement minimal par cle (repris de _wf3_margins, sous-ensemble robuste)
DC_MAP = {'1X': ('1', 'X'), 'X2': ('X', '2'), '12': ('1', '2')}
def minute_bin(m):
    if m <= 15: return '1-15'
    if m <= 30: return '16-30'
    if m <= 45: return '31-45'
    if m <= 60: return '46-60'
    if m <= 75: return '61-75'
    return '76-90'

def win_key(market, key, row, ft):
    a, b = int(row['score_a']), int(row['score_b'])
    ha, hb = row['ht_score_a'], row['ht_score_b']
    ht_ok = ha is not None and not (isinstance(ha, float) and math.isnan(ha))
    if ht_ok: ha, hb = int(ha), int(hb)
    t = a + b; s = sgn(a, b)
    if market == '1X2': return key == s
    if market == 'Mi-tps 1X2': return (key == sgn(ha, hb)) if ht_ok else None
    if market == 'Double Chance': return s in DC_MAP[key]
    if market == 'Mi-tps DC': return (sgn(ha, hb) in DC_MAP[key]) if ht_ok else None
    if market == 'Score exact': return key == f"{a}-{b}"
    if market == 'Mi-tps CS': return (key == f"{ha}-{hb}") if ht_ok else None
    if market == '2ème mi-tps - CS': return (key == f"{a-ha}-{b-hb}") if ht_ok else None
    if market == '+/-': return (t > 3.5) if key.startswith('>') else (t < 3.5)
    if market == 'HT/FT': return (key == f"{sgn(ha, hb)}/{s}") if ht_ok else None
    if market == 'Total de buts': return int(key) == t
    if market == 'G/NG': return (a > 0 and b > 0) == (key == 'Oui')
    if market == 'Les deux équipes marquent / 1ère mi temps':
        return ((ha > 0 and hb > 0) == (key == 'Oui')) if ht_ok else None
    if market == '1X2 & Total':
        part, totk = key.split(' / ')
        okt = (t > 3.5) if totk.startswith('>') else (t < 3.5)
        return (part == s) and okt
    if market == '1X2 & G/NG':
        if key.startswith('1 gagne et les deux'): return s == '1' and b > 0
        if key.startswith('1 gagne et seulement'): return s == '1' and b == 0
        if key.startswith('X et les deux'): return s == 'X' and a > 0
        if key.startswith('X et aucun'): return a == 0 and b == 0
        if key.startswith('2 gagne et les deux'): return s == '2' and a > 0
        if key.startswith('2 gagne et seulement'): return s == '2' and a == 0
        return None
    if market == 'Total equipe domicile': return (a > 3.5) if key.startswith('>') else (a < 3.5)
    if market == 'Total equipe extérieur': return (b > 3.5) if key.startswith('>') else (b < 3.5)
    if market == 'G/NG equipe domicile': return (a > 0) == (key == 'Oui')
    if market == 'G/NG equipe extérieur': return (b > 0) == (key == 'Oui')
    if market == 'Pair/Impair': return (t % 2 == 0) == (key == 'Pair')
    if market == 'Minute du premier but':
        if ft == 'NA': return None
        if key == 'Pas de but': return ft is None
        if ft is None: return False
        gj = row['goals_json']
        if gj is None or (isinstance(gj, float) and math.isnan(gj)): return None
        g = json.loads(gj) if isinstance(gj, str) else gj
        if not g: return None
        g = sorted(g, key=lambda x: int(x['minute']))
        return minute_bin(int(g[0]['minute'])) == key
    if market == 'FTTS':
        if ft == 'NA': return None
        if key == 'Pas de but': return ft is None
        if ft is None: return False
        return key == ('1' if ft == 'Home' else '2')
    if market == 'Multi-Buts':
        if '0, 1 ou 2' in key: return t in (0, 1, 2)
        if '1, 2 ou 3' in key: return t in (1, 2, 3)
        if '2, 3 ou 4' in key: return t in (2, 3, 4)
        return t > 4
    return None

rows = []
ftts_rows = []
exch = defaultdict(lambda: [0, 0])   # (a,b) -> [n, first_home]
for _, row in df.iterrows():
    em = parse_em(row)
    if em is None: continue
    ft = first_team(row)
    a, b = int(row['score_a']), int(row['score_b'])
    if ft not in (None, 'NA') and a > 0 and b > 0:
        key = (a, b)
        exch[key][0] += 1
        exch[key][1] += int(ft == 'Home')
    # grille Score exact devig (cap-aware, marge uniforme renormalisee)
    q_grid, p00g = np.nan, np.nan
    se = em.get('Score exact')
    if se:
        inv, num = 0.0, 0.0
        p00_inv = 0.0
        for k, o in se.items():
            o = float(o)
            if o > CAP: continue
            sa, sb = map(int, k.split('-'))
            iv = 1.0 / o
            inv += iv
            if sa + sb > 0: num += iv * sa / (sa + sb)
            else: p00_inv = iv
        if inv > 0:
            q_grid = num / inv          # P(1er buteur = home), renormalise (0-0 inclus au denominateur)
            p00g = p00_inv / inv
    ftts = em.get('FTTS')
    if ftts and not np.isnan(q_grid):
        try:
            o1, o2, o0 = float(ftts['1']), float(ftts['2']), float(ftts['Pas de but'])
            S = 1/o1 + 1/o2 + 1/o0
            ftts_rows.append({'event_id': row['id'], 'ts': row['expected_start'],
                              'o1': o1, 'o2': o2, 'o0': o0,
                              'p1': (1/o1)/S, 'p2': (1/o2)/S, 'p0': (1/o0)/S,
                              'q_grid': q_grid, 'p00_grid': p00g,
                              'win1': None if ft == 'NA' else int(ft == 'Home'),
                              'win2': None if ft == 'NA' else int(ft == 'Away'),
                              'oh': row['odds_home']})
        except (KeyError, TypeError, ValueError):
            pass
    for mk, sels in em.items():
        for k, o in sels.items():
            try: o = float(o)
            except (TypeError, ValueError): continue
            if o < FLOOR or o > CAP: continue
            w = win_key(mk, k, row, ft)
            if w is None: continue
            rows.append({'event_id': row['id'], 'ts': row['expected_start'],
                         'market': mk, 'key': k, 'odds': o, 'win': int(w)})
sel = pd.DataFrame(rows)
print(f"\nA2. marge effective par selection (= 1/(freq*cote) - 1), hors caps/floors:")
marg = sel.groupby('market').apply(
    lambda d: pd.Series({'n': len(d), 'payout': (d['win'] * d['odds']).mean()}), include_groups=False)
marg['v_eff_%'] = (1 / marg['payout'] - 1) * 100
print(marg.sort_values('v_eff_%').to_string(float_format=lambda x: f"{x:.3f}"))

# ============================================================ C. Score exact cap-aware
print("\n" + "=" * 90)
print("C. SCORE EXACT CAP-AWARE  (p_model = (1/cote)/1.12, cotes<100 uniquement)")
print("=" * 90)
V_SE = 0.12
cs = sel[sel['market'] == 'Score exact'].copy()
cs['p_model'] = (1 / cs['odds']) / (1 + V_SE)
out = []
for k, g in cs.groupby('key'):
    n = len(g); wins = g['win'].sum(); p = g['p_model'].mean()
    pv = stats.binomtest(int(wins), n, p).pvalue
    sa, sb = map(int, k.split('-'))
    out.append({'score': k, 'tot': sa+sb, 'n_noncap': n, 'avg_odds': g['odds'].mean(),
                'p_model': p, 'freq': wins/n, 'ratio': wins/n/p,
                'roi_flat': (g['win']*g['odds']).mean()-1, 'pval': pv})
sc = pd.DataFrame(out).sort_values(['tot', 'score'])
print(sc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
E = cs.groupby('key')['p_model'].sum(); O = cs.groupby('key')['win'].sum()
chi2 = (((O - E) ** 2) / E).sum()
print(f"\nchi2 (28 scores, E=somme p_model par cellule, cotes non cappees): "
      f"chi2={chi2:.1f} ddl~27 p={1-stats.chi2.cdf(chi2, 27):.3e}")
print(f"somme p_model moyenne par event: {cs.groupby('event_id')['p_model'].sum().mean():.4f} "
      f"(doit etre ~<1, masse cappee exclue)")

# ============================================================ E. tables par cle
print("\n" + "=" * 90)
print("E. BIAIS PAR CLE (freq*cote = payout ; payout attendu = 1/(1+v_marche))")
print("=" * 90)
for mk in ['FTTS', 'Minute du premier but', 'Multi-Buts', 'HT/FT', 'Mi-tps 1X2']:
    sub = sel[sel['market'] == mk]
    t = sub.groupby('key').apply(lambda d: pd.Series({
        'n': len(d), 'avg_odds': d['odds'].mean(), 'freq': d['win'].mean(),
        'p_unif': ((1/d['odds'])/(1+0.12 if mk != 'Mi-tps 1X2' else 1.08)).mean(),
        'payout': (d['win']*d['odds']).mean()}), include_groups=False)
    t['pval'] = [stats.binomtest(int(r['freq']*r['n']), int(r['n']), min(r['p_unif'], 1)).pvalue
                 for _, r in t.iterrows()]
    print(f"\n### {mk}")
    print(t.sort_values('avg_odds').to_string(float_format=lambda x: f"{x:.4f}"))

# ============================================================ D. FTTS deep-dive
print("\n" + "=" * 90)
print("D. FTTS DEEP-DIVE")
print("=" * 90)
fd = pd.DataFrame(ftts_rows)
fd = fd.dropna(subset=['q_grid'])
print(f"events avec FTTS + grille: {len(fd)} ; win connu: {fd['win1'].notna().sum()}")

# D1: p1 implicite FTTS vs q_grid (la propre grille du moteur)
d1 = fd.dropna(subset=['win1']).copy()
d1['win1'] = d1['win1'].astype(int)
print(f"\nD1. moyenne p1_FTTS={d1['p1'].mean():.4f}  q_grid={d1['q_grid'].mean():.4f}  "
      f"freq reelle win1={d1['win1'].mean():.4f}")
# regression logit p1 ~ logit q_grid
eps = 1e-6
lq = np.log(np.clip(d1['q_grid'], eps, 1-eps) / (1 - np.clip(d1['q_grid'], eps, 1-eps)))
lp = np.log(np.clip(d1['p1'], eps, 1-eps) / (1 - np.clip(d1['p1'], eps, 1-eps)))
sl, ic, r, _, _ = stats.linregress(lq, lp)
print(f"D1. logit(p1_FTTS) = {ic:+.4f} + {sl:.4f} * logit(q_grid)   r={r:.4f}")
print("    -> slope<1 = compression des probas FTTS vers 0.5 vs sa propre grille")
# calibration de q_grid vs realite par decile
d1['dec'] = pd.qcut(d1['q_grid'], 10, duplicates='drop')
cal = d1.groupby('dec', observed=True).apply(lambda d: pd.Series({
    'n': len(d), 'q_grid': d['q_grid'].mean(), 'p1_ftts': d['p1'].mean(),
    'freq_win1': d['win1'].mean()}), include_groups=False)
print("\nD1. calibration par decile de q_grid (q_grid vs p1_FTTS vs freq reelle):")
print(cal.to_string(float_format=lambda x: f"{x:.4f}"))

# D2: echangeabilite de l'ordre des buts : P(1er=home | score a-b) vs a/(a+b)
print("\nD2. ordre des buts echangeable ? cellules a,b>0, n>=50:")
rows2 = []
for (a, b), (n, h) in sorted(exch.items()):
    if n < 50: continue
    p0 = a / (a + b)
    pv = stats.binomtest(h, n, p0).pvalue
    rows2.append({'score': f"{a}-{b}", 'n': n, 'attendu_a/(a+b)': p0,
                  'freq_1er_home': h / n, 'pval': pv})
print(pd.DataFrame(rows2).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# D3: walk-forward des regles FTTS
print("\nD3. WALK-FORWARD 70/30 (split temporel)")
d1 = d1.sort_values('ts').reset_index(drop=True)
cut = int(len(d1) * 0.7)
tr, te = d1.iloc[:cut], d1.iloc[cut:]
print(f"train={len(tr)} OOS={len(te)}")

def roi_rule(d, side, mask):
    sub = d[mask]
    if side == 1:
        return len(sub), sub['win1'].mean() if len(sub) else np.nan, \
               (sub['win1'] * sub['o1']).mean() - 1 if len(sub) else np.nan, sub['o1'].mean() if len(sub) else np.nan
    else:
        w2 = 1 - sub['win1'] - (sub['q_grid'] * 0)  # placeholder
        return None

# Regle A : FTTS '1' cote <= seuil — choisir seuil sur train
print("\nRegle A: parier FTTS '1' si o1 <= s")
best = None
for s in [1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60]:
    m = tr['o1'] <= s
    n = m.sum()
    if n < 300: continue
    roi = (tr.loc[m, 'win1'] * tr.loc[m, 'o1']).mean() - 1
    print(f"  train s={s:.2f}: n={n} roi={roi:+.4f}")
    if best is None or roi > best[1]: best = (s, roi)
s = best[0]
m = te['o1'] <= s
roi_o = (te.loc[m, 'win1'] * te.loc[m, 'o1']).mean() - 1
print(f"  -> seuil retenu {s:.2f} | OOS: n={m.sum()} wr={te.loc[m,'win1'].mean():.4f} "
      f"avg_odds={te.loc[m,'o1'].mean():.3f} ROI_OOS={roi_o:+.4f}")
binp = stats.binomtest(int(te.loc[m,'win1'].sum()), int(m.sum()), (1/te.loc[m,'o1']).mean()).pvalue
print(f"     binom OOS vs implicite brute: p={binp:.4f}")

# Regle B : FTTS '2' si o2 <= 1.5 (symetrie cote away)
for s2 in [1.5, 1.7]:
    m2tr = tr['o2'] <= s2
    m2te = te['o2'] <= s2
    w2tr = (1 - tr.loc[m2tr, 'win1']) * (tr.loc[m2tr, 'p0'] * 0 + 1)  # win2 = 1-win1 si but (win1 deja 0/1; 0-0 -> win1=0 MAIS win2=0)
    # correction: win2 vrai = win2 colonne
    pass
fd2 = d1.copy()
fd2['win2v'] = fd2['win2'].astype(float)
tr2, te2 = fd2.iloc[:cut], fd2.iloc[cut:]
for s2 in [1.4, 1.5, 1.6, 1.8]:
    mtr = tr2['o2'] <= s2; mte = te2['o2'] <= s2
    if mtr.sum() < 50:
        print(f"Regle B (FTTS '2' o2<={s2}): train n={mtr.sum()} insuffisant"); continue
    rtr = (tr2.loc[mtr, 'win2v'] * tr2.loc[mtr, 'o2']).mean() - 1
    rte = (te2.loc[mte, 'win2v'] * te2.loc[mte, 'o2']).mean() - 1 if mte.sum() else np.nan
    print(f"Regle B o2<={s2}: train n={mtr.sum()} roi={rtr:+.4f} | OOS n={mte.sum()} roi={rte:+.4f}")

# Regle C : value vs grille : parier '1' si q_grid * o1 > 1 + t
print("\nRegle C: parier '1' si q_grid*o1 > 1+t (t choisi sur train)")
best = None
for t in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
    m = tr['q_grid'] * tr['o1'] > 1 + t
    n = m.sum()
    if n < 300: continue
    roi = (tr.loc[m, 'win1'] * tr.loc[m, 'o1']).mean() - 1
    print(f"  train t={t:.2f}: n={n} roi={roi:+.4f}")
    if best is None or roi > best[1]: best = (t, roi)
if best:
    t = best[0]
    m = te['q_grid'] * te['o1'] > 1 + t
    roi_o = (te.loc[m, 'win1'] * te.loc[m, 'o1']).mean() - 1
    print(f"  -> t retenu {t:.2f} | OOS: n={m.sum()} wr={te.loc[m,'win1'].mean():.4f} "
          f"avg_odds={te.loc[m,'o1'].mean():.3f} ROI_OOS={roi_o:+.4f}")

# Regle C2 : pareil cote 2
m_anyc = (fd2['q_grid'].notna())
fd2['q2_grid'] = 1 - fd2['q_grid'] - fd2['p00_grid']
tr2, te2 = fd2.iloc[:cut], fd2.iloc[cut:]
best = None
for t in [0.00, 0.05, 0.10]:
    m = tr2['q2_grid'] * tr2['o2'] > 1 + t
    if m.sum() < 100: continue
    roi = (tr2.loc[m, 'win2v'] * tr2.loc[m, 'o2']).mean() - 1
    print(f"  C2 train t={t:.2f}: n={m.sum()} roi={roi:+.4f}")
    if best is None or roi > best[1]: best = (t, roi)
if best:
    t = best[0]
    m = te2['q2_grid'] * te2['o2'] > 1 + t
    if m.sum():
        roi_o = (te2.loc[m, 'win2v'] * te2.loc[m, 'o2']).mean() - 1
        print(f"  -> C2 t={t:.2f} | OOS: n={m.sum()} roi={roi_o:+.4f}")

# stabilite par folds (regle A seuil 1.5 fixe, tout l'historique)
print("\nStabilite regle 'FTTS 1 @ o1<=1.50' par quintile temporel:")
d1['fold'] = pd.qcut(np.arange(len(d1)), 5, labels=False)
for f in range(5):
    sub = d1[(d1['fold'] == f) & (d1['o1'] <= 1.50)]
    roi = (sub['win1'] * sub['o1']).mean() - 1
    print(f"  fold {f}: n={len(sub)} wr={sub['win1'].mean():.4f} roi={roi:+.4f}")

print("\nDONE")
