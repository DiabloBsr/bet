# -*- coding: utf-8 -*-
"""WF3 - LA CARTE DES MARGES (ligue 8035, Sporty-Tech virtual football)
1. Overround par marche (moyenne, std, par niveau de cote favori)
2. Biais favori-longshot par marche x bucket de cote
3. Top cellules (marche x bucket) -> walk-forward 70/30
4. Score exact : proba devig par score vs frequence reelle
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 400)
pd.set_option('display.max_columns', 30)

eng = create_engine(load_settings().db_url)

# ---------------------------------------------------------------- load data
Q = """
SELECT e.id, e.team_a, e.team_b, e.expected_start, e.round_info,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
     ON m.mid = o.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
with eng.connect() as c:
    df = pd.read_sql(text(Q), c)

df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df = df.sort_values('expected_start').reset_index(drop=True)
print(f"Matchs finis dedupliques avec cotes d'ouverture : {len(df)}")

# check minute range of goals
mins = []
for gj in df['goals_json'].dropna():
    g = json.loads(gj) if isinstance(gj, str) else gj
    for ev in (g or []):
        mins.append(int(ev['minute']))
mins = np.array(mins)
print(f"minutes des buts: min={mins.min()} max={mins.max()} | n={len(mins)}")

# ---------------------------------------------------------------- settlement
def sgn(a, b):
    return '1' if a > b else ('2' if b > a else 'X')

DC_MAP = {'1X': ('1', 'X'), 'X2': ('X', '2'), '12': ('1', '2')}

def first_goal(row):
    """retourne ('ok', dict|None) ou ('missing', None)"""
    gj = row['goals_json']
    total = int(row['score_a']) + int(row['score_b'])
    if gj is None or (isinstance(gj, float) and math.isnan(gj)):
        return ('ok', None) if total == 0 else ('missing', None)
    g = json.loads(gj) if isinstance(gj, str) else gj
    if not g:
        return ('ok', None) if total == 0 else ('missing', None)
    g = sorted(g, key=lambda x: int(x['minute']))
    return ('ok', g[0])

def minute_bin(m):
    if m <= 15: return '1-15'
    if m <= 30: return '16-30'
    if m <= 45: return '31-45'
    if m <= 60: return '46-60'
    if m <= 75: return '61-75'
    return '76-90'

def win_key(market, key, row):
    """retourne True/False si la selection `key` gagne, None si non-evaluable"""
    a, b = int(row['score_a']), int(row['score_b'])
    ha, hb = row['ht_score_a'], row['ht_score_b']
    ht_ok = ha is not None and hb is not None and not (isinstance(ha, float) and math.isnan(ha))
    if ht_ok:
        ha, hb = int(ha), int(hb)
    t = a + b
    s_ft = sgn(a, b)

    if market == '1X2':
        return key == s_ft
    if market == 'Mi-tps 1X2':
        return (key == sgn(ha, hb)) if ht_ok else None
    if market == 'Double Chance':
        return s_ft in DC_MAP[key]
    if market == 'Mi-tps DC':
        return (sgn(ha, hb) in DC_MAP[key]) if ht_ok else None
    if market == 'Score exact':
        return key == f"{a}-{b}"
    if market == 'Mi-tps CS':
        return (key == f"{ha}-{hb}") if ht_ok else None
    if market == '2ème mi-tps - CS':
        return (key == f"{a-ha}-{b-hb}") if ht_ok else None
    if market == '+/-':
        return (t > 3.5) if key.startswith('>') else (t < 3.5)
    if market == 'HT/FT':
        return (key == f"{sgn(ha, hb)}/{s_ft}") if ht_ok else None
    if market == 'Total de buts':
        return int(key) == t
    if market == 'G/NG':
        return (a > 0 and b > 0) == (key == 'Oui')
    if market == 'Les deux équipes marquent / 1ère mi temps':
        return ((ha > 0 and hb > 0) == (key == 'Oui')) if ht_ok else None
    if market == '1X2 & Total':
        part, tot = key.split(' / ')
        okt = (t > 3.5) if tot.startswith('>') else (t < 3.5)
        return (part == s_ft) and okt
    if market == '1X2 & G/NG':
        if key.startswith('1 gagne et les deux'):  return s_ft == '1' and b > 0
        if key.startswith('1 gagne et seulement'): return s_ft == '1' and b == 0
        if key.startswith('X et les deux'):        return s_ft == 'X' and a > 0
        if key.startswith('X et aucun'):           return a == 0 and b == 0
        if key.startswith('2 gagne et les deux'):  return s_ft == '2' and a > 0
        if key.startswith('2 gagne et seulement'): return s_ft == '2' and a == 0
        return None
    if market == 'Total equipe domicile':
        return (a > 3.5) if key.startswith('>') else (a < 3.5)
    if market == 'Total equipe extérieur':
        return (b > 3.5) if key.startswith('>') else (b < 3.5)
    if market == 'G/NG equipe domicile':
        return (a > 0) == (key == 'Oui')
    if market == 'G/NG equipe extérieur':
        return (b > 0) == (key == 'Oui')
    if market == 'Pair/Impair':
        return (t % 2 == 0) == (key == 'Pair')
    if market == 'Minute du premier but':
        st, fg = first_goal(row)
        if st == 'missing':
            return None
        if key == 'Pas de but':
            return fg is None
        return fg is not None and minute_bin(int(fg['minute'])) == key
    if market == 'FTTS':
        st, fg = first_goal(row)
        if st == 'missing':
            return None
        if key == 'Pas de but':
            return fg is None
        if fg is None:
            return False
        return key == ('1' if fg['team'] == 'Home' else '2')
    if market == 'Multi-Buts':
        if '0, 1 ou 2' in key: return t in (0, 1, 2)
        if '1, 2 ou 3' in key: return t in (1, 2, 3)
        if '2, 3 ou 4' in key: return t in (2, 3, 4)
        return t > 4
    return None

# markets ou la somme des probas vraies vaut 2 (selections chevauchantes)
COVER2 = {'Double Chance', 'Mi-tps DC'}
NONPART = {'Multi-Buts'}  # couverture variable (1 a 3 gagnants)

# ---------------------------------------------------------------- build selection table
rows_sel = []      # une ligne par selection
over_rows = []     # une ligne par (event, marche) pour l'overround
parse_fail = Counter()

for idx, row in df.iterrows():
    em = row['extra_markets']
    if em is None:
        continue
    em = json.loads(em) if isinstance(em, str) else em
    markets = dict(em)
    markets['1X2'] = {'1': row['odds_home'], 'X': row['odds_draw'], '2': row['odds_away']}
    for mk, sels in markets.items():
        inv = {}
        for k, o in sels.items():
            try:
                o = float(o)
            except (TypeError, ValueError):
                continue
            if o < 1.01:        # cotes degenerees (ex 0.95)
                parse_fail[(mk, 'odds<1.01')] += 1
                continue
            inv[k] = 1.0 / o
        if not inv:
            continue
        S = sum(inv.values())
        cover = 2.0 if mk in COVER2 else 1.0
        fav = min(sels[k] for k in inv)
        over_rows.append({'event_id': row['id'], 'ts': row['expected_start'],
                          'market': mk, 'sum_inv': S, 'overround': S / cover - 1.0,
                          'fav_odds': fav, 'n_sel': len(inv)})
        for k, iv in inv.items():
            w = win_key(mk, k, row)
            if w is None:
                parse_fail[(mk, 'unsettled')] += 1
                continue
            o = 1.0 / iv
            p_dev = iv / S * cover if mk not in NONPART else iv  # devig proportionnel
            rows_sel.append({'event_id': row['id'], 'ts': row['expected_start'],
                             'market': mk, 'key': k, 'odds': o, 'inv': iv,
                             'p_dev': p_dev, 'win': int(w)})

sel = pd.DataFrame(rows_sel)
ovr = pd.DataFrame(over_rows)
print(f"\nselections evaluees: {len(sel)} | marches x events: {len(ovr)}")
if parse_fail:
    print("anomalies:", dict(parse_fail))

# sanity check : nb de gagnants par (event, marche) pour partitions
chk = sel.groupby(['event_id', 'market'])['win'].sum().reset_index()
for mk in sorted(sel['market'].unique()):
    c = chk[chk['market'] == mk]['win']
    if mk in COVER2:
        exp = 2
    elif mk in NONPART:
        exp = None
    else:
        exp = 1
    bad = (c != exp).mean() if exp is not None else np.nan
    print(f"  sanity {mk:45s} winners mean={c.mean():.3f}  (attendu {exp})  pct_hors_attendu={bad if exp else float('nan'):.4f}" if exp else
          f"  sanity {mk:45s} winners mean={c.mean():.3f}  (couverture variable)")

# ---------------------------------------------------------------- PART 1 : overround map
print("\n" + "=" * 100)
print("PART 1 - OVERROUND PAR MARCHE (cotes d'ouverture)")
print("=" * 100)
g = ovr.groupby('market')['overround'].agg(['mean', 'std', 'count'])
g = g.sort_values('mean')
print((g * np.array([100, 100, 1])).rename(columns={'mean': 'overround_%', 'std': 'std_%'}).round(2))

# par niveau de cote favori
print("\n--- overround par niveau de cote du favori (marches a partition) ---")
ovr['fav_bucket'] = pd.cut(ovr['fav_odds'], [1.0, 1.3, 1.6, 2.0, 3.0, 100.5],
                           labels=['<1.3', '1.3-1.6', '1.6-2', '2-3', '3+'])
piv = ovr[~ovr['market'].isin(NONPART)].pivot_table(
    index='market', columns='fav_bucket', values='overround', aggfunc='mean', observed=True) * 100
print(piv.round(2))

# ---------------------------------------------------------------- PART 2 : favori-longshot par marche
print("\n" + "=" * 100)
print("PART 2 - BIAIS FAVORI-LONGSHOT : marche x bucket de cote")
print("frequence reelle vs proba implicite devigorisee (devig proportionnel)")
print("=" * 100)
BUCKETS = [1.01, 1.5, 2, 3, 5, 10, 20, 1000]
BLAB = ['1-1.5', '1.5-2', '2-3', '3-5', '5-10', '10-20', '20+']
sel['bucket'] = pd.cut(sel['odds'], BUCKETS, labels=BLAB, right=False)

cell_rows = []
for (mk, bk), gdf in sel.groupby(['market', 'bucket'], observed=True):
    n = len(gdf)
    if n < 30:
        continue
    wins = gdf['win'].sum()
    freq = wins / n
    p_imp = gdf['p_dev'].mean()
    avg_o = gdf['odds'].mean()
    ev = freq * avg_o - 1.0          # EV brut par unite si on prend la cote moyenne
    ev_exact = (gdf['win'] * gdf['odds']).mean() - 1.0  # ROI flat reel
    try:
        pval = stats.binomtest(int(wins), n, p_imp).pvalue
    except ValueError:
        pval = np.nan
    cell_rows.append({'market': mk, 'bucket': bk, 'n': n, 'avg_odds': avg_o,
                      'freq_reelle': freq, 'p_implicite': p_imp,
                      'ratio': freq / p_imp if p_imp > 0 else np.nan,
                      'roi_flat': ev_exact, 'pval_binom': pval})
cells = pd.DataFrame(cell_rows)

for mk in sorted(cells['market'].unique()):
    sub = cells[cells['market'] == mk]
    print(f"\n### {mk}")
    print(sub[['bucket', 'n', 'avg_odds', 'freq_reelle', 'p_implicite', 'ratio', 'roi_flat', 'pval_binom']]
          .to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------- PART 3 : top cellules + walk-forward
print("\n" + "=" * 100)
print("PART 3 - TOP 10 CELLULES (ratio freq/implicite, n>=100) + WALK-FORWARD 70/30")
print("=" * 100)
top = cells[cells['n'] >= 100].sort_values('ratio', ascending=False).head(10)
print(top.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# walk-forward : split temporel sur les events
ev_sorted = df['id'].tolist()          # df deja trie par expected_start
cut = int(len(ev_sorted) * 0.7)
train_ids = set(ev_sorted[:cut])
sel['is_train'] = sel['event_id'].isin(train_ids)
tr, te = sel[sel['is_train']], sel[~sel['is_train']]
print(f"\ntrain: {cut} events ({tr.shape[0]} selections) | OOS: {len(ev_sorted)-cut} events ({te.shape[0]} selections)")

def cell_stats(d):
    out = []
    for (mk, bk), gdf in d.groupby(['market', 'bucket'], observed=True):
        n = len(gdf)
        if n == 0:
            continue
        wins = gdf['win'].sum()
        p_imp = gdf['p_dev'].mean()
        roi = (gdf['win'] * gdf['odds']).mean() - 1.0
        freq = wins / n
        try:
            pv = stats.binomtest(int(wins), n, p_imp).pvalue if p_imp > 0 else np.nan
        except ValueError:
            pv = np.nan
        out.append({'market': mk, 'bucket': bk, 'n': n, 'freq': freq, 'p_imp': p_imp,
                    'ratio': freq / p_imp if p_imp > 0 else np.nan, 'roi': roi,
                    'avg_odds': gdf['odds'].mean(), 'pval': pv})
    return pd.DataFrame(out)

cs_tr = cell_stats(tr)
cand = cs_tr[(cs_tr['n'] >= 150) & (cs_tr['roi'] > 0)].sort_values('roi', ascending=False)
print(f"\ncellules candidates (train, n>=150, ROI_train>0) : {len(cand)}")
print(cand.head(15).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- evaluation OOS des candidates ---")
wf_rows = []
for _, r in cand.iterrows():
    sub = te[(te['market'] == r['market']) & (te['bucket'] == r['bucket'])]
    if len(sub) == 0:
        continue
    roi = (sub['win'] * sub['odds']).mean() - 1.0
    wf_rows.append({'market': r['market'], 'bucket': r['bucket'],
                    'n_train': r['n'], 'roi_train': r['roi'], 'pval_train': r['pval'],
                    'n_oos': len(sub), 'wr_oos': sub['win'].mean(),
                    'avg_odds_oos': sub['odds'].mean(), 'roi_oos': roi})
wf = pd.DataFrame(wf_rows)
if len(wf):
    print(wf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # portefeuille : candidates avec pval_train < 0.05
    strict = wf[wf['pval_train'] < 0.05]
    if len(strict):
        ids = list(zip(strict['market'], strict['bucket']))
        mask = te.apply(lambda x: (x['market'], x['bucket']) in ids, axis=1)
        port = te[mask]
        roi_p = (port['win'] * port['odds']).mean() - 1.0
        print(f"\nPortefeuille strict (pval_train<0.05): n_oos={len(port)} "
              f"wr={port['win'].mean():.4f} avg_odds={port['odds'].mean():.3f} ROI_OOS={roi_p:+.4f}")

# ---------------------------------------------------------------- PART 4 : Score exact detaille
print("\n" + "=" * 100)
print("PART 4 - SCORE EXACT : proba devig par score vs frequence reelle")
print("=" * 100)
cs = sel[sel['market'] == 'Score exact']
n_ev = cs['event_id'].nunique()
rows4 = []
for k, gdf in cs.groupby('key'):
    n = len(gdf)
    wins = gdf['win'].sum()
    freq = wins / n
    p_imp = gdf['p_dev'].mean()
    avg_o = gdf['odds'].mean()
    roi = (gdf['win'] * gdf['odds']).mean() - 1.0
    try:
        pv = stats.binomtest(int(wins), n, p_imp).pvalue
    except ValueError:
        pv = np.nan
    a, b = map(int, k.split('-'))
    rows4.append({'score': k, 'tot': a + b, 'n': n, 'avg_odds': avg_o,
                  'p_implicite': p_imp, 'freq_reelle': freq,
                  'ratio': freq / p_imp if p_imp > 0 else np.nan,
                  'roi_flat': roi, 'pval': pv})
sc = pd.DataFrame(rows4).sort_values(['tot', 'score'])
print(f"(n events avec marche Score exact = {n_ev})")
print(sc.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- agregation par total de buts (Score exact devig vs reel) ---")
agg = sc.groupby('tot').apply(
    lambda d: pd.Series({'p_imp_sum': d['p_implicite'].sum(),
                         'freq_sum': d['freq_reelle'].sum()}), include_groups=False)
agg['ratio'] = agg['freq_sum'] / agg['p_imp_sum']
print(agg.to_string(float_format=lambda x: f"{x:.4f}"))

print("\n--- symetrie home/away : score a-b vs b-a ---")
sym_rows = []
for k in sc['score']:
    a, b = map(int, k.split('-'))
    if a > b:
        mirror = f"{b}-{a}"
        if mirror in set(sc['score']):
            r1 = sc[sc['score'] == k].iloc[0]
            r2 = sc[sc['score'] == mirror].iloc[0]
            sym_rows.append({'home_win': k, 'away_win': mirror,
                             'p_imp_H': r1['p_implicite'], 'p_imp_A': r2['p_implicite'],
                             'freq_H': r1['freq_reelle'], 'freq_A': r2['freq_reelle'],
                             'ratio_H': r1['ratio'], 'ratio_A': r2['ratio']})
print(pd.DataFrame(sym_rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# chi2 global : distribution reelle des 28 scores vs distribution implicite moyenne
obs = sc.set_index('score')['freq_reelle'] * n_ev
expp = sc.set_index('score')['p_implicite']
expp = expp / expp.sum()
chi2, pchi = stats.chisquare(obs, expp * obs.sum())
print(f"\nchi2 global score exact (28 cases, reel vs implicite devig): chi2={chi2:.1f} p={pchi:.3e}")
print("\nDONE")
