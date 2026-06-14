# -*- coding: utf-8 -*-
"""WF3 — TIMELINE part 4 : le biais d'ordre Home-first.

M. Minutes des buts Home vs Away (KS) — le moteur place-t-il les buts Home plus tôt ?
N. P(Home marque 1er | score final h-a) vs interleaving aléatoire h/(h+a) (Poisson-binomial)
O. Momentum sous null per-(half, k_half) — le null le plus conditionné
P. FTTS '1' : recherche d'un sous-ensemble exploitable (train 70% -> OOS 30%)
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(13)
eng = create_engine(load_settings().db_url)

q = """
SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start,
       r.score_a, r.score_b, r.goals_json
FROM events e JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0'
"""
df = pd.read_sql(text(q), eng)
df = df.sort_values('id').drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df = df.dropna(subset=['score_a', 'score_b']).reset_index(drop=True)
qo = """
SELECT o.event_id, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM odds_snapshots o
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m ON m.mid = o.id
"""
od = pd.read_sql(text(qo), eng)
df = df.merge(od, left_on='id', right_on='event_id', how='left')
def parse_goals(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        g = json.loads(s) if isinstance(s, str) else s
        return g if isinstance(g, list) else None
    except Exception:
        return None
df['goals'] = df.goals_json.apply(parse_goals)
df['total'] = (df.score_a + df.score_b).astype(int)
TL = df[df.goals.notna()].copy()
TL['tl_h'] = TL.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Home'))
TL['tl_a'] = TL.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Away'))
C = TL[(TL.tl_h == TL.score_a) & (TL.tl_a == TL.score_b)].copy()
print(f"clean: {len(C)}")

# ================================================================ M
print("=" * 70)
print("M — MINUTES HOME vs AWAY")
hm, am = [], []
for _, r in C.iterrows():
    for g in r.goals:
        (hm if g['team'] == 'Home' else am).append(int(g['minute']))
hm, am = np.array(hm), np.array(am)
print(f"home goals n={len(hm)} mean={hm.mean():.2f} | away goals n={len(am)} mean={am.mean():.2f}")
ks, p = stats.ks_2samp(hm, am)
print(f"KS home vs away minutes: D={ks:.4f} p={p:.3e}")
mw = stats.mannwhitneyu(hm, am)
print(f"Mann-Whitney: p={mw.pvalue:.3e}")
print("share <=45: home", round(np.mean(hm <= 45), 4), "away", round(np.mean(am <= 45), 4))

# ================================================================ N
print("\n" + "=" * 70)
print("N — ORDRE DES BUTS vs INTERLEAVING ALEATOIRE")
# pour chaque match avec h>0 et a>0 : qui marque le 1er vs proba h/(h+a)
obs_first = 0
ps = []
both = C[(C.tl_h > 0) & (C.tl_a > 0)]
for _, r in both.iterrows():
    g0 = min(r.goals, key=lambda g: int(g['minute']))
    obs_first += (g0['team'] == 'Home')
    ps.append(r.tl_h / (r.tl_h + r.tl_a))
ps = np.array(ps)
mu, var = ps.sum(), (ps * (1 - ps)).sum()
zN = (obs_first - mu) / np.sqrt(var)
print(f"matchs h>0 & a>0: n={len(both)} | home first obs={obs_first} attendu(random interleave)={mu:.1f} z={zN:+.2f} p={2*stats.norm.sf(abs(zN)):.3e}")
# détail par score
for (h, a) in [(1, 1), (2, 1), (1, 2), (2, 2), (3, 1), (1, 3), (3, 2), (2, 3)]:
    ss = both[(both.tl_h == h) & (both.tl_a == a)]
    if len(ss) < 50:
        continue
    of = sum(min(r.goals, key=lambda g: int(g['minute']))['team'] == 'Home' for _, r in ss.iterrows())
    pexp = h / (h + a)
    pb = stats.binomtest(of, len(ss), pexp)
    print(f"  score {h}-{a}: n={len(ss):4d} P(home first)={of/len(ss):.4f} vs random={pexp:.4f} p={pb.pvalue:.3e}")
# et le DERNIER but ? (symétrie)
obs_last = 0
for _, r in both.iterrows():
    gl = max(r.goals, key=lambda g: int(g['minute']))
    obs_last += (gl['team'] == 'Home')
zL = (obs_last - mu) / np.sqrt(var)
print(f"home LAST: obs={obs_last} attendu={mu:.1f} z={zL:+.2f} p={2*stats.norm.sf(abs(zL)):.3e}")

# ================================================================ O
print("\n" + "=" * 70)
print("O — MOMENTUM, NULL PER-(HALF, K_HALF)")
C['mins'] = C.goals.apply(lambda gs: sorted(int(g['minute']) for g in gs))
C['k1'] = C.mins.apply(lambda ms: sum(1 for m in ms if m <= 45))
C['k2'] = C.mins.apply(lambda ms: sum(1 for m in ms if m > 45))
pmf = {}
sup1, sup2 = np.arange(1, 46), np.arange(46, 91)
for half in (1, 2):
    for _, r in C.iterrows():
        pass
h1 = {}; h2 = {}
for _, r in C.iterrows():
    k1c, k2c = min(r.k1, 4), min(r.k2, 4)
    for m in r.mins:
        if m <= 45:
            h1.setdefault(k1c, []).append(m)
        else:
            h2.setdefault(k2c, []).append(m)
pmf1 = {k: np.bincount(v, minlength=46)[1:46] / len(v) for k, v in h1.items()}
pmf2 = {k: np.bincount(v, minlength=91)[46:91] / len(v) for k, v in h2.items()}
def stats_mins(list_ms):
    gaps = []; near5 = denom = 0
    for ms in list_ms:
        n = len(ms)
        if n >= 2:
            gaps.extend(ms[i + 1] - ms[i] for i in range(n - 1))
        for j in range(n):
            if ms[j] <= 85:
                denom += 1
                if j + 1 < n and ms[j + 1] - ms[j] <= 5:
                    near5 += 1
    return np.array(gaps), near5 / denom
gaps_o, mom_o = stats_mins(C.mins.tolist())
NSIM = 200
k1s, k2s = C.k1.values, C.k2.values
sim_mom, sim_p5 = [], []
for s in range(NSIM):
    sims = []
    for k1, k2 in zip(k1s, k2s):
        ms = []
        if k1:
            ms.extend(rng.choice(sup1, size=k1, p=pmf1[min(k1, 4)]))
        if k2:
            ms.extend(rng.choice(sup2, size=k2, p=pmf2[min(k2, 4)]))
        if ms:
            sims.append(sorted(ms))
    g, m = stats_mins(sims)
    sim_mom.append(m); sim_p5.append(np.mean(g <= 5))
def mc_p(o, sims):
    sims = np.array(sims)
    return min((np.sum(sims <= o) + 1) / (len(sims) + 1), (np.sum(sims >= o) + 1) / (len(sims) + 1)) * 2
print(f"OBS momentum={mom_o:.4f} P(gap<=5)={np.mean(gaps_o<=5):.4f}")
print(f"NULL per-(half,k): momentum={np.mean(sim_mom):.4f}±{np.std(sim_mom):.4f} z={(mom_o-np.mean(sim_mom))/np.std(sim_mom):+.2f} MC p={mc_p(mom_o, sim_mom):.4f}")
print(f"NULL per-(half,k): P(gap<=5)={np.mean(sim_p5):.4f}±{np.std(sim_p5):.4f} obs={np.mean(gaps_o<=5):.4f} MC p={mc_p(np.mean(gaps_o<=5), sim_p5):.4f}")

# ================================================================ P
print("\n" + "=" * 70)
print("P — FTTS '1' : SOUS-ENSEMBLES EXPLOITABLES ?")
def em_parse(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return None
df['em'] = df.extra_markets.apply(em_parse)
def get_ftts(em):
    if em and isinstance(em, dict):
        m = em.get('FTTS')
        if m and all(k in m for k in ['1', '2', 'Pas de but']):
            return m
    return None
df['ftts_odds'] = df.em.apply(get_ftts)
def ftts_outcome(row):
    if row.total == 0:
        return 'Pas de but'
    if row.goals is None or not row.goals:
        return None
    g0 = min(row.goals, key=lambda g: int(g['minute']))
    return '1' if g0['team'] == 'Home' else '2'
df['ftts_out'] = df.apply(ftts_outcome, axis=1)
S = df[df.ftts_odds.notna() & df.ftts_out.notna() & df.odds_home.notna()].sort_values('expected_start').reset_index(drop=True)
S['c1'] = S.ftts_odds.apply(lambda d: d['1'])
S['c2'] = S.ftts_odds.apply(lambda d: d['2'])
S['hit1'] = (S.ftts_out == '1')
print(f"n={len(S)} | ROI global FTTS'1'={(S.hit1*S.c1-1).mean()*100:+.2f}%")
ntr = int(len(S) * 0.7)
TR, TE = S.iloc[:ntr].copy(), S.iloc[ntr:].copy()
# par quintile de cote c1 sur train
TR['q'] = pd.qcut(TR.c1, 5, duplicates='drop')
print("\nROI train par quintile de cote FTTS'1':")
edges = None
grp = TR.groupby('q', observed=True)
res = grp.apply(lambda g: pd.Series({'n': len(g), 'WR': g.hit1.mean(), 'cote': g.c1.mean(), 'ROI%': (g.hit1 * g.c1 - 1).mean() * 100}), include_groups=False)
print(res.round(3).to_string())
# aussi par favori away (odds_away < odds_home)
TR['away_fav'] = TR.odds_away < TR.odds_home
TE['away_fav'] = TE.odds_away < TE.odds_home
for cond_name, cond_tr, cond_te in [
    ('away favori', TR.away_fav, TE.away_fav),
    ('home favori', ~TR.away_fav, ~TE.away_fav),
    ('c1 >= 2.0', TR.c1 >= 2.0, TE.c1 >= 2.0),
    ('c1 >= 2.2', TR.c1 >= 2.2, TE.c1 >= 2.2),
    ('c1 1.8-2.6', TR.c1.between(1.8, 2.6), TE.c1.between(1.8, 2.6)),
]:
    g_tr = TR[cond_tr]; g_te = TE[cond_te]
    roi_tr = (g_tr.hit1 * g_tr.c1 - 1).mean() * 100
    roi_te = (g_te.hit1 * g_te.c1 - 1).mean() * 100
    print(f"  bet '1' si {cond_name:12s}: train n={len(g_tr):4d} ROI={roi_tr:+6.2f}% | OOS n={len(g_te):4d} ROI={roi_te:+6.2f}% WR={g_te.hit1.mean():.3f} cote={g_te.c1.mean():.2f}")

# le biais d'ordre est-il constant ? P(home first | both score) par tranche de force relative
both2 = S[(S.score_a > 0) & (S.score_b > 0)].copy()
both2['rel'] = both2.odds_home / both2.odds_away
both2['hfirst'] = both2.ftts_out == '1'
both2['q'] = pd.qcut(both2.rel, 4)
print("\nP(home first | les 2 marquent) par quartile odds_home/odds_away:")
print(both2.groupby('q', observed=True).apply(lambda g: pd.Series({'n': len(g), 'P(hfirst)': g.hfirst.mean(), 'expected_random': (g.score_a / (g.score_a + g.score_b)).mean()}), include_groups=False).round(4).to_string())
print("\nDONE")
