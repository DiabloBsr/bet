# -*- coding: utf-8 -*-
"""WF3 — TIMELINE part 3.

H. Redo 'Minute du premier but' avec sample corrigé (0-0 inclus via total==0)
I. Cross-market : P(0-0) pricée identiquement (Score exact 0-0, TdB 0, FTTS PdB, FG PdB) ?
J. FTTS calibration (1/2/Pas de but) + ROI + walk-forward
K. Architecture par mi-temps : homogénéité placement intra-half vs k_half ;
   momentum re-testé sous null intra-half (k1,k2 fixés)
L. Détail N=1 : histogramme minute du but unique
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(11)
eng = create_engine(load_settings().db_url)

q = """
SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
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
def em_parse(s):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except Exception:
        return None
df['em'] = df.extra_markets.apply(em_parse)

# first-goal minute: timeline si dispo, sinon NaN ; total==0 -> pas de but
df['first_min'] = df.goals.apply(lambda g: (min(int(x['minute']) for x in g) if g else np.nan) if g is not None else np.nan)
df['fg_known'] = (df.total == 0) | df.first_min.notna()

# ================================================================ H
print("=" * 70)
print("H — 'Minute du premier but' REDO (0-0 inclus)")
BUCKETS = ['1-15', '16-30', '31-45', '46-60', '61-75', '76-90', 'Pas de but']
def fg_bucket(row):
    if row.total == 0:
        return 'Pas de but'
    if np.isnan(row.first_min):
        return None
    fm = int(row.first_min)
    for b in BUCKETS[:-1]:
        lo, hi = map(int, b.split('-'))
        if lo <= fm <= hi:
            return b
    return None
def get_fg_odds(em):
    if em and isinstance(em, dict):
        m = em.get('Minute du premier but')
        if m and all(b in m for b in BUCKETS):
            return m
    return None
df['fg_odds'] = df.em.apply(get_fg_odds)
df['fg_bucket'] = df.apply(fg_bucket, axis=1)
S = df[df.fg_odds.notna() & df.fg_bucket.notna()].sort_values('expected_start').reset_index(drop=True)
print(f"n: {len(S)} (dont 0-0: {(S.fg_bucket=='Pas de but').sum()})")
print("bucket | n_hit | freq_emp | implied_raw | avg_cote | ROI full | binom p")
for b in BUCKETS:
    hits = (S.fg_bucket == b)
    cotes = S.fg_odds.apply(lambda d: d[b])
    roi = (hits * cotes - 1).mean()
    pbin = stats.binomtest(int(hits.sum()), len(S), float((1 / cotes).mean())).pvalue
    print(f"{b:>10} | {hits.sum():4d} | {hits.mean():.4f} | {(1/cotes).mean():.4f} | {cotes.mean():6.2f} | {roi*100:+7.2f}% | {pbin:.3e}")
ntr = int(len(S) * 0.7)
TR, TE = S.iloc[:ntr], S.iloc[ntr:]
print(f"walk-forward train={len(TR)} OOS={len(TE)}")
for b in BUCKETS:
    htr = (TR.fg_bucket == b); ctr = TR.fg_odds.apply(lambda d: d[b])
    hte = (TE.fg_bucket == b); cte = TE.fg_odds.apply(lambda d: d[b])
    roi_tr = (htr * ctr - 1).mean(); roi_te = (hte * cte - 1).mean()
    flag = " <== candidate" if roi_tr > 0.02 else ""
    print(f"{b:>10} | train={roi_tr*100:+7.2f}% | OOS={roi_te*100:+7.2f}% (hits={hte.sum()}/{len(TE)}, WR={hte.mean():.4f}, cote={cte.mean():.2f}){flag}")

# ================================================================ I
print("\n" + "=" * 70)
print("I — COHERENCE CROSS-MARKET DE P(0-0)")
def p00_quotes(em):
    if not em or not isinstance(em, dict):
        return None
    out = {}
    se = em.get('Score exact', {});  out['SE 0-0'] = se.get('0-0')
    td = em.get('Total de buts', {}); out['TdB 0'] = td.get('0')
    ft = em.get('FTTS', {});          out['FTTS PdB'] = ft.get('Pas de but')
    fg = em.get('Minute du premier but', {}); out['FG PdB'] = fg.get('Pas de but')
    return out if all(v is not None for v in out.values()) else None
qq = df.em.apply(p00_quotes).dropna()
Q = pd.DataFrame(qq.tolist())
print(f"n with all 4 quotes: {len(Q)}")
print(Q.describe().loc[['mean', 'std', 'min', 'max']].round(3).to_string())
same = ((Q['SE 0-0'] == Q['TdB 0']) & (Q['TdB 0'] == Q['FTTS PdB']) & (Q['FTTS PdB'] == Q['FG PdB'])).mean()
print(f"part des matchs où les 4 cotes 0-0 identiques: {same:.4f}")
for c in Q.columns[1:]:
    d = (Q[c] - Q['SE 0-0'])
    print(f"  {c} - SE0-0: mean={d.mean():+.3f} max abs={d.abs().max():.3f}")

# ================================================================ J
print("\n" + "=" * 70)
print("J — FTTS (premier buteur) : CALIBRATION + ROI")
def ftts_outcome(row):
    if row.total == 0:
        return 'Pas de but'
    if row.goals is None or not row.goals:
        return None
    g0 = sorted(row.goals, key=lambda g: int(g['minute']))[0]
    return '1' if g0['team'] == 'Home' else '2'
def get_ftts(em):
    if em and isinstance(em, dict):
        m = em.get('FTTS')
        if m and all(k in m for k in ['1', '2', 'Pas de but']):
            return m
    return None
df['ftts_odds'] = df.em.apply(get_ftts)
df['ftts_out'] = df.apply(ftts_outcome, axis=1)
S2 = df[df.ftts_odds.notna() & df.ftts_out.notna()].sort_values('expected_start').reset_index(drop=True)
print(f"n: {len(S2)}")
for k in ['1', '2', 'Pas de but']:
    hits = (S2.ftts_out == k)
    cotes = S2.ftts_odds.apply(lambda d: d[k])
    roi = (hits * cotes - 1).mean()
    pbin = stats.binomtest(int(hits.sum()), len(S2), float((1 / cotes).mean())).pvalue
    print(f"{k:>10} | {hits.sum():4d} | freq={hits.mean():.4f} | implied_raw={(1/cotes).mean():.4f} | cote={cotes.mean():5.2f} | ROI={roi*100:+7.2f}% | p={pbin:.3e}")
ntr = int(len(S2) * 0.7)
TR2, TE2 = S2.iloc[:ntr], S2.iloc[ntr:]
for k in ['1', '2', 'Pas de but']:
    htr = (TR2.ftts_out == k); ctr = TR2.ftts_odds.apply(lambda d: d[k])
    hte = (TE2.ftts_out == k); cte = TE2.ftts_odds.apply(lambda d: d[k])
    print(f"{k:>10} | train={(htr*ctr-1).mean()*100:+7.2f}% | OOS={(hte*cte-1).mean()*100:+7.2f}% (hits={hte.sum()}/{len(TE2)}, cote={cte.mean():.2f})")

# ================================================================ K
print("\n" + "=" * 70)
print("K — ARCHITECTURE PAR MI-TEMPS")
TL = df[df.goals.notna()].copy()
TL['tl_h'] = TL.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Home'))
TL['tl_a'] = TL.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Away'))
C = TL[(TL.tl_h == TL.score_a) & (TL.tl_a == TL.score_b)].copy()
C['mins'] = C.goals.apply(lambda gs: sorted(int(g['minute']) for g in gs))
C['k1'] = C.mins.apply(lambda ms: sum(1 for m in ms if m <= 45))
C['k2'] = C.mins.apply(lambda ms: sum(1 for m in ms if m > 45))
print(f"clean matches: {len(C)}")
# corr k1,k2
print(f"corr(k1,k2) = {C.k1.corr(C.k2):+.4f} (p={stats.pearsonr(C.k1, C.k2)[1]:.3e})")
ct12 = pd.crosstab(C.k1.clip(upper=4), C.k2.clip(upper=4))
chi2_12, p12, dof12, _ = stats.chi2_contingency(ct12)
print(f"chi2 indep k1 x k2: chi2={chi2_12:.1f} dof={dof12} p={p12:.3e}")

# homogeneite intra-half: minute-block x k_half
h1_goals, h2_goals = [], []
for _, r in C.iterrows():
    for m in r.mins:
        if m <= 45:
            h1_goals.append((r.k1, m))
        else:
            h2_goals.append((r.k2, m))
H1 = pd.DataFrame(h1_goals, columns=['k', 'm']); H2 = pd.DataFrame(h2_goals, columns=['k', 'm'])
ctH1 = pd.crosstab(H1.k.clip(upper=4), pd.cut(H1.m, [0, 15, 30, 45]))
c1, pH1, d1, _ = stats.chi2_contingency(ctH1)
ctH2 = pd.crosstab(H2.k.clip(upper=4), pd.cut(H2.m, [45, 60, 75, 90]))
c2, pH2, d2, _ = stats.chi2_contingency(ctH2)
print(f"H1: placement (blocs 15') homogène selon k1 ? chi2={c1:.1f} dof={d1} p={pH1:.3e}")
print(ctH1.apply(lambda r: r / r.sum(), axis=1).round(3).to_string())
print(f"H2: placement homogène selon k2 ? chi2={c2:.1f} dof={d2} p={pH2:.3e}")
print(ctH2.apply(lambda r: r / r.sum(), axis=1).round(3).to_string())

# momentum sous null intra-half : (k1,k2) fixes, minutes iid pmf H1 / pmf H2
pmf1 = np.bincount(H1.m.values, minlength=46)[1:46].astype(float); pmf1 /= pmf1.sum()
pmf2 = np.bincount(H2.m.values, minlength=91)[46:91].astype(float); pmf2 /= pmf2.sum()
sup1, sup2 = np.arange(1, 46), np.arange(46, 91)
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
obs_ms = C.mins.tolist()
gaps_o, mom_o = stats_mins(obs_ms)
print(f"\nOBS: momentum={mom_o:.4f} P(gap<=5)={np.mean(gaps_o<=5):.4f} P(gap=0)={np.mean(gaps_o==0):.4f}")
NSIM = 300
k1s, k2s = C.k1.values, C.k2.values
sim_mom, sim_p5, sim_p0 = [], [], []
for s in range(NSIM):
    sims = []
    for k1, k2 in zip(k1s, k2s):
        ms = []
        if k1:
            ms.extend(rng.choice(sup1, size=k1, p=pmf1))
        if k2:
            ms.extend(rng.choice(sup2, size=k2, p=pmf2))
        if ms:
            sims.append(sorted(ms))
    g, m = stats_mins(sims)
    sim_mom.append(m); sim_p5.append(np.mean(g <= 5)); sim_p0.append(np.mean(g == 0))
def mc_p(o, sims):
    sims = np.array(sims)
    return min((np.sum(sims <= o) + 1) / (len(sims) + 1), (np.sum(sims >= o) + 1) / (len(sims) + 1)) * 2
print(f"NULL intra-half (k1,k2 fixes): momentum={np.mean(sim_mom):.4f}±{np.std(sim_mom):.4f} z={(mom_o-np.mean(sim_mom))/np.std(sim_mom):+.2f} MC p={mc_p(mom_o, sim_mom):.4f}")
print(f"NULL intra-half: P(gap<=5)={np.mean(sim_p5):.4f}±{np.std(sim_p5):.4f} obs={np.mean(gaps_o<=5):.4f} MC p={mc_p(np.mean(gaps_o<=5), sim_p5):.4f}")
print(f"NULL intra-half: P(gap=0)={np.mean(sim_p0):.4f}±{np.std(sim_p0):.4f} obs={np.mean(gaps_o==0):.4f} MC p={mc_p(np.mean(gaps_o==0), sim_p0):.4f}")

# ht_score marche 'Mi-tps CS' : verifier que k1 split home/away suit la grille HT
# (juste stats descriptives HT)
print(f"\ndistribution k1: {C.k1.value_counts(normalize=True).sort_index().round(4).to_dict()}")
print(f"distribution k2: {C.k2.value_counts(normalize=True).sort_index().round(4).to_dict()}")
print(f"E[k1]={C.k1.mean():.3f} E[k2]={C.k2.mean():.3f} ratio={C.k2.mean()/C.k1.mean():.3f}")

# ================================================================ L
print("\n" + "=" * 70)
print("L — N=1 : minute du but unique (histo par 5 min)")
one = C[C.mins.apply(len) == 1].mins.apply(lambda ms: ms[0])
h, edges = np.histogram(one, bins=range(1, 97, 5))
for i in range(len(h)):
    print(f"  {edges[i]:2d}-{edges[i+1]-1:2d}: {h[i]:4d} {'#' * (h[i] // 5)}")
print(f"n={len(one)}")
print("\nDONE")
