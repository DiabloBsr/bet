# -*- coding: utf-8 -*-
"""WF3 — Facette TIMELINE du moteur Sporty-Tech (ligue 8035).

Sections:
  S1  Distribution des minutes de but (1-90), pics 45/90, KS/chi2
  S2  Espacement entre buts consécutifs vs null iid (Monte Carlo)
  S3  Momentum : P(but dans les 5 min après un but) vs null
  S4  Late goals 80'+ : favori vs underdog, équipe menée
  S5  P(over 2.5 | 1er but <15') vs marché
  S6  Cohérence ht_score vs goals_json (minute<=45)
  S7  Marché 'Minute du premier but' : calibration + ROI + walk-forward
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(42)
eng = create_engine(load_settings().db_url)

# ---------------------------------------------------------------- load
q = """
SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start, e.competition,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0'
"""
df = pd.read_sql(text(q), eng)
print("rows raw:", len(df), "| competitions:", df.competition.value_counts().to_dict())
df = df.sort_values('id').drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df = df.dropna(subset=['score_a', 'score_b']).reset_index(drop=True)
print("rows dedup:", len(df))

# opening odds (MIN snapshot id per event)
qo = """
SELECT o.event_id, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM odds_snapshots o
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m
  ON m.mid = o.id
"""
od = pd.read_sql(text(qo), eng)
df = df.merge(od, left_on='id', right_on='event_id', how='left')
print("with opening odds:", df.odds_home.notna().sum())

def parse_goals(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    try:
        g = json.loads(s) if isinstance(s, str) else s
        return g if isinstance(g, list) else None
    except Exception:
        return None

df['goals'] = df.goals_json.apply(parse_goals)
has_tl = df.goals.notna()
print("with goals_json timeline:", has_tl.sum())

# sanity: timeline total vs score
mask_ok = []
for _, r in df[has_tl].iterrows():
    mask_ok.append(len(r.goals) == (r.score_a + r.score_b))
print("timeline count == FT total:", sum(mask_ok), "/", len(mask_ok))

TL = df[has_tl].reset_index(drop=True)

# flat goals table
rows = []
for i, r in TL.iterrows():
    for g in r.goals:
        rows.append((r.id, int(g['minute']), g['team'], g.get('homeScore'), g.get('awayScore')))
G = pd.DataFrame(rows, columns=['event_id', 'minute', 'team', 'hs', 'as_'])
print("total goals:", len(G), "| minute min/max:", G.minute.min(), G.minute.max())

# ================================================================ S1
print("\n" + "=" * 70)
print("S1 — DISTRIBUTION DES MINUTES")
mins = G.minute.values
cnt = np.bincount(mins, minlength=max(92, mins.max() + 1))
print("minute counts 1..10:", cnt[1:11].tolist())
print("minute counts 40..50:", list(zip(range(40, 51), cnt[40:51].tolist())))
print("minute counts 85..91:", list(zip(range(85, 92), cnt[85:92].tolist())))
print("any minute 0:", cnt[0], "| any >90:", cnt[91:].sum())

# chi2 vs uniform on observed support 1..90
obs = cnt[1:91].astype(float)
exp = np.full(90, obs.sum() / 90)
chi2, p = stats.chisquare(obs, exp)
print(f"chi2 uniform(1-90): chi2={chi2:.1f} df=89 p={p:.3e}")

# KS vs uniform (continuous approx with jitter)
u = (mins - 1 + rng.random(len(mins))) / 90.0
ks, pks = stats.kstest(u, 'uniform')
print(f"KS vs uniform: D={ks:.4f} p={pks:.3e}")

# halves and linear trend
h1, h2 = (mins <= 45).sum(), (mins > 45).sum()
pb = stats.binomtest(int(h2), int(h1 + h2), 0.5)
print(f"1st half {h1} vs 2nd half {h2} -> P(2nd)={h2/(h1+h2):.4f} binom p={pb.pvalue:.3e}")
# linear regression of counts on minute (excluding 45 & 90 spikes)
mm = np.arange(1, 91)
mask_lin = (mm != 45) & (mm != 90)
sl, ic, rv, pv, se = stats.linregress(mm[mask_lin], obs[mask_lin])
print(f"linear trend (sans 45/90): slope={sl:.3f} buts/min/min (r={rv:.3f}, p={pv:.3e}); intercept={ic:.1f}")
# spike tests: 45 and 90 vs local neighbors
for spike, neigh in [(45, [41, 42, 43, 44]), (90, [86, 87, 88, 89])]:
    base = obs[np.array(neigh) - 1].mean()
    pp = stats.poisson.sf(obs[spike - 1] - 1, base)
    print(f"minute {spike}: n={int(obs[spike-1])} vs voisins mean={base:.1f} -> Poisson p={pp:.2e} (x{obs[spike-1]/base:.2f})")
# piecewise: per 15-min block
for b in range(6):
    lo, hi = b * 15 + 1, b * 15 + 15
    print(f"  block {lo:2d}-{hi:2d}: {obs[lo-1:hi].sum():6.0f} goals  ({obs[lo-1:hi].sum()/obs.sum()*100:.2f}%)  mean/min={obs[lo-1:hi].mean():.1f}")

# two-segment uniform fit (rate1 1-45, rate2 46-90)
exp2 = np.concatenate([np.full(45, obs[:45].mean()), np.full(45, obs[45:].mean())])
chi2b, pb2 = stats.chisquare(obs, exp2 * obs.sum() / exp2.sum())
print(f"GOF 2-paliers (1-45 / 46-90): chi2={chi2b:.1f} df=88 p={pb2:.3e}")
# 2-paliers + spikes 45/90 retirés
o3 = obs[mask_lin]
e3 = np.where(mm[mask_lin] <= 45, obs[:44].mean(), obs[45:89].mean())
chi2c, pc = stats.chisquare(o3, e3 * o3.sum() / e3.sum())
print(f"GOF 2-paliers sans minutes 45/90: chi2={chi2c:.1f} df={len(o3)-1} p={pc:.3e}")
# linear GOF sans spikes
elin = sl * mm[mask_lin] + ic
chi2l, pl = stats.chisquare(o3, elin * o3.sum() / elin.sum())
print(f"GOF lineaire sans minutes 45/90: chi2={chi2l:.1f} df={len(o3)-2} p={pl:.3e}")
p_marginal = obs / obs.sum()   # empirical minute pmf used for nulls

# ================================================================ S2
print("\n" + "=" * 70)
print("S2 — ESPACEMENT ENTRE BUTS CONSECUTIFS")
gaps_obs = []
n_goals_per_match = []
for _, r in TL.iterrows():
    ms = sorted(int(g['minute']) for g in r.goals)
    n_goals_per_match.append(len(ms))
    gaps_obs.extend(np.diff(ms).tolist())
gaps_obs = np.array(gaps_obs)
print(f"n gaps={len(gaps_obs)} mean={gaps_obs.mean():.2f} median={np.median(gaps_obs)} P(gap=0)={np.mean(gaps_obs==0):.4f} P(gap<=5)={np.mean(gaps_obs<=5):.4f}")

# null: same N per match, minutes iid from empirical marginal
NSIM = 400
minutes_support = np.arange(1, 91)
def sim_once():
    gaps = []
    near5 = 0; denom = 0
    for n in n_goals_per_match:
        if n == 0:
            continue
        ms = np.sort(rng.choice(minutes_support, size=n, p=p_marginal))
        if n >= 2:
            gaps.append(np.diff(ms))
        for j in range(n):
            if ms[j] <= 85:
                denom += 1
                if j + 1 < n and (ms[j + 1] - ms[j]) <= 5:
                    near5 += 1
    g = np.concatenate(gaps) if gaps else np.array([])
    return g, near5, denom

sim_gap_means, sim_p0, sim_p5, sim_mom = [], [], [], []
sim_gaps_pool = []
for s in range(NSIM):
    g, n5, dn = sim_once()
    sim_gap_means.append(g.mean()); sim_p0.append(np.mean(g == 0)); sim_p5.append(np.mean(g <= 5))
    sim_mom.append(n5 / dn)
    if s < 50:
        sim_gaps_pool.append(g)
sim_gaps_pool = np.concatenate(sim_gaps_pool)
def mc_p(obs_val, sims):
    sims = np.array(sims)
    return min((np.sum(sims <= obs_val) + 1) / (len(sims) + 1), (np.sum(sims >= obs_val) + 1) / (len(sims) + 1)) * 2
print(f"null gap mean={np.mean(sim_gap_means):.2f}±{np.std(sim_gap_means):.2f}  obs={gaps_obs.mean():.2f} -> MC p={mc_p(gaps_obs.mean(), sim_gap_means):.4f}")
print(f"null P(gap=0)={np.mean(sim_p0):.4f}±{np.std(sim_p0):.4f}  obs={np.mean(gaps_obs==0):.4f} -> MC p={mc_p(np.mean(gaps_obs==0), sim_p0):.4f}")
print(f"null P(gap<=5)={np.mean(sim_p5):.4f}±{np.std(sim_p5):.4f}  obs={np.mean(gaps_obs<=5):.4f} -> MC p={mc_p(np.mean(gaps_obs<=5), sim_p5):.4f}")
ks2, pks2 = stats.ks_2samp(gaps_obs, sim_gaps_pool)
print(f"KS 2-sample gaps obs vs null pool: D={ks2:.4f} p={pks2:.3e}")
# exponential check on gaps>0
gpos = gaps_obs[gaps_obs > 0].astype(float)
ksexp, pksexp = stats.kstest((gpos - 0.5 + rng.random(len(gpos))) / gpos.mean(), 'expon')
print(f"KS gaps>0 vs exponentielle(mean): D={ksexp:.4f} p={pksexp:.3e}")

# ================================================================ S3
print("\n" + "=" * 70)
print("S3 — MOMENTUM : P(but dans les 5 min suivant un but)")
near5 = 0; denom = 0
for _, r in TL.iterrows():
    ms = sorted(int(g['minute']) for g in r.goals)
    for j in range(len(ms)):
        if ms[j] <= 85:
            denom += 1
            if j + 1 < len(ms) and (ms[j + 1] - ms[j]) <= 5:
                near5 += 1
p_obs = near5 / denom
print(f"obs: {near5}/{denom} = {p_obs:.4f}")
print(f"null (N fixe/match, minutes iid): {np.mean(sim_mom):.4f}±{np.std(sim_mom):.4f} -> MC p={mc_p(p_obs, sim_mom):.4f}")
z = (p_obs - np.mean(sim_mom)) / np.std(sim_mom)
print(f"z vs null = {z:+.2f}")

# ================================================================ S4
print("\n" + "=" * 70)
print("S4 — LATE GOALS 80'+ : FAVORI OU UNDERDOG ?")
TLo = TL[TL.odds_home.notna() & (TL.odds_home != TL.odds_away)]
fav_map = {r.id: ('Home' if r.odds_home < r.odds_away else 'Away') for _, r in TLo.iterrows()}
Gf = G[G.event_id.isin(fav_map)].copy()
Gf['is_fav'] = Gf.apply(lambda x: x.team == fav_map[x.event_id], axis=1)
late = Gf.minute >= 80
tab = pd.crosstab(late, Gf.is_fav)
print(tab)
p_fav_early = Gf[~late].is_fav.mean(); p_fav_late = Gf[late].is_fav.mean()
chi2f, pf, _, _ = stats.chi2_contingency(tab)
print(f"P(but par favori) early(<80)={p_fav_early:.4f} vs late(80+)={p_fav_late:.4f}  chi2 p={pf:.4f}")
# trailing team effect
def pre_state(row):
    if row.hs is None or row.as_ is None:
        return None
    hs, as_ = int(row.hs), int(row.as_)
    if row.team == 'Home':
        hs -= 1
        diff = hs - as_
    else:
        as_ -= 1
        diff = as_ - hs
    return 'trailing' if diff < 0 else ('level' if diff == 0 else 'leading')
G['state'] = G.apply(pre_state, axis=1)
for nm, gg in [('early(<80)', G[G.minute < 80]), ('late(80+)', G[G.minute >= 80])]:
    vc = gg.state.value_counts(normalize=True)
    print(f"{nm}: scorer trailing={vc.get('trailing',0):.4f} level={vc.get('level',0):.4f} leading={vc.get('leading',0):.4f} (n={len(gg)})")
tab2 = pd.crosstab(G.minute >= 80, G.state == 'trailing')
chi2t, pt, _, _ = stats.chi2_contingency(tab2)
print(f"chi2 'scorer trailing' late vs early: p={pt:.4f}")

# ================================================================ S5
print("\n" + "=" * 70)
print("S5 — P(over 2.5 | 1er but <15') vs MARCHE")
def implied_over25(em_raw):
    try:
        em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
        t = em.get('Total de buts')
        if not t:
            return None
        inv = {k: 1.0 / v for k, v in t.items() if v and v > 0}
        s = sum(inv.values())
        return sum(pp for k, pp in inv.items() if k.isdigit() and int(k) >= 3) / s
    except Exception:
        return None
TL['imp_o25'] = TL.extra_markets.apply(implied_over25)
TL['total'] = (TL.score_a + TL.score_b).astype(int)
TL['first_min'] = TL.goals.apply(lambda g: min(int(x['minute']) for x in g) if g else None)
sub = TL[TL.imp_o25.notna()]
print(f"n with 'Total de buts' market: {len(sub)}")
print(f"unconditional: empirical P(o2.5)={(sub.total>=3).mean():.4f}  market implied mean={sub.imp_o25.mean():.4f}")
early1 = sub[sub.first_min.notna() & (sub.first_min < 15)]
print(f"| 1er but <15': n={len(early1)} empirical P(o2.5)={(early1.total>=3).mean():.4f} vs implied prematch mean={early1.imp_o25.mean():.4f}")
bt = stats.binomtest(int((early1.total >= 3).sum()), len(early1), early1.imp_o25.mean())
print(f"  binomial vs implied prematch: p={bt.pvalue:.3e}")
for w in [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]:
    ss = sub[sub.first_min.notna() & sub.first_min.between(w[0], w[1])]
    if len(ss) > 30:
        print(f"  1er but {w[0]:2d}-{w[1]:2d}: n={len(ss):4d} P(o2.5)={(ss.total>=3).mean():.4f} P(o3.5)={(ss.total>=4).mean():.4f} total mean={ss.total.mean():.2f} implied_o25={ss.imp_o25.mean():.4f}")
nob = sub[sub.first_min.isna()]
print(f"  pas de but: n={len(nob)}")

# ================================================================ S6
print("\n" + "=" * 70)
print("S6 — COHERENCE ht_score vs goals_json(minute<=45)")
ht = TL[TL.ht_score_a.notna() & TL.ht_score_b.notna()]
mism = []
for _, r in ht.iterrows():
    h = sum(1 for g in r.goals if int(g['minute']) <= 45 and g['team'] == 'Home')
    a = sum(1 for g in r.goals if int(g['minute']) <= 45 and g['team'] == 'Away')
    if h != int(r.ht_score_a) or a != int(r.ht_score_b):
        mism.append((r.id, int(r.ht_score_a), int(r.ht_score_b), h, a,
                     sorted(int(g['minute']) for g in r.goals)))
print(f"matches avec HT dispo: {len(ht)} | mismatch: {len(mism)} ({len(mism)/max(len(ht),1)*100:.2f}%)")
for m in mism[:15]:
    print("  ", m)
for cut in (44, 46):
    bad = 0
    for _, r in ht.iterrows():
        h = sum(1 for g in r.goals if int(g['minute']) <= cut and g['team'] == 'Home')
        a = sum(1 for g in r.goals if int(g['minute']) <= cut and g['team'] == 'Away')
        if h != int(r.ht_score_a) or a != int(r.ht_score_b):
            bad += 1
    print(f"mismatch avec cutoff {cut}: {bad}")
bad_ft = 0
for _, r in TL.iterrows():
    h = sum(1 for g in r.goals if g['team'] == 'Home')
    a = sum(1 for g in r.goals if g['team'] == 'Away')
    if h != int(r.score_a) or a != int(r.score_b):
        bad_ft += 1
print(f"FT mismatch (timeline vs score): {bad_ft}/{len(TL)}")

# ================================================================ S7
print("\n" + "=" * 70)
print("S7 — MARCHE 'Minute du premier but' : CALIBRATION + ROI")
BUCKETS = ['1-15', '16-30', '31-45', '46-60', '61-75', '76-90', 'Pas de but']
def fg_bucket(fm):
    if fm is None or (isinstance(fm, float) and np.isnan(fm)):
        return 'Pas de but'
    fm = int(fm)
    for b in BUCKETS[:-1]:
        lo, hi = map(int, b.split('-'))
        if lo <= fm <= hi:
            return b
    return '76-90' if fm > 90 else None
def get_fg_odds(em_raw):
    try:
        em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
        m = em.get('Minute du premier but')
        if m and all(b in m for b in BUCKETS):
            return m
    except Exception:
        pass
    return None
TL['fg_odds'] = TL.extra_markets.apply(get_fg_odds)
TL['fg_bucket'] = TL.first_min.apply(fg_bucket)
S = TL[TL.fg_odds.notna()].sort_values('expected_start').reset_index(drop=True)
print(f"n with market: {len(S)}")
print("\nbucket | n_hit | freq_emp | implied_norm_mean | avg_cote | ROI flat full | binom p (vs 1/cote)")
for b in BUCKETS:
    hits = (S.fg_bucket == b)
    cotes = S.fg_odds.apply(lambda d: d[b])
    inv_all = S.fg_odds.apply(lambda d: sum(1.0 / v for v in d.values()))
    implied = (1.0 / cotes) / inv_all
    roi = (hits * cotes - 1).mean()
    pbin = stats.binomtest(int(hits.sum()), len(S), float((1.0 / cotes).mean())).pvalue
    print(f"{b:>10} | {hits.sum():4d} | {hits.mean():.4f} | {implied.mean():.4f} | {cotes.mean():6.2f} | {roi*100:+6.2f}% | {pbin:.3e}")
ovr = S.fg_odds.apply(lambda d: sum(1.0 / v for v in d.values()))
print(f"overround moyen marché FG: {ovr.mean():.4f}")

# walk-forward
ntr = int(len(S) * 0.7)
TR, TE = S.iloc[:ntr], S.iloc[ntr:]
print(f"\nwalk-forward: train n={len(TR)} / OOS n={len(TE)}")
for b in BUCKETS:
    hits_tr = (TR.fg_bucket == b); cotes_tr = TR.fg_odds.apply(lambda d: d[b])
    roi_tr = (hits_tr * cotes_tr - 1).mean()
    hits_te = (TE.fg_bucket == b); cotes_te = TE.fg_odds.apply(lambda d: d[b])
    roi_te = (hits_te * cotes_te - 1).mean()
    flag = " <== candidate (ROI train >2%)" if roi_tr > 0.02 else ""
    print(f"{b:>10} | ROI train={roi_tr*100:+6.2f}% | ROI OOS={roi_te*100:+6.2f}% (hits OOS={hits_te.sum()}/{len(TE)}, WR={hits_te.mean():.4f}, avg cote={cotes_te.mean():.2f}){flag}")

print("\nDONE")
