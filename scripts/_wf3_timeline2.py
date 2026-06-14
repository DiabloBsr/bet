# -*- coding: utf-8 -*-
"""WF3 — TIMELINE follow-up.

A. 0-0 : existe-t-il dans results ? goals_json NULL pattern vs score.
B. Marginale des minutes PAR nombre de buts N — invariance ? (architecture score-first)
C. Null per-N : momentum + gaps re-testés
D. Conditionnel E[total | fenêtre 1er but] : obs vs null per-N (test architecture)
E. HT mismatches : pattern (ht=0:0 placeholder ?), cohérence FT des mêmes lignes
F. 'Total de buts' : calibration + ROI (le bucket '0' ne sort jamais ?) + walk-forward
G. Fit rampes linéaires par mi-temps (paramètres moteur)
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(7)
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

# ================================================================ A
print("=" * 70)
print("A — LE 0-0 EXISTE-T-IL ?")
print("dedup matches:", len(df))
zz = df[(df.score_a == 0) & (df.score_b == 0)]
print(f"0-0 dans results (dedup): {len(zz)}  ({len(zz)/len(df)*100:.2f}%)")
print("  parmi eux, goals_json NULL:", zz.goals.isna().sum(), "| goals_json []:", (zz.goals.apply(lambda g: g is not None and len(g) == 0)).sum())
print("total distribution:", df.total.value_counts().sort_index().to_dict())
print("goals_json NULL par total:")
print(df.groupby('total').goals.apply(lambda s: s.isna().mean()).round(4).to_dict())

# subset propre : timeline cohérente FT
TL = df[df.goals.notna()].copy()
TL['tl_h'] = TL.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Home'))
TL['tl_a'] = TL.goals.apply(lambda gs: sum(1 for g in gs if g['team'] == 'Away'))
TL['ft_ok'] = (TL.tl_h == TL.score_a) & (TL.tl_a == TL.score_b)
print(f"\nTL={len(TL)}, FT-coherent={TL.ft_ok.sum()} ({TL.ft_ok.mean()*100:.1f}%)")
C = TL[TL.ft_ok].copy()   # clean subset pour B-D

# ================================================================ E
print("\n" + "=" * 70)
print("E — PATTERN DES MISMATCHES")
ht = TL[TL.ht_score_a.notna()].copy()
def ht_tl(gs, team):
    return sum(1 for g in gs if int(g['minute']) <= 45 and g['team'] == team)
ht['tlh1'] = ht.goals.apply(lambda gs: ht_tl(gs, 'Home'))
ht['tla1'] = ht.goals.apply(lambda gs: ht_tl(gs, 'Away'))
ht['ht_ok'] = (ht.tlh1 == ht.ht_score_a) & (ht.tla1 == ht.ht_score_b)
mis = ht[~ht.ht_ok]
print(f"HT mismatches: {len(mis)}/{len(ht)}")
print("  dont ht_score==0:0 :", ((mis.ht_score_a == 0) & (mis.ht_score_b == 0)).sum())
print("  dont FT aussi incoherent :", (~mis.ft_ok).sum())
print("  FT mismatches total:", (~ht.ft_ok).sum(), "| dont HT aussi mismatch:", (~ht.ft_ok & ~ht.ht_ok).sum())
ftm = ht[~ht.ft_ok]
print("  FT mismatch: score résultat vs timeline (10 ex):")
for _, r in ftm.head(10).iterrows():
    print(f"    id={r.id} result {int(r.score_a)}-{int(r.score_b)} ht {int(r.ht_score_a)}-{int(r.ht_score_b)} timeline {r.tl_h}-{r.tl_a} mins={[int(g['minute']) for g in r.goals]}")
# ht=0:0 alors que timeline a des buts <=45 ET ft_ok -> placeholder HT
ph = mis[mis.ft_ok & (mis.ht_score_a == 0) & (mis.ht_score_b == 0) & ((mis.tlh1 + mis.tla1) > 0)]
print(f"  HT-mismatch avec FT ok et ht=0:0 et timeline a des buts H1 (=placeholder HT): {len(ph)}")

# ================================================================ B
print("\n" + "=" * 70)
print("B — MARGINALE DES MINUTES PAR N (architecture score-first ?)")
rows = []
for _, r in C.iterrows():
    n = len(r.goals)
    for g in r.goals:
        rows.append((r.id, n, int(g['minute'])))
GG = pd.DataFrame(rows, columns=['eid', 'n', 'minute'])
print("clean goals:", len(GG))
blocks = [(1, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
tab = []
for n in range(1, 8):
    gn = GG[GG.n == (n if n < 7 else GG.n.max())] if False else GG[GG.n == n]
    if len(gn) < 100:
        continue
    shares = [gn.minute.between(a, b).mean() for a, b in blocks]
    tab.append([n, len(gn)] + [round(s, 4) for s in shares])
T = pd.DataFrame(tab, columns=['N', 'n_goals'] + [f"{a}-{b}" for a, b in blocks])
print(T.to_string(index=False))
# chi2 homogeneity minutes x N (blocks)
ct = pd.crosstab(GG.n.clip(upper=6), pd.cut(GG.minute, [0, 15, 30, 45, 60, 75, 90]))
chi2h, ph2, dof, _ = stats.chi2_contingency(ct)
print(f"chi2 homogénéité blocs x N(clip6): chi2={chi2h:.1f} dof={dof} p={ph2:.3e}")

# ================================================================ C + D
print("\n" + "=" * 70)
print("C/D — NULL PER-N : minutes iid de la marginale conditionnelle à N")
# marginales par N (clip à 6+)
GG['nc'] = GG.n.clip(upper=6)
pmf_by_n = {}
for n, gn in GG.groupby('nc'):
    cnt = np.bincount(gn.minute.values, minlength=91)[1:91].astype(float)
    pmf_by_n[n] = cnt / cnt.sum()
support = np.arange(1, 91)

match_ns = C.goals.apply(len).values
match_ns_c = np.clip(match_ns, 0, 6)

# observed stats on clean subset
def stats_from_minutes(list_of_sorted_minutes):
    gaps = []
    near5 = denom = 0
    fw_tot = {b: [0, 0.0] for b in blocks}  # window -> [count, sum totals]
    for ms in list_of_sorted_minutes:
        n = len(ms)
        if n == 0:
            continue
        if n >= 2:
            gaps.extend((ms[i + 1] - ms[i]) for i in range(n - 1))
        for j in range(n):
            if ms[j] <= 85:
                denom += 1
                if j + 1 < n and ms[j + 1] - ms[j] <= 5:
                    near5 += 1
        f = ms[0]
        for a, b in blocks:
            if a <= f <= b:
                fw_tot[(a, b)][0] += 1
                fw_tot[(a, b)][1] += n
                break
    gaps = np.array(gaps)
    return gaps, near5 / denom, {w: (v[1] / v[0] if v[0] else np.nan, v[0]) for w, v in fw_tot.items()}

obs_minutes = [sorted(int(g['minute']) for g in gs) for gs in C.goals if len(gs) > 0]
gaps_o, mom_o, fw_o = stats_from_minutes(obs_minutes)
print(f"OBS clean: n_gaps={len(gaps_o)} mean_gap={gaps_o.mean():.2f} P(gap<=5)={np.mean(gaps_o<=5):.4f} P(gap=0)={np.mean(gaps_o==0):.4f} momentum={mom_o:.4f}")

NSIM = 300
sim_mom, sim_p5, sim_p0, sim_gapmean = [], [], [], []
sim_fw = {w: [] for w in blocks}
for s in range(NSIM):
    sims = []
    for n in match_ns:
        if n == 0:
            continue
        nc = min(n, 6)
        sims.append(np.sort(rng.choice(support, size=n, p=pmf_by_n[nc])).tolist())
    g, m, fw = stats_from_minutes(sims)
    sim_gapmean.append(g.mean()); sim_p5.append(np.mean(g <= 5)); sim_p0.append(np.mean(g == 0)); sim_mom.append(m)
    for w in blocks:
        sim_fw[w].append(fw[w][0])
def mc_p(o, sims):
    sims = np.array(sims)
    return min((np.sum(sims <= o) + 1) / (len(sims) + 1), (np.sum(sims >= o) + 1) / (len(sims) + 1)) * 2
print(f"NULL per-N: momentum={np.mean(sim_mom):.4f}±{np.std(sim_mom):.4f} -> z={(mom_o-np.mean(sim_mom))/np.std(sim_mom):+.2f} MC p={mc_p(mom_o, sim_mom):.4f}")
print(f"NULL per-N: P(gap<=5)={np.mean(sim_p5):.4f}±{np.std(sim_p5):.4f} obs={np.mean(gaps_o<=5):.4f} -> MC p={mc_p(np.mean(gaps_o<=5), sim_p5):.4f}")
print(f"NULL per-N: P(gap=0)={np.mean(sim_p0):.4f}±{np.std(sim_p0):.4f} obs={np.mean(gaps_o==0):.4f} -> MC p={mc_p(np.mean(gaps_o==0), sim_p0):.4f}")
print(f"NULL per-N: mean gap={np.mean(sim_gapmean):.2f}±{np.std(sim_gapmean):.2f} obs={gaps_o.mean():.2f} -> MC p={mc_p(gaps_o.mean(), sim_gapmean):.4f}")
print("\nD — E[total | fenêtre 1er but] : obs vs null per-N")
for w in blocks:
    sims = np.array(sim_fw[w])
    o, n_o = fw_o[w]
    print(f"  1er but {w[0]:2d}-{w[1]:2d}: obs E[total]={o:.3f} (n={n_o})  null={sims.mean():.3f}±{sims.std():.3f}  z={(o-sims.mean())/sims.std():+.2f}")

# ================================================================ F
print("\n" + "=" * 70)
print("F — 'Total de buts' : CALIBRATION + ROI")
def get_tdb(em_raw):
    try:
        em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
        t = em.get('Total de buts')
        if t and all(str(k) in t for k in range(7)):
            return {int(k): float(v) for k, v in t.items()}
    except Exception:
        pass
    return None
df['tdb'] = df.extra_markets.apply(get_tdb)
S = df[df.tdb.notna()].sort_values('expected_start').reset_index(drop=True)
S['tot_b'] = S.total.clip(upper=6)
print(f"n with market: {len(S)} (NB: '6' = 6+ supposé)")
print("bucket | n_hit | freq_emp | 1/cote_mean | avg_cote | ROI full | binom p")
for k in range(7):
    hits = (S.tot_b == k)
    cotes = S.tdb.apply(lambda d: d[k])
    roi = (hits * cotes - 1).mean()
    pbin = stats.binomtest(int(hits.sum()), len(S), float((1 / cotes).mean())).pvalue
    print(f"  {k}    | {hits.sum():5d} | {hits.mean():.4f} | {(1/cotes).mean():.4f} | {cotes.mean():6.2f} | {roi*100:+7.2f}% | {pbin:.3e}")
ntr = int(len(S) * 0.7)
TR, TE = S.iloc[:ntr], S.iloc[ntr:]
print(f"walk-forward train={len(TR)} OOS={len(TE)}")
for k in range(7):
    htr = (TR.tot_b == k); ctr = TR.tdb.apply(lambda d: d[k])
    hte = (TE.tot_b == k); cte = TE.tdb.apply(lambda d: d[k])
    roi_tr = (htr * ctr - 1).mean(); roi_te = (hte * cte - 1).mean()
    flag = " <== candidate" if roi_tr > 0.02 else ""
    print(f"  {k} | train={roi_tr*100:+7.2f}% | OOS={roi_te*100:+7.2f}% (hits={hte.sum()}/{len(TE)}, avg cote={cte.mean():.2f}){flag}")

# 0-0 jamais ? verif croisée Score exact
zz_all = df[(df.score_a == 0) & (df.score_b == 0)]
print(f"\n0-0 total (toutes lignes results dedup): {len(zz_all)}")

# ================================================================ G
print("\n" + "=" * 70)
print("G — FIT RAMPES PAR MI-TEMPS (paramètres moteur)")
mins_all = GG.minute.values
cnt = np.bincount(mins_all, minlength=91)[1:91].astype(float)
mm = np.arange(1, 91)
for lo, hi, nm in [(1, 45, 'H1'), (46, 90, 'H2')]:
    x = mm[(mm >= lo) & (mm <= hi)]
    y = cnt[lo - 1:hi]
    sl, ic, rv, pv, se = stats.linregress(x, y)
    yfit = sl * x + ic
    chi2g, pg = stats.chisquare(y, yfit * y.sum() / yfit.sum())
    print(f"{nm}: slope={sl:.3f}±{se:.3f} intercept={ic:.1f} r={rv:.3f} | rate(min {lo})={sl*lo+ic:.0f} -> rate(min {hi})={sl*hi+ic:.0f} ratio={(sl*hi+ic)/(sl*lo+ic):.2f} | GOF lin chi2={chi2g:.1f} df={len(x)-2} p={pg:.3e}")
# share H1 vs H2
print(f"share H1={np.mean(mins_all<=45):.4f} H2={np.mean(mins_all>45):.4f}")
print("\nDONE")
