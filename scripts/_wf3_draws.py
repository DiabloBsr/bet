# -*- coding: utf-8 -*-
"""
WF3 — LA STRUCTURE DES NULS (ligue 8035, Sporty-Tech)
1. Surface P(X) reelle vs P(X) marche vs P(X) Poisson independant (memes marges 1X2)
2. Distribution des scores de nul (0-0/1-1/2-2/3-3+) par tranche de cote draw
3. Nuls sur matchs tres desequilibres (fav <=1.3, draw 6-9) : reel vs implicite
4. Nul mi-temps (Mi-tps 1X2 'X') : calibration fine + combo HT/FT X/X
5. Edges -> walk-forward 70/30 chronologique
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from scraper.config import load_settings
from sqlalchemy import create_engine

pd.set_option('display.width', 200)
eng = create_engine(load_settings().db_url)

Q = """
SELECT e.id AS event_id, e.round_info, e.team_a, e.team_b, e.expected_start,
       os.id AS snap_id, os.odds_home, os.odds_draw, os.odds_away, os.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
FROM events e
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id = e.id
JOIN odds_snapshots os ON os.id = m.mid
JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
df = pd.read_sql(Q, eng)
df = df.sort_values('snap_id').drop_duplicates(['team_a', 'team_b', 'expected_start'], keep='first')
df = df.sort_values('expected_start').reset_index(drop=True)
df = df[(df.odds_home > 1) & (df.odds_draw > 1) & (df.odds_away > 1)].reset_index(drop=True)
N = len(df)
print(f"=== events dedup avec cotes+resultat : {N} ===")

# ---------- devig 1X2 (proportionnel) ----------
inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['overround'] = inv
df['p_home'] = (1/df.odds_home) / inv
df['p_draw'] = (1/df.odds_draw) / inv
df['p_away'] = (1/df.odds_away) / inv
df['is_draw'] = (df.score_a == df.score_b).astype(int)
df['draw_score'] = np.where(df.is_draw == 1, df.score_a.clip(upper=3), -1)
print(f"overround 1X2 : mean={inv.mean():.4f} std={inv.std():.4f}")
print(f"taux de nul global : {df.is_draw.mean():.4f}  | p_draw marche moyen : {df.p_draw.mean():.4f}")

# ---------- Poisson independant cale sur les memes marges 1X2 ----------
# On cherche (lh, la) tq P(H>A)=p_home et P(A>H)=p_away sous Poisson indep.
# Parametrisation: mu = lh+la, d = lh-la. Skellam via somme tronquee.
KMAX = 12
def skellam_probs(lh, la):
    ph = np.exp(-lh) * lh ** np.arange(KMAX+1) / np.array([math.factorial(k) for k in range(KMAX+1)])
    pa = np.exp(-la) * la ** np.arange(KMAX+1) / np.array([math.factorial(k) for k in range(KMAX+1)])
    M = np.outer(ph, pa)
    pH = np.tril(M, -1).sum(); pD = np.trace(M); pA = np.triu(M, 1).sum()
    s = pH + pD + pA
    return pH/s, pD/s, pA/s

def fit_poisson(ph_t, pa_t):
    """trouve (lh,la) reproduisant p_home et p_away ; retourne aussi p_draw poisson"""
    def err(params):
        lh, la = params
        pH, pD, pA = skellam_probs(lh, la)
        return (pH - ph_t), (pA - pa_t)
    # nested 1D: pour mu fixe, trouver d tq pH/(pH+pA) colle, puis ajuster mu pour le draw...
    # plus simple: 2D Newton grossier
    lh, la = 1.3, 1.1
    for _ in range(60):
        pH, pD, pA = skellam_probs(lh, la)
        e1, e2 = pH - ph_t, pA - pa_t
        if abs(e1) < 1e-7 and abs(e2) < 1e-7:
            break
        eps = 1e-4
        pH1, _, pA1 = skellam_probs(lh+eps, la)
        pH2, _, pA2 = skellam_probs(lh, la+eps)
        J = np.array([[(pH1-pH)/eps, (pH2-pH)/eps], [(pA1-pA)/eps, (pA2-pA)/eps]])
        try:
            d = np.linalg.solve(J, [e1, e2])
        except np.linalg.LinAlgError:
            break
        lh = float(np.clip(lh - d[0], 0.02, 8)); la = float(np.clip(la - d[1], 0.02, 8))
    return lh, la, skellam_probs(lh, la)[1]

# cache sur grille arrondie pour vitesse
cache = {}
lhs, las, pdraw_poiss = [], [], []
for ph_t, pa_t in zip(df.p_home.round(4), df.p_away.round(4)):
    key = (ph_t, pa_t)
    if key not in cache:
        cache[key] = fit_poisson(ph_t, pa_t)
    lh, la, pd_p = cache[key]
    lhs.append(lh); las.append(la); pdraw_poiss.append(pd_p)
df['lam_h'] = lhs; df['lam_a'] = las; df['p_draw_poisson'] = pdraw_poiss
df['mu'] = df.lam_h + df.lam_a

print("\n================ 1) SURFACE P(X) : reel vs marche vs Poisson ================")
print(f"lambda implicites (Poisson cale 1X2) : mu mean={df.mu.mean():.3f} [{df.mu.quantile(.05):.2f}-{df.mu.quantile(.95):.2f}]")
print(f"p_draw poisson moyen = {df.p_draw_poisson.mean():.4f} vs marche {df.p_draw.mean():.4f} vs reel {df.is_draw.mean():.4f}")

# calibration globale marche
n_draw = int(df.is_draw.sum())
p_bin = stats.binomtest(n_draw, N, df.p_draw.mean()).pvalue
print(f"binomial global (reel {n_draw}/{N}={n_draw/N:.4f} vs p_draw moy {df.p_draw.mean():.4f}) p={p_bin:.4f}")

# surface par bins de (p_home, p_away) -> on resume par |p_home - p_away| (desequilibre) et mu
df['skew'] = (df.p_home - df.p_away).abs()
df['fav_odds'] = df[['odds_home', 'odds_away']].min(axis=1)
bins_skew = [0, .1, .2, .3, .4, .5, .65, 1.0]
df['skew_bin'] = pd.cut(df.skew_bin if hasattr(df, 'skew_bin') else df['skew'], bins_skew, include_lowest=True)
tab = df.groupby('skew_bin', observed=True).agg(
    n=('is_draw', 'size'), real=('is_draw', 'mean'),
    market=('p_draw', 'mean'), poisson=('p_draw_poisson', 'mean'),
    draw_odds=('odds_draw', 'mean'))
tab['real-market'] = tab.real - tab.market
tab['real-poisson'] = tab.real - tab.poisson
tab['pval_vs_market'] = [stats.binomtest(int(r.real*r.n+0.5), int(r.n), r.market).pvalue for r in tab.itertuples()]
print("\n--- P(X) par desequilibre |p1-p2| ---")
print(tab.round(4).to_string())

# bins par cote draw directement (vue marchande)
bins_dx = [3.0, 3.3, 3.6, 4.0, 4.5, 5.5, 7.0, 12.0]
df['dx_bin'] = pd.cut(df.odds_draw, bins_dx, include_lowest=True)
tab2 = df.groupby('dx_bin', observed=True).agg(
    n=('is_draw', 'size'), real=('is_draw', 'mean'),
    market=('p_draw', 'mean'), poisson=('p_draw_poisson', 'mean'))
tab2['real-market'] = tab2.real - tab2.market
tab2['pval'] = [stats.binomtest(int(r.real*r.n+0.5), int(r.n), r.market).pvalue if r.n > 0 else 1 for r in tab2.itertuples()]
print("\n--- P(X) par cote draw ---")
print(tab2.round(4).to_string())

# le marche price-t-il le draw comme Poisson ? correlation des residus
resid_mkt = df.p_draw - df.p_draw_poisson
print(f"\np_draw_marche - p_draw_poisson : mean={resid_mkt.mean():+.4f} std={resid_mkt.std():.4f}")
print(f"  -> correlation(p_draw_mkt, p_draw_poisson) = {np.corrcoef(df.p_draw, df.p_draw_poisson)[0,1]:.4f}")
# test: lequel predit mieux le reel ? log-loss
eps = 1e-9
ll_mkt = -(df.is_draw*np.log(df.p_draw+eps) + (1-df.is_draw)*np.log(1-df.p_draw+eps)).mean()
ll_poi = -(df.is_draw*np.log(df.p_draw_poisson.clip(eps, 1-eps)) + (1-df.is_draw)*np.log(1-df.p_draw_poisson.clip(eps, 1-eps))).mean()
print(f"log-loss draw : marche={ll_mkt:.5f}  poisson={ll_poi:.5f}  (diff={ll_poi-ll_mkt:+.5f})")
# test apparie sur les contributions
contrib_diff = (-(df.is_draw*np.log(df.p_draw+eps) + (1-df.is_draw)*np.log(1-df.p_draw+eps))
                + (df.is_draw*np.log(df.p_draw_poisson.clip(eps, 1-eps)) + (1-df.is_draw)*np.log(1-df.p_draw_poisson.clip(eps, 1-eps))))
t, p_t = stats.ttest_1samp(contrib_diff, 0)
print(f"  t-test apparie (mkt meilleur si <0) : t={t:.2f} p={p_t:.4f}")

print("\n================ 2) SCORES DE NUL par tranche de cote draw ================")
dd = df[df.is_draw == 1]
ct = pd.crosstab(dd.dx_bin, dd.draw_score, normalize='index')
ct_n = pd.crosstab(dd.dx_bin, dd.draw_score)
ct.columns = [f"{int(c)}-{int(c)}" if c < 3 else "3-3+" for c in ct.columns]
print(f"nuls totaux: {len(dd)}")
print((ct.round(3)).to_string())
print("\nn par tranche:"); print(ct_n.sum(axis=1).to_string())
# vs Poisson : P(0-0|draw) attendue
dd2 = df[df.is_draw == 1].copy()
for k in range(4):
    pk = np.exp(-dd2.lam_h)*dd2.lam_h**k/math.factorial(k) * np.exp(-dd2.lam_a)*dd2.lam_a**k/math.factorial(k)
    dd2[f'p{k}{k}'] = pk
dd2['pdrawtot'] = dd2.p_draw_poisson
exp_frac = dd2.groupby('dx_bin', observed=True).apply(
    lambda g: pd.Series({f"{k}-{k}": (g[f'p{k}{k}']/g.pdrawtot).mean() for k in range(3)}), include_groups=False)
print("\n--- attendu Poisson P(k-k | draw) ---")
print(exp_frac.round(3).to_string())
# chi2 global obs vs poisson-expected sur scores 0-0..2-2,3+
obs = ct_n.values.astype(float)
# expected: pour chaque tranche, n * frac poisson
rows = []
for b, g in dd2.groupby('dx_bin', observed=True):
    n = len(g)
    e = [float((g[f'p{k}{k}']/g.pdrawtot).mean())*n for k in range(3)]
    e.append(max(n - sum(e), 1e-9))
    rows.append(e)
exp = np.array(rows)
if obs.shape == exp.shape:
    chi2 = ((obs - exp)**2/np.maximum(exp, 1e-9)).sum()
    dof = (obs.shape[0]-1)*(obs.shape[1]-1)
    print(f"\nchi2 obs-vs-poisson scores de nul : chi2={chi2:.1f} dof~{dof} p={1-stats.chi2.cdf(chi2, dof):.4g}")

# Score exact 'X-X' du marche vs reel (les cotes CS draw sont-elles calibrees ?)
def parse_em(s):
    if s is None: return {}
    return json.loads(s) if isinstance(s, str) else s
df['em'] = df.extra_markets.apply(parse_em)
cs_rows = []
for r in df.itertuples():
    cs = r.em.get('Score exact') or {}
    for k in ['0-0', '1-1', '2-2', '3-3']:
        o = cs.get(k)
        if o and o > 1 and o < 100:
            h, a = map(int, k.split('-'))
            cs_rows.append((r.dx_bin, k, o, 1.0 if (r.score_a == h and r.score_b == a) else 0.0))
cs = pd.DataFrame(cs_rows, columns=['dx_bin', 'score', 'odds', 'hit'])
g = cs.groupby('score').agg(n=('hit', 'size'), real=('hit', 'mean'), implied=('odds', lambda s: (1/s).mean()))
g['edge_brut'] = g.real - g.implied
g['pval'] = [stats.binomtest(int(r.real*r.n+0.5), int(r.n), r.implied).pvalue for r in g.itertuples()]
print("\n--- Score exact k-k : reel vs 1/cote (brut, avec marge) ---")
print(g.round(4).to_string())

print("\n================ 3) NULS SUR MATCHS TRES DESEQUILIBRES ================")
for lo, hi in [(1.0, 1.20), (1.20, 1.30), (1.30, 1.40)]:
    sub = df[(df.fav_odds > lo) & (df.fav_odds <= hi)]
    if len(sub) == 0: continue
    n, nd = len(sub), int(sub.is_draw.sum())
    pm = sub.p_draw.mean(); pp = sub.p_draw_poisson.mean()
    pv = stats.binomtest(nd, n, pm).pvalue
    roi = (sub.is_draw*sub.odds_draw - 1).mean()
    print(f"fav ({lo:.2f},{hi:.2f}] : n={n:4d} draws={nd:3d} reel={nd/n:.4f} mkt={pm:.4f} poiss={pp:.4f} "
          f"cote_draw_moy={sub.odds_draw.mean():.2f} p_binom={pv:.4f} ROI_backX={roi:+.4f}")
sub = df[(df.fav_odds <= 1.30) & (df.odds_draw >= 6) & (df.odds_draw <= 9)]
n, nd = len(sub), int(sub.is_draw.sum())
if n:
    pm = sub.p_draw.mean()
    print(f"\nfav<=1.30 & draw 6-9 : n={n} draws={nd} reel={nd/n:.4f} implicite_devig={pm:.4f} "
          f"1/cote_brut={ (1/sub.odds_draw).mean():.4f} p_binom={stats.binomtest(nd, n, pm).pvalue:.4f}")
    print(f"  ROI back draw plat : {(sub.is_draw*sub.odds_draw - 1).mean():+.4f}")
    print(f"  ROI lay draw (commission 0) : {((1-sub.is_draw)*1 - sub.is_draw*(sub.odds_draw-1)).mean():+.4f}")

print("\n================ 4) NUL MI-TEMPS ================")
ht = df.dropna(subset=['ht_score_a', 'ht_score_b']).copy()
ht['is_htdraw'] = (ht.ht_score_a == ht.ht_score_b).astype(int)
ht['ht_x_odds'] = ht.em.apply(lambda e: (e.get('Mi-tps 1X2') or {}).get('X'))
ht['htft_xx_odds'] = ht.em.apply(lambda e: (e.get('HT/FT') or {}).get('X/X'))
ht = ht[(ht.ht_x_odds > 1)].copy()
# devig mi-temps
ht['ht_1'] = ht.em.apply(lambda e: (e.get('Mi-tps 1X2') or {}).get('1'))
ht['ht_2'] = ht.em.apply(lambda e: (e.get('Mi-tps 1X2') or {}).get('2'))
ht = ht.dropna(subset=['ht_1', 'ht_2'])
invh = 1/ht.ht_1 + 1/ht.ht_x_odds + 1/ht.ht_2
ht['p_htx'] = (1/ht.ht_x_odds)/invh
print(f"n={len(ht)} | HT draw reel={ht.is_htdraw.mean():.4f} | p_htx devig moyen={ht.p_htx.mean():.4f} "
      f"| overround HT moy={invh.mean():.4f}")
print(f"binomial global p={stats.binomtest(int(ht.is_htdraw.sum()), len(ht), ht.p_htx.mean()).pvalue:.4f}")

bins_htx = [1.8, 2.1, 2.25, 2.4, 2.6, 3.0, 4.5]
ht['htx_bin'] = pd.cut(ht.ht_x_odds, bins_htx, include_lowest=True)
t4 = ht.groupby('htx_bin', observed=True).agg(n=('is_htdraw', 'size'), real=('is_htdraw', 'mean'),
                                              devig=('p_htx', 'mean'), brut=('ht_x_odds', lambda s: (1/s).mean()))
t4['real-devig'] = t4.real - t4.devig
t4['pval'] = [stats.binomtest(int(r.real*r.n+0.5), int(r.n), r.devig).pvalue if r.n else 1 for r in t4.itertuples()]
t4['ROI_back'] = [((ht[ht.htx_bin == i].is_htdraw*ht[ht.htx_bin == i].ht_x_odds) - 1).mean() for i in t4.index]
print("\n--- calibration X-HT par cote ---")
print(t4.round(4).to_string())

# par profil de match (fav_odds)
ht['fav_bin'] = pd.cut(ht.fav_odds, [1.0, 1.3, 1.6, 2.0, 2.6, 10], include_lowest=True)
t5 = ht.groupby('fav_bin', observed=True).agg(n=('is_htdraw', 'size'), real=('is_htdraw', 'mean'), devig=('p_htx', 'mean'))
t5['diff'] = t5.real - t5.devig
t5['pval'] = [stats.binomtest(int(r.real*r.n+0.5), int(r.n), r.devig).pvalue for r in t5.itertuples()]
print("\n--- X-HT par cote du favori FT ---")
print(t5.round(4).to_string())

# lien HT-FT : P(X FT | X HT) vs P(X FT) — persistance
ht['is_ftdraw'] = (ht.score_a == ht.score_b).astype(int)
p_x_given_xht = ht[ht.is_htdraw == 1].is_ftdraw.mean()
p_x_given_noxht = ht[ht.is_htdraw == 0].is_ftdraw.mean()
print(f"\nP(X FT | X HT) = {p_x_given_xht:.4f}  vs P(X FT | non-X HT) = {p_x_given_noxht:.4f}  vs P(X FT) = {ht.is_ftdraw.mean():.4f}")

# combo X/X (HT/FT)
hx = ht[(ht.htft_xx_odds > 1) & (ht.htft_xx_odds < 50)].copy()
hx['hit_xx'] = ((hx.ht_score_a == hx.ht_score_b) & (hx.score_a == hx.score_b)).astype(int)
n, nh = len(hx), int(hx.hit_xx.sum())
imp = (1/hx.htft_xx_odds).mean()
print(f"\nHT/FT X/X : n={n} hits={nh} reel={nh/n:.4f} 1/cote_brut={imp:.4f} "
      f"p_binom={stats.binomtest(nh, n, imp).pvalue:.4g}")
print(f"  ROI back X/X plat : {(hx.hit_xx*hx.htft_xx_odds - 1).mean():+.4f}  cote moy={hx.htft_xx_odds.mean():.2f}")
hx['xx_bin'] = pd.cut(hx.htft_xx_odds, [4, 5.5, 6.5, 7.5, 9, 12, 50], include_lowest=True)
t6 = hx.groupby('xx_bin', observed=True).agg(n=('hit_xx', 'size'), real=('hit_xx', 'mean'),
                                             brut=('htft_xx_odds', lambda s: (1/s).mean()))
t6['ROI'] = [((hx[hx.xx_bin == i].hit_xx*hx[hx.xx_bin == i].htft_xx_odds) - 1).mean() for i in t6.index]
t6['pval'] = [stats.binomtest(int(r.real*r.n+0.5), int(r.n), r.brut).pvalue if r.n else 1 for r in t6.itertuples()]
print(t6.round(4).to_string())

# independance conditionnelle : le moteur tire-t-il HT et 2T independamment ?
# si independant: P(X/X) = P(XHT) * P(2T draw)  avec 2T draw = (score2t_h==score2t_a)
ht['h2_h'] = ht.score_a - ht.ht_score_a; ht['h2_a'] = ht.score_b - ht.ht_score_b
ht['is_2tdraw'] = (ht.h2_h == ht.h2_a).astype(int)
p_joint = ((ht.is_htdraw == 1) & (ht.is_2tdraw == 1)).mean()
p_indep = ht.is_htdraw.mean() * ht.is_2tdraw.mean()
ctab = pd.crosstab(ht.is_htdraw, ht.is_2tdraw)
chi2, p_chi, _, _ = stats.chi2_contingency(ctab)
print(f"\nindependance HT-draw vs 2T-draw : P(joint)={p_joint:.4f} vs P(indep)={p_indep:.4f} chi2={chi2:.2f} p={p_chi:.4g}")

print("\n================ 5) WALK-FORWARD des candidats ================")
df = df.sort_values('expected_start').reset_index(drop=True)
cut = int(N*0.7)
train, test = df.iloc[:cut], df.iloc[cut:]
print(f"train={len(train)} test={len(test)} (split chrono)")

def wf_report(name, mask_fn, odds_col, hit_col, dtrain, dtest):
    tr = dtrain[mask_fn(dtrain)]; te = dtest[mask_fn(dtest)]
    if len(tr) == 0 or len(te) == 0:
        print(f"{name}: vide"); return None
    roi_tr = (tr[hit_col]*tr[odds_col] - 1).mean()
    roi_te = (te[hit_col]*te[odds_col] - 1).mean()
    wr_te = te[hit_col].mean()
    print(f"{name}: train n={len(tr)} ROI={roi_tr:+.4f} | OOS n={len(te)} ROI={roi_te:+.4f} WR={wr_te:.4f} cote_moy={te[odds_col].mean():.2f}")
    return dict(name=name, n_oos=len(te), roi_oos=roi_te, wr_oos=wr_te, avg=te[odds_col].mean())

df['hit_draw'] = df.is_draw.astype(float)
train, test = df.iloc[:cut], df.iloc[cut:]

# candidat A : back X quand p_draw_poisson > p_draw_marche + delta (modele > marche)
for delta in [0.0, 0.01, 0.02]:
    wf_report(f"A backX poiss>mkt+{delta}", lambda d, dl=delta: (d.p_draw_poisson - d.p_draw) > dl,
              'odds_draw', 'hit_draw', train, test)

# candidat B : back X sur desequilibres extremes si bord identifie
wf_report("B backX fav<=1.30", lambda d: d.fav_odds <= 1.30, 'odds_draw', 'hit_draw', train, test)
wf_report("B' layX fav<=1.30 (ROI lay)", lambda d: d.fav_odds <= 1.30, 'odds_draw', 'hit_draw', train, test)

# candidat C : X-HT back par tranche (sur ht frame)
ht = ht.sort_values('expected_start').reset_index(drop=True)
cuth = int(len(ht)*0.7)
htr, hte = ht.iloc[:cuth], ht.iloc[cuth:]
ht['hit_htx'] = ht.is_htdraw.astype(float)
htr, hte = ht.iloc[:cuth], ht.iloc[cuth:]
for lo, hi in [(1.8, 2.25), (2.25, 2.6), (2.6, 4.5)]:
    wf_report(f"C backX-HT cote({lo},{hi}]", lambda d, l=lo, h=hi: (d.ht_x_odds > l) & (d.ht_x_odds <= h),
              'ht_x_odds', 'hit_htx', htr, hte)

# candidat D : HT/FT X/X
hx = hx.sort_values('expected_start').reset_index(drop=True)
cutx = int(len(hx)*0.7)
hx['hit'] = hx.hit_xx.astype(float)
xtr, xte = hx.iloc[:cutx], hx.iloc[cutx:]
wf_report("D backX/X tous", lambda d: d.htft_xx_odds > 0, 'htft_xx_odds', 'hit', xtr, xte)
wf_report("D' backX/X cote<=7", lambda d: d.htft_xx_odds <= 7, 'htft_xx_odds', 'hit', xtr, xte)

# candidat E : Score exact k-k si bord trouve en (2) — back 0-0 / 1-1 selon edge train
cs_full = []
for r in df.itertuples():
    csm = r.em.get('Score exact') or {}
    for k, (hh, aa) in {'0-0': (0, 0), '1-1': (1, 1), '2-2': (2, 2)}.items():
        o = csm.get(k)
        if o and 1 < o < 100:
            cs_full.append((r.expected_start, k, o, 1.0 if (r.score_a == hh and r.score_b == aa) else 0.0))
csf = pd.DataFrame(cs_full, columns=['expected_start', 'score', 'odds', 'hit']).sort_values('expected_start')
for sc in ['0-0', '1-1', '2-2']:
    s = csf[csf.score == sc].reset_index(drop=True)
    c = int(len(s)*0.7)
    tr, te = s.iloc[:c], s.iloc[c:]
    roi_tr = (tr.hit*tr.odds - 1).mean(); roi_te = (te.hit*te.odds - 1).mean()
    print(f"E backCS {sc}: train n={len(tr)} ROI={roi_tr:+.4f} | OOS n={len(te)} ROI={roi_te:+.4f} WR={te.hit.mean():.4f} cote={te.odds.mean():.2f}")

print("\ndone")
