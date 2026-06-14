# -*- coding: utf-8 -*-
"""
WF3 draws — iteration 2
A. Le triple 1X2 vit-il EXACTEMENT sur la variete Poisson independante 2D ? (max |diff|)
   + quantization des lambdas / triples uniques (fingerprint moteur)
B. Les scores reels suivent-ils la grille 'Score exact' devig plutot que Poisson(lh,la) ?
   (conditionnel au nul : chi2 vs grille, chi2 vs poisson)
C. Favoris extremes : calibration 1X2 complete par bin de cote favori + DC '12' / DC fav
D. Walk-forward : DC '12' sur fav<=1.20 ; back favori extreme ; back outsider ?
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from scraper.config import load_settings
from sqlalchemy import create_engine

pd.set_option('display.width', 220)
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
inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['p_home'] = (1/df.odds_home)/inv; df['p_draw'] = (1/df.odds_draw)/inv; df['p_away'] = (1/df.odds_away)/inv
df['is_draw'] = (df.score_a == df.score_b).astype(int)

KMAX = 14
FACT = np.array([math.factorial(k) for k in range(KMAX+1)], dtype=float)
def pois_vec(lam):
    return np.exp(-lam)*lam**np.arange(KMAX+1)/FACT
def skellam(lh, la):
    M = np.outer(pois_vec(lh), pois_vec(la))
    s = M.sum()
    return np.tril(M, -1).sum()/s, np.trace(M)/s, np.triu(M, 1).sum()/s

def fit_poisson(ph_t, pa_t):
    lh, la = 1.3, 1.1
    for _ in range(80):
        pH, pD, pA = skellam(lh, la)
        e1, e2 = pH-ph_t, pA-pa_t
        if abs(e1) < 1e-10 and abs(e2) < 1e-10:
            break
        eps = 1e-5
        pH1, _, pA1 = skellam(lh+eps, la)
        pH2, _, pA2 = skellam(lh, la+eps)
        J = np.array([[(pH1-pH)/eps, (pH2-pH)/eps], [(pA1-pA)/eps, (pA2-pA)/eps]])
        try:
            d = np.linalg.solve(J, [e1, e2])
        except np.linalg.LinAlgError:
            break
        lh = float(np.clip(lh-d[0], 0.02, 9)); la = float(np.clip(la-d[1], 0.02, 9))
    return lh, la, skellam(lh, la)[1]

print("================ A) VARIETE POISSON & FINGERPRINT ================")
cache = {}
res = []
for ph_t, pa_t in zip(df.p_home.values, df.p_away.values):
    key = (round(ph_t, 6), round(pa_t, 6))
    if key not in cache:
        cache[key] = fit_poisson(ph_t, pa_t)
    res.append(cache[key])
df['lam_h'] = [r[0] for r in res]; df['lam_a'] = [r[1] for r in res]; df['p_draw_pois'] = [r[2] for r in res]
diff = (df.p_draw - df.p_draw_pois).abs()
print(f"|p_draw_mkt - p_draw_poisson_fit| : mean={diff.mean():.6f} max={diff.max():.6f} q99={diff.quantile(.99):.6f}")
print(f"  -> {int((diff < 0.001).sum())}/{N} matchs avec ecart < 0.001")

# triples uniques / quantization
trip = df[['odds_home', 'odds_draw', 'odds_away']].round(2)
uniq = trip.drop_duplicates()
print(f"triples 1X2 uniques : {len(uniq)} sur {N} matchs")
lh_r = df.lam_h.round(2); la_r = df.lam_a.round(2)
print(f"lambda_h : uniques@0.01={df.lam_h.round(2).nunique()} mean={df.lam_h.mean():.3f} [{df.lam_h.min():.2f},{df.lam_h.max():.2f}]")
print(f"lambda_a : uniques@0.01={df.lam_a.round(2).nunique()} mean={df.lam_a.mean():.3f} [{df.lam_a.min():.2f},{df.lam_a.max():.2f}]")
print(f"mu=lh+la : mean={(df.lam_h+df.lam_a).mean():.3f} std={(df.lam_h+df.lam_a).std():.3f}")
print(f"home adv lh-la : mean={(df.lam_h-df.lam_a).mean():.3f} std={(df.lam_h-df.lam_a).std():.3f}")
print(f"corr(lam_h, lam_a) = {np.corrcoef(df.lam_h, df.lam_a)[0,1]:+.3f}")
# le total implicite (mu) depend-il du desequilibre ?
df['skw'] = (df.p_home - df.p_away).abs()
df['mu'] = df.lam_h + df.lam_a
print("mu par skew bin:")
print(df.groupby(pd.cut(df["skw"], [0, .2, .4, .6, 1]), observed=True)['mu'].agg(['mean', 'std', 'count']).round(3).to_string())

# le reel suit-il les lambdas ? buts reels vs mu
print(f"\nbuts reels mean={(df.score_a+df.score_b).mean():.3f} vs mu mean={df.mu.mean():.3f}")
print(f"home goals mean={df.score_a.mean():.3f} vs lam_h={df.lam_h.mean():.3f} ; away {df.score_b.mean():.3f} vs lam_a={df.lam_a.mean():.3f}")
# variance/dispersion
print(f"var(home goals)={df.score_a.var():.3f} (Poisson => ~mean) ; var(away)={df.score_b.var():.3f}")
# correlation des buts H/A (independance ?)
rho, p_rho = stats.pearsonr(df.score_a, df.score_b)
print(f"corr(score_a, score_b) = {rho:+.4f} p={p_rho:.4f}")
# partial: residuals vs lambda
ra = df.score_a - df.lam_h; rb = df.score_b - df.lam_a
rho2, p_rho2 = stats.pearsonr(ra, rb)
print(f"corr residus (score-lambda) = {rho2:+.4f} p={p_rho2:.4f}  <- test independance conditionnelle")

print("\n================ B) SCORES REELS : GRILLE CS vs POISSON ================")
def parse_em(s):
    if s is None: return {}
    return json.loads(s) if isinstance(s, str) else s
df['em'] = df.extra_markets.apply(parse_em)

# grille CS devig (cellules < 100 seulement) -> proba de chaque cellule
def cs_grid(em):
    cs = em.get('Score exact') or {}
    cells = {}
    for k, o in cs.items():
        try:
            h, a = map(int, k.split('-'))
        except Exception:
            continue
        if o and 1 < o < 100:
            cells[(h, a)] = 1.0/o
    s = sum(cells.values())
    if s <= 0: return {}
    return {k: v/s for k, v in cells.items()}

# log-loss du score reel : grille CS vs Poisson(lh,la)
ll_cs, ll_po, used = [], [], 0
cond_draw_cs = np.zeros(4)   # P(0-0),P(1-1),P(2-2),P(3-3+) attendu par grille, conditionne au nul
cond_draw_po = np.zeros(4)
cond_draw_obs = np.zeros(4)
n_draws_used = 0
for r in df.itertuples():
    grid = cs_grid(r.em)
    if not grid:
        continue
    used += 1
    ph = pois_vec(r.lam_h); pa = pois_vec(r.lam_a)
    # conditionnel nul (grille)
    gd = {k: v for k, v in grid.items() if k[0] == k[1]}
    sgd = sum(gd.values())
    pod = [ph[k]*pa[k] for k in range(4)]
    spod = sum(ph[k]*pa[k] for k in range(KMAX+1))
    if r.is_draw:
        n_draws_used += 1
        kk = min(int(r.score_a), 3)
        cond_draw_obs[kk] += 1
        for k in range(3):
            cond_draw_cs[k] += gd.get((k, k), 0)/sgd if sgd > 0 else 0
            cond_draw_po[k] += pod[k]/spod
        cond_draw_cs[3] += (1 - sum(gd.get((k, k), 0) for k in range(3))/sgd) if sgd > 0 else 0
        cond_draw_po[3] += 1 - sum(pod[:3])/spod
    key = (int(r.score_a), int(r.score_b))
    p_cs = grid.get(key, 0.002)
    p_po = ph[key[0]]*pa[key[1]] if key[0] <= KMAX and key[1] <= KMAX else 1e-4
    ll_cs.append(-math.log(max(p_cs, 1e-6)))
    ll_po.append(-math.log(max(p_po, 1e-6)))
ll_cs = np.array(ll_cs); ll_po = np.array(ll_po)
print(f"matchs avec grille CS: {used}")
print(f"log-loss score exact : grille_devig={ll_cs.mean():.4f}  poisson_fit={ll_po.mean():.4f}")
t, p_t = stats.ttest_rel(ll_cs, ll_po)
print(f"  t-test apparie (grille meilleure si t<0) : t={t:.2f} p={p_t:.4g}")

print(f"\nconditionnel au nul (n={n_draws_used}) :")
obs = cond_draw_obs
exp_cs = cond_draw_cs/n_draws_used*obs.sum()
exp_po = cond_draw_po/n_draws_used*obs.sum()
print(f"  obs      : {obs/obs.sum()}")
print(f"  grille   : {exp_cs/exp_cs.sum()}")
print(f"  poisson  : {exp_po/exp_po.sum()}")
chi_cs = ((obs-exp_cs)**2/np.maximum(exp_cs, 1e-9)).sum()
chi_po = ((obs-exp_po)**2/np.maximum(exp_po, 1e-9)).sum()
print(f"  chi2 vs grille = {chi_cs:.1f} (p={1-stats.chi2.cdf(chi_cs,3):.4g}) ; chi2 vs poisson = {chi_po:.1f} (p={1-stats.chi2.cdf(chi_po,3):.4g})")

# la grille CS devig draw total vs p_draw 1X2
pdraw_cs = []
for r in df.itertuples():
    grid = cs_grid(r.em)
    if grid:
        pdraw_cs.append(sum(v for k, v in grid.items() if k[0] == k[1]))
    else:
        pdraw_cs.append(np.nan)
df['p_draw_cs'] = pdraw_cs
d2 = (df.p_draw_cs - df.p_draw).dropna()
print(f"\np_draw(grille CS devig) - p_draw(1X2 devig) : mean={d2.mean():+.4f} std={d2.std():.4f} max|.|={d2.abs().max():.4f}")

print("\n================ C) FAVORIS EXTREMES : CALIBRATION 1X2 COMPLETE ================")
df['fav_odds'] = df[['odds_home', 'odds_away']].min(axis=1)
df['fav_is_home'] = df.odds_home <= df.odds_away
df['p_fav'] = np.where(df.fav_is_home, df.p_home, df.p_away)
df['p_dog'] = np.where(df.fav_is_home, df.p_away, df.p_home)
df['fav_win'] = np.where(df.fav_is_home, df.score_a > df.score_b, df.score_b > df.score_a).astype(int)
df['dog_win'] = np.where(df.fav_is_home, df.score_b > df.score_a, df.score_a > df.score_b).astype(int)
df['fav_odds_val'] = np.where(df.fav_is_home, df.odds_home, df.odds_away)
df['dog_odds_val'] = np.where(df.fav_is_home, df.odds_away, df.odds_home)

bins = [1.0, 1.10, 1.20, 1.30, 1.45, 1.60, 2.0, 3.5]
df['fb'] = pd.cut(df.fav_odds, bins, include_lowest=True)
rows = []
for b, g in df.groupby('fb', observed=True):
    n = len(g)
    rows.append(dict(bin=str(b), n=n,
                     fav_real=g.fav_win.mean(), fav_mkt=g.p_fav.mean(),
                     p_fav_binom=stats.binomtest(int(g.fav_win.sum()), n, g.p_fav.mean()).pvalue,
                     draw_real=g.is_draw.mean(), draw_mkt=g.p_draw.mean(),
                     p_draw_binom=stats.binomtest(int(g.is_draw.sum()), n, g.p_draw.mean()).pvalue,
                     dog_real=g.dog_win.mean(), dog_mkt=g.p_dog.mean(),
                     p_dog_binom=stats.binomtest(int(g.dog_win.sum()), n, g.p_dog.mean()).pvalue,
                     roi_fav=(g.fav_win*g.fav_odds_val-1).mean(),
                     roi_draw=(g.is_draw*g.odds_draw-1).mean(),
                     roi_dog=(g.dog_win*g.dog_odds_val-1).mean()))
print(pd.DataFrame(rows).round(4).to_string(index=False))

# DC '12' odds
def dc12(em):
    dc = em.get('Double Chance') or {}
    return dc.get('12')
df['dc12_odds'] = df.em.apply(dc12)
sub = df[(df.fav_odds <= 1.20) & df.dc12_odds.notna() & (df.dc12_odds > 1)]
if len(sub):
    hit = 1 - sub.is_draw
    print(f"\nDC '12' fav<=1.20 : n={len(sub)} cote moy={sub.dc12_odds.mean():.3f} reel={hit.mean():.4f} "
          f"1/cote={ (1/sub.dc12_odds).mean():.4f} ROI={ (hit*sub.dc12_odds-1).mean():+.4f}")

print("\n================ D) WALK-FORWARD ================")
cut = int(N*0.7)
def wf(name, d, mask_fn, odds_col, hit_col):
    tr = d.iloc[:cut]; te = d.iloc[cut:]
    tr = tr[mask_fn(tr)]; te = te[mask_fn(te)]
    if len(tr) == 0 or len(te) == 0:
        print(f"{name}: vide"); return
    roi_tr = (tr[hit_col]*tr[odds_col]-1).mean()
    roi_te = (te[hit_col]*te[odds_col]-1).mean()
    print(f"{name}: train n={len(tr)} ROI={roi_tr:+.4f} | OOS n={len(te)} ROI={roi_te:+.4f} "
          f"WR={te[hit_col].mean():.4f} cote_moy={te[odds_col].mean():.3f}")

df['hit_no_draw'] = (1 - df.is_draw).astype(float)
df['hit_fav'] = df.fav_win.astype(float)
df['hit_dog'] = df.dog_win.astype(float)
wf("DC12 fav<=1.20", df, lambda d: (d.fav_odds <= 1.20) & (d.dc12_odds > 1), 'dc12_odds', 'hit_no_draw')
wf("DC12 fav<=1.15", df, lambda d: (d.fav_odds <= 1.15) & (d.dc12_odds > 1), 'dc12_odds', 'hit_no_draw')
wf("DC12 fav<=1.30", df, lambda d: (d.fav_odds <= 1.30) & (d.dc12_odds > 1), 'dc12_odds', 'hit_no_draw')
wf("backFAV fav<=1.20", df, lambda d: d.fav_odds <= 1.20, 'fav_odds_val', 'hit_fav')
wf("backFAV fav<=1.15", df, lambda d: d.fav_odds <= 1.15, 'fav_odds_val', 'hit_fav')
wf("backDOG fav<=1.20", df, lambda d: d.fav_odds <= 1.20, 'dog_odds_val', 'hit_dog')
wf("backDOG fav 1.2-1.3", df, lambda d: (d.fav_odds > 1.2) & (d.fav_odds <= 1.3), 'dog_odds_val', 'hit_dog')
# DC12 sur toutes cotes (le no-draw est-il sous-price partout ?)
wf("DC12 tous", df, lambda d: d.dc12_odds > 1, 'dc12_odds', 'hit_no_draw')
print("\ndone")
