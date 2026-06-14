# -*- coding: utf-8 -*-
"""
WF3 — GRILLE DES SCORES, ITERATION 4 : VERIFICATIONS FINALES DU MODELE MOTEUR
Modèle candidat (issu it.3) :
  P_gen(a,b) = q_T(t) * s(a|t)
  cotes Total(t) = 1/(1.12 * q_T(t)) ; cotes CS(a,b) = min(100, 1/(1.12 * q_T(t) * s(a|t)))
V1: identité interne exacte  somme_row(1/CS) == 1/odds_Total(t)  (par match, par row)
V2: 1X2 et +/-3.5 dérivent-ils de la même grille ? (hybride -> 1X2 vs 1X2 publié, marge 1.06)
V3: cotes "fair" des cellules cappées (via split réel t=5/6) + ROI théorique du cap
V4: famille de la loi du TOTAL : truncPoisson vs Binomiale(N,p) tronquée (N=6..30) vs réalité
V5: G/NG et Mi-tps dérivent-ils aussi ? (G/NG vs grille — rapide)
"""
import sys, json, warnings, math
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings
pd.set_option('display.width', 220)
eng = create_engine(load_settings().db_url)

q = """
SELECT e.id AS event_id, e.team_a, e.team_b, e.expected_start,
       r.score_a, r.score_b, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots os WHERE os.event_id = e.id)
WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
df = pd.read_sql(text(q), eng)
df = df.sort_values('event_id').drop_duplicates(subset=['team_a','team_b','expected_start'], keep='first')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)
CELLS = [(a,b) for a in range(7) for b in range(7) if a+b<=6]
N_CELLS = len(CELLS); LBL = [f"{a}-{b}" for a,b in CELLS]
A = np.array([a for a,b in CELLS]); B = np.array([b for a,b in CELLS]); T = A+B
ROWS = [[j for j in range(N_CELLS) if T[j]==t] for t in range(7)]
ems = df['extra_markets'].map(lambda s: json.loads(s) if isinstance(s,str) else (s or {}))
odds_grid = np.full((len(df), N_CELLS), np.nan)
for i, em in enumerate(ems):
    cs = em.get('Score exact')
    if not cs: continue
    for j,c in enumerate(LBL):
        v = cs.get(c)
        if v and v>1.0: odds_grid[i,j]=v
mask = ~np.isnan(odds_grid).any(axis=1)
df = df[mask].reset_index(drop=True); odds_grid = odds_grid[mask]; ems = ems[mask].reset_index(drop=True)
n = len(df); imp = 1/odds_grid; capped = odds_grid>=99.99
tot_real = (df['score_a']+df['score_b']).values.astype(int)
a_real = df['score_a'].values.astype(int)
tot_odds = np.full((n,7), np.nan)
for i, em in enumerate(ems):
    tm = em.get('Total de buts')
    for t_ in range(7):
        v = tm.get(str(t_))
        if v and v>1.0: tot_odds[i,t_]=v
timp = 1/tot_odds
t_devig = timp/timp.sum(axis=1)[:,None]
print(f"[0] N={n}")

# ---------------------------------------------------------------- V1
print("\n[V1] IDENTITE  sum_row(1/CS) / (1/odds_Total) — rows 100% non cappées")
row_sum = np.stack([imp[:, js].sum(axis=1) for js in ROWS], axis=1)
for t_ in range(7):
    okr = ~capped[:, ROWS[t_]].any(axis=1)
    if okr.sum()<50: print(f"  t={t_}: trop peu de rows non cappées"); continue
    ratio = row_sum[okr,t_]/timp[okr,t_]
    print(f"  t={t_}: n={okr.sum()} mean={ratio.mean():.5f} sd={ratio.std():.5f} max|1-r|={np.abs(1-ratio).max():.5f}")
ovr_tot = timp.sum(axis=1)
print(f"  overround Total: mean={ovr_tot.mean():.5f} sd={ovr_tot.std():.5f}")
ovr3 = (1/df[['odds_home','odds_draw','odds_away']].values).sum(axis=1)
print(f"  overround 1X2:   mean={ovr3.mean():.5f} sd={ovr3.std():.5f}")

# ---------------------------------------------------------------- V2
print("\n[V2] LE 1X2 ET +/- DERIVENT-ILS DE LA MEME GRILLE LATENTE ?")
split = imp/row_sum[:, T]
hyb = t_devig[:, T]*split   # grille latente (distordue par cap sur rows 5-6)
h_j = [j for j in range(N_CELLS) if A[j]>B[j]]; d_j = [j for j in range(N_CELLS) if A[j]==B[j]]
a_j = [j for j in range(N_CELLS) if A[j]<B[j]]
g3 = np.column_stack([hyb[:,h_j].sum(1), hyb[:,d_j].sum(1), hyb[:,a_j].sum(1)])
imp3 = 1/df[['odds_home','odds_draw','odds_away']].values
p3 = imp3/imp3.sum(axis=1)[:,None]
ncap = capped.sum(axis=1)
nocap = ncap <= np.percentile(ncap, 10)
print(f"  matchs peu cappés (ncap<=p10={np.percentile(ncap,10):.0f}): n={nocap.sum()} | ncap mean={ncap.mean():.1f}")
for k,lab in enumerate(['H','D','A']):
    d_all = g3[:,k]-p3[:,k]; d_nc = g3[nocap,k]-p3[nocap,k]
    print(f"  {lab}: MAE all={np.abs(d_all).mean():.5f} | MAE peu-cappé={np.abs(d_nc).mean():.5f} max|d| peu-cappé={np.abs(d_nc).max():.5f}")
# +/- 3.5
p_over_grid = t_devig[:,4:7].sum(axis=1)
rows = []
for i, em in enumerate(ems):
    ou = em.get('+/-')
    ks = list(ou.keys()) if ou else []
    ov = [k for k in ks if k.startswith('>')]; un = [k for k in ks if k.startswith('<')]
    if len(ov)==1 and len(un)==1:
        io, iu = 1/ou[ov[0]], 1/ou[un[0]]
        rows.append((i, io/(io+iu), io+iu))
ou_df = pd.DataFrame(rows, columns=['i','p_over_mkt','ovr'])
po_g = p_over_grid[ou_df['i'].values]
print(f"  +/-3.5: MAE(P_over grilleTotal vs marché +/-)={np.abs(po_g-ou_df['p_over_mkt']).mean():.5f} "
      f"max={np.abs(po_g-ou_df['p_over_mkt']).max():.5f} | overround +/- mean={ou_df['ovr'].mean():.5f}")

# ---------------------------------------------------------------- V3
print("\n[V3] CELLULES CAPPEES : fair odds via split REEL des rows 5/6")
for t_ in (5,6):
    js = ROWS[t_]
    mt = tot_real==t_
    cnt = np.array([ (a_real[mt]==A[j]).sum() for j in js ], float)
    sp_real = cnt/cnt.sum()
    qt = t_devig[:,t_].mean()
    print(f"  t={t_} (n={mt.sum()}): split réel = " + " ".join(f"{LBL[j]}:{sp_real[k]:.3f}" for k,j in enumerate(js)))
    fair = 1/(qt*sp_real)
    print(f"        q_T moy={qt:.4f} -> fair odds moyennes: " + " ".join(f"{LBL[j]}:{fair[k]:.0f}" for k,j in enumerate(js)))
kc = ( (np.eye(N_CELLS)[ (df['score_a'].astype(int).astype(str)+'-'+df['score_b'].astype(int).astype(str)).map({c:j for j,c in enumerate(LBL)}).values ] * capped).sum() )
print(f"  cellules cappées: obs={int(kc)}/{capped.sum()} -> freq={kc/capped.sum():.5f} ; cote 100 -> ROI={100*kc/capped.sum()-1:.3f}")

# ---------------------------------------------------------------- V4
print("\n[V4] FAMILLE DE LA LOI DU TOTAL (fit par matching de moyenne, KL moyen vs Total devig)")
tt = np.arange(7)
def kl_for_family(pmf_grid, mean_grid, name):
    mu = (t_devig*tt).sum(axis=1)
    lam = np.interp(mu, mean_grid, np.arange(len(mean_grid), dtype=float))
    idx = np.clip(np.round(lam).astype(int), 0, len(mean_grid)-1)
    fit = pmf_grid[idx]
    kl = (t_devig*np.log(np.clip(t_devig,1e-12,1)/np.clip(fit,1e-12,1))).sum(axis=1)
    res = t_devig-fit
    print(f"  {name}: KL moyen={kl.mean():.5f} | résidu max |mean| par t = "
          + " ".join(f"t{t_}:{res[:,t_].mean():+.4f}" for t_ in range(7)))
    return kl.mean()
# truncated Poisson
lamg = np.arange(0.3, 8.0, 0.002)
pois = np.exp(-lamg[:,None]+tt[None,:]*np.log(lamg[:,None])-np.array([math.lgamma(t+1) for t in tt])[None,:])
pois = pois/pois.sum(axis=1)[:,None]
kl_for_family(pois, (pois*tt).sum(axis=1), "truncPoisson")
# truncated Binomial(N,p)
best = None
for N in range(6, 31):
    pg = np.arange(0.005, 0.995, 0.001)
    pmf = np.array([math.comb(N,int(t_)) for t_ in tt])[None,:] * pg[:,None]**tt[None,:] * (1-pg[:,None])**(N-tt[None,:])
    pmf = pmf/pmf.sum(axis=1)[:,None]
    klv = kl_for_family(pmf, (pmf*tt).sum(axis=1), f"truncBinom N={N}") if N in (8,10,12,14,16,20) else None
    mu = (t_devig*tt).sum(axis=1)
    mg = (pmf*tt).sum(axis=1)
    lam = np.interp(mu, mg, np.arange(len(mg), dtype=float))
    idx = np.clip(np.round(lam).astype(int), 0, len(mg)-1)
    fit = pmf[idx]
    kl = (t_devig*np.log(np.clip(t_devig,1e-12,1)/np.clip(fit,1e-12,1))).sum(axis=1).mean()
    if best is None or kl < best[1]: best = (N, kl)
print(f"  >>> meilleur N binomial = {best[0]} (KL={best[1]:.5f})")
# Conway-Maxwell-Poisson tronquée (2 param -> fit nu global par grid, lambda par match)
def cmp_pmf(lam, nu):
    lw = tt*np.log(lam) - nu*np.array([math.lgamma(t+1) for t in tt])
    w = np.exp(lw-lw.max()); return w/w.sum()
print("  CMP tronquée: scan nu...")
best_c = None
for nu in np.arange(0.8, 2.2, 0.05):
    lamg2 = np.exp(np.arange(-1.5, 3.5, 0.01))
    pmfs = np.stack([cmp_pmf(l, nu) for l in lamg2])
    mg = (pmfs*tt).sum(axis=1)
    o = np.argsort(mg)
    mu = (t_devig*tt).sum(axis=1)
    lam = np.interp(mu, mg[o], np.arange(len(mg), dtype=float))
    idx = np.clip(np.round(lam).astype(int), 0, len(mg)-1)
    fit = pmfs[o][idx]
    kl = (t_devig*np.log(np.clip(t_devig,1e-12,1)/np.clip(fit,1e-12,1))).sum(axis=1).mean()
    if best_c is None or kl < best_c[1]: best_c = (nu, kl)
print(f"  >>> meilleure CMP: nu={best_c[0]:.2f} (KL={best_c[1]:.5f})")

# ---------------------------------------------------------------- V5
print("\n[V5] G/NG derive-t-il de la grille ?")
gg_j = [j for j in range(N_CELLS) if A[j]>0 and B[j]>0]
p_gg_grid = hyb[:, gg_j].sum(axis=1)
rows = []
for i, em in enumerate(ems):
    g = em.get('G/NG')
    if not g: continue
    kk = {k.lower(): v for k,v in g.items()}
    oui = next((v for k,v in kk.items() if 'oui' in k or 'yes' in k or k=='gg'), None)
    non = next((v for k,v in kk.items() if 'non' in k or 'no' in k or k=='ng'), None)
    if oui and non:
        io, inn = 1/oui, 1/non
        rows.append((i, io/(io+inn), io+inn))
if rows:
    gdf = pd.DataFrame(rows, columns=['i','p_gg_mkt','ovr'])
    pg_g = p_gg_grid[gdf['i'].values]
    real_gg = ((df['score_a']>0)&(df['score_b']>0)).values[gdf['i'].values].astype(float)
    print(f"  n={len(gdf)} overround G/NG={gdf['ovr'].mean():.5f}")
    print(f"  MAE(grille vs marché)={np.abs(pg_g-gdf['p_gg_mkt']).mean():.5f} (matchs sans cap: "
          f"{np.abs((pg_g-gdf['p_gg_mkt'])[nocap[gdf['i'].values]]).mean():.5f})")
    print(f"  réel P(GG)={real_gg.mean():.4f} marché={gdf['p_gg_mkt'].mean():.4f} grille={pg_g.mean():.4f}")
else:
    print("  G/NG introuvable")
print("\nDONE")
