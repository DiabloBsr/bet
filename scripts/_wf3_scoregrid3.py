# -*- coding: utf-8 -*-
"""
WF3 — GRILLE DES SCORES, ITERATION 3 : DECOMPOSITION TOTAL x SPLIT + INFERENCE PROPRE
Constats itération 2 :
  - latente DC reproduit le 1X2 (MAE 0.003) mais le moteur n'est PAS DC (résidus systématiques massifs)
  - résidus structurés par TOTAL (a+b) et par équilibre -> hypothèse : moteur = loi du TOTAL x SPLIT conditionnel
  - W2 (EV latente > thr sur cotes CS) ROI OOS +7.8% SANS test de signif -> à refaire proprement

Sections :
  H1. Modèle hybride  P(a,b) = TotalDevig(t) * split_grille(a|t)  -> test vs réalité
  H2. La loi du TOTAL du moteur : truncated-Poisson ? (résidus par t) + dispersion
  H3. La loi du SPLIT : binomiale(t, w) ? overdispersion par t, asymétrie home/away
  H4. Le split publié EST-il le split réel ? chi2 (t,a) FDR — rows non cappées
  H5. La marge cellule par cellule sous le modèle hybride (= règle de pricing)
  H6. WALK-FORWARD propre : W2-redo (bootstrap par match), variantes hybride + anomalies split
"""
import sys, json, warnings, math
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 220); pd.set_option('display.max_columns', 60)
rng = np.random.default_rng(42)
eng = create_engine(load_settings().db_url)

q = """
SELECT e.id AS event_id, e.round_info, e.team_a, e.team_b, e.expected_start,
       r.score_a, r.score_b,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots os WHERE os.event_id = e.id)
WHERE e.round_info != '0' AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
"""
df = pd.read_sql(text(q), eng)
df = df.sort_values('event_id').drop_duplicates(subset=['team_a','team_b','expected_start'], keep='first')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)

CELLS = [(a,b) for a in range(7) for b in range(7) if a+b <= 6]
N_CELLS = len(CELLS); LBL = [f"{a}-{b}" for a,b in CELLS]
cell_idx = {c:j for j,c in enumerate(LBL)}
A = np.array([a for a,b in CELLS]); B = np.array([b for a,b in CELLS]); T = A + B
LOGFACT = np.array([math.lgamma(a+1)+math.lgamma(b+1) for a,b in CELLS])
ROWS = [[j for j in range(N_CELLS) if T[j]==t] for t in range(7)]

ems = df['extra_markets'].map(lambda s: json.loads(s) if isinstance(s,str) else (s or {}))
n0 = len(df)
odds_grid = np.full((n0, N_CELLS), np.nan)
for i, em in enumerate(ems):
    cs = em.get('Score exact')
    if not cs: continue
    for j,c in enumerate(LBL):
        v = cs.get(c)
        if v and v > 1.0: odds_grid[i,j] = v
mask = ~np.isnan(odds_grid).any(axis=1)
df = df[mask].reset_index(drop=True); odds_grid = odds_grid[mask]; ems = ems[mask].reset_index(drop=True)
n = len(df)
imp = 1.0/odds_grid; capped = odds_grid >= 99.99
print(f"[0] N={n}")

res_key = df['score_a'].astype(int).astype(str)+'-'+df['score_b'].astype(int).astype(str)
y = res_key.map(cell_idx).values.astype(int)
Y = np.zeros((n, N_CELLS)); Y[np.arange(n), y] = 1.0
tot_real = (df['score_a']+df['score_b']).values.astype(int)
a_real = df['score_a'].values.astype(int)
out3 = np.where(df['score_a']>df['score_b'],0,np.where(df['score_a']==df['score_b'],1,2)).astype(int)
imp3 = np.column_stack([1/df['odds_home'],1/df['odds_draw'],1/df['odds_away']])
p1x2 = imp3/imp3.sum(axis=1)[:,None]

tot_odds = np.full((n,7), np.nan)
for i, em in enumerate(ems):
    tm = em.get('Total de buts')
    if not tm: continue
    for t_ in range(7):
        v = tm.get(str(t_))
        if v and v > 1.0: tot_odds[i,t_] = v
assert (~np.isnan(tot_odds).any(axis=1)).all()
timp = 1/tot_odds
t_devig = timp/timp.sum(axis=1)[:,None]

def bh_fdr(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    rk = p[o]*m/(np.arange(m)+1); rk = np.minimum.accumulate(rk[::-1])[::-1]
    out = np.empty(m); out[o] = np.clip(rk,0,1); return out

def grid_test(P, label, sub=None):
    m = np.ones(n,bool) if sub is None else sub
    nn = m.sum(); rows = []
    for j,c in enumerate(LBL):
        p = P[m,j]; k = Y[m,j].sum(); E = p.sum(); V = (p*(1-p)).sum()
        z = (k-E)/np.sqrt(V) if V>0 else 0.0
        rows.append(dict(cell=c, obs=int(k), exp=E, z=z, p=2*stats.norm.sf(abs(z))))
    t = pd.DataFrame(rows); t['q_fdr'] = bh_fdr(t['p'].values)
    chi2 = ((t['obs']-t['exp'])**2/np.maximum(t['exp'],1e-9)).sum()
    ll = -np.log(np.clip(P[m][np.arange(nn), y[m]],1e-12,1)).mean()
    print(f"\n=== [{label}] n={nn} chi2={chi2:.1f} (df=27) p={stats.chi2.sf(chi2,27):.3e} | log-loss CS={ll:.4f} ===")
    sig = t[t['q_fdr']<0.05].sort_values('q_fdr')
    print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(sig) else "  aucune cellule significative (FDR 5%)")
    return t, chi2, ll

# ----------------------------------------------------------------------------
# H1. HYBRIDE : TotalDevig x split grille
# ----------------------------------------------------------------------------
row_sum = np.stack([imp[:, js].sum(axis=1) for js in ROWS], axis=1)   # (n,7)
split = imp / row_sum[:, T]                                           # P(cell | total) publié
hyb = t_devig[:, T] * split
print("="*100)
t_h, chi_h, ll_h = grid_test(hyb, "HYBRIDE TotalDevig x split publié")

# variante : power-devig du Total
k_pow = np.ones(n)
for i in range(n):
    f = lambda k: (timp[i]**k).sum()-1
    lo, hi = 1.0, 3.0
    for _ in range(60):
        mid = (lo+hi)/2
        if f(mid) > 0: lo = mid
        else: hi = mid
    k_pow[i] = (lo+hi)/2
t_pow = timp**k_pow[:,None]; t_pow = t_pow/t_pow.sum(axis=1)[:,None]
hyb_pow = t_pow[:, T]*split
t_hp, chi_hp, ll_hp = grid_test(hyb_pow, "HYBRIDE TotalPowerDevig x split publié")
print(f"\n[H1] k power devig Total: mean={k_pow.mean():.4f} sd={k_pow.std():.4f}")

# ----------------------------------------------------------------------------
# H2. LOI DU TOTAL
# ----------------------------------------------------------------------------
print("\n"+"="*100)
print("[H2] LOI DU TOTAL — le Total devig est-il une Poisson tronquée(<=6) renormalisée ?")
lam_grid = np.arange(0.3, 8.0, 0.002)
pois = np.exp(-lam_grid[:,None] + np.arange(7)[None,:]*np.log(lam_grid[:,None]) -
              np.array([math.lgamma(t+1) for t in range(7)])[None,:])
pois = pois/pois.sum(axis=1)[:,None]
mean_tp = (pois*np.arange(7)).sum(axis=1)
mu_mkt = (t_devig*np.arange(7)).sum(axis=1)
lam_hat = np.interp(mu_mkt, mean_tp, lam_grid)
idx = np.searchsorted(lam_grid, lam_hat).clip(0, len(lam_grid)-1)
tp_fit = pois[idx]
res_t = t_devig - tp_fit
print(f"  lambda_hat: mean={lam_hat.mean():.4f} sd={lam_hat.std():.4f} p5={np.percentile(lam_hat,5):.3f} p95={np.percentile(lam_hat,95):.3f}")
print("  résidu (TotalDevig - truncPoisson) par t :")
for t_ in range(7):
    v = res_t[:,t_]
    print(f"   t={t_}: mean={v.mean():+.5f} sd={v.std():.5f} t-stat={v.mean()/(v.std()/np.sqrt(n)):+.1f}")
var_mkt = (t_devig*np.arange(7)**2).sum(axis=1) - mu_mkt**2
var_tp = (tp_fit*np.arange(7)**2).sum(axis=1) - (tp_fit*np.arange(7)).sum(axis=1)**2
print(f"  dispersion: var(Total devig)/var(truncPois même moyenne) mean={(var_mkt/var_tp).mean():.4f} sd={(var_mkt/var_tp).std():.4f}")
# et la réalité vs Total devig (calibration du marché Total)
print("  calibration marché Total vs réalité (z par t):")
for t_ in range(7):
    p = t_devig[:,t_]; k = (tot_real==t_).sum(); E = p.sum(); V = (p*(1-p)).sum()
    print(f"   t={t_}: obs={k} exp={E:.1f} z={(k-E)/np.sqrt(V):+.2f}")
chi_tot = sum((((tot_real==t_).sum()-t_devig[:,t_].sum())**2/t_devig[:,t_].sum()) for t_ in range(7))
print(f"  chi2 Total devig vs réalité = {chi_tot:.1f} (df=6) p={stats.chi2.sf(chi_tot,6):.3e}")
chi_tot2 = sum((((tot_real==t_).sum()-t_pow[:,t_].sum())**2/t_pow[:,t_].sum()) for t_ in range(7))
print(f"  chi2 Total POWER-devig vs réalité = {chi_tot2:.1f} (df=6) p={stats.chi2.sf(chi_tot2,6):.3e}")

# ----------------------------------------------------------------------------
# H3. LOI DU SPLIT — binomiale ?
# ----------------------------------------------------------------------------
print("\n"+"="*100)
print("[H3] LOI DU SPLIT publié vs Binomiale(t, w)")
# w estimé par match : moyenne ponderée de E[a|t]/t sur rows 2..4 (peu cappées)
w_est = np.zeros(n)
for i in range(n):
    num = den = 0.0
    for t_ in (2,3,4):
        js = ROWS[t_]
        if capped[i, js].any(): continue
        s = imp[i, js]; s = s/s.sum()
        num += (s*A[js]).sum(); den += t_
    w_est[i] = num/den if den>0 else 0.5
print(f"  w (part home du split): mean={w_est.mean():.4f} sd={w_est.std():.4f}")
print(f"  corr(w, pH devig 1X2) = {np.corrcoef(w_est, p1x2[:,0])[0,1]:.4f}")
# log-ratio split publié vs binom(t,w) par (t,a) — matchs équilibrés w in [0.45,0.6]
mbal = (w_est>0.45)&(w_est<0.60)
print(f"  table log[split_publié / binom] — matchs 0.45<w<0.60 (n={mbal.sum()}), rows sans cap:")
for t_ in range(1,7):
    js = ROWS[t_]
    okr = mbal & ~capped[:, js].any(axis=1)
    if okr.sum() < 100: print(f"   t={t_}: n<100, skip"); continue
    s = imp[okr][:, js]; s = s/s.sum(axis=1)[:,None]
    w = w_est[okr]
    outs = []
    for kk, j in enumerate(js):
        a_ = A[j]
        binp = math.comb(t_, int(a_)) * w**a_ * (1-w)**(t_-a_)
        lr = np.log(s[:,kk]/binp)
        outs.append(f"a={a_}:{lr.mean():+.3f}")
    # overdispersion du split
    Ea = (s*A[js]).sum(axis=1); Va = (s*A[js]**2).sum(axis=1)-Ea**2
    phi = Va/(t_*w*(1-w))
    print(f"   t={t_}: phi={phi.mean():.3f}±{phi.std():.3f} | " + " ".join(outs))

# ----------------------------------------------------------------------------
# H4. SPLIT PUBLIE vs SPLIT REEL (le coeur : la grille est-elle le générateur ?)
# ----------------------------------------------------------------------------
print("\n"+"="*100)
print("[H4] SPLIT PUBLIE vs REALITE — P(a | total=t) ; rows non cappées uniquement")
rows4 = []
for t_ in range(1,7):
    js = ROWS[t_]
    okr = ~capped[:, js].any(axis=1) & (tot_real == t_)
    nn = okr.sum()
    if nn < 50: continue
    s = imp[okr][:, js]; s = s/s.sum(axis=1)[:,None]
    for kk, j in enumerate(js):
        a_ = A[j]
        obs = (a_real[okr]==a_).sum()
        p = s[:,kk]; E = p.sum(); V = (p*(1-p)).sum()
        z = (obs-E)/np.sqrt(V) if V>0 else 0
        rows4.append(dict(t=t_, a=int(a_), n=nn, obs=int(obs), exp=E, z=z, p=2*stats.norm.sf(abs(z))))
t4 = pd.DataFrame(rows4); t4['q_fdr'] = bh_fdr(t4['p'].values)
chi4 = (t4['z']**2).sum(); df4 = len(t4)-6
print(t4.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"  chi2 somme z² = {chi4:.1f} (~df={df4}) p={stats.chi2.sf(chi4, df4):.3e}")
sig4 = t4[t4['q_fdr']<0.05]
print(f"  cellules (t,a) significatives FDR5%: {len(sig4)}")

# ----------------------------------------------------------------------------
# H5. MARGE PAR CELLULE SOUS HYBRIDE
# ----------------------------------------------------------------------------
print("\n"+"="*100)
print("[H5] MARGE = imp / hybride (uncapped) — règle de pricing")
m_cell = np.where(~capped, imp/np.maximum(hyb,1e-12), np.nan)
mt = pd.DataFrame(dict(cell=LBL, total=T, marge=np.nanmean(m_cell,axis=0),
                       sd=np.nanstd(m_cell,axis=0), n=(~capped).sum(axis=0)))
print(mt.sort_values(['total','cell']).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
# la marge depend-elle du total uniquement ? (le Total porte 1.12, le split ~neutre ?)
print("  overround par row du CS (somme imp row / TotalDevig row), rows non cappées:")
for t_ in range(7):
    js = ROWS[t_]
    okr = ~capped[:, js].any(axis=1)
    if okr.sum() < 100: continue
    ratio = row_sum[okr, t_]/t_devig[okr, t_]
    print(f"   t={t_}: mean={ratio.mean():.4f} sd={ratio.std():.4f} n={okr.sum()}")

# ----------------------------------------------------------------------------
# H6. WALK-FORWARD PROPRE
# ----------------------------------------------------------------------------
print("\n"+"="*100)
cut = int(n*0.7); tr = np.arange(n)<cut; te = ~tr
print(f"[H6] WALK-FORWARD train={tr.sum()} OOS={te.sum()}")

# reconstruire latente DC (comme itération 2) pour W2-redo
DC_js = [cell_idx['0-0'],cell_idx['1-0'],cell_idx['0-1'],cell_idx['1-1']]
latent = np.zeros((n, N_CELLS))
for i in range(n):
    ok = ~capped[i]; fitm = ok.copy()
    for j in DC_js: fitm[j] = False
    js = np.where(fitm)[0]
    Xd = np.column_stack([np.ones(len(js)), A[js], B[js]])
    yv = np.log(imp[i,js]) + LOGFACT[js]; w = imp[i,js]; W = np.sqrt(w)
    coef,*_ = np.linalg.lstsq(Xd*W[:,None], yv*W, rcond=None)
    b0, llh, lla = coef; lh_, la_ = np.exp(llh), np.exp(lla)
    pred = np.exp(b0 + A*llh + B*lla - LOGFACT)
    s = np.zeros(N_CELLS)
    s[cell_idx['0-0']] = -lh_*la_; s[cell_idx['1-1']] = -1.0
    s[cell_idx['1-0']] = la_; s[cell_idx['0-1']] = lh_
    jd = [j for j in DC_js if ok[j]]
    rho_ = 0.0
    if jd:
        r_ = imp[i,jd]/pred[jd]; wj = imp[i,jd]; sj = s[jd]
        den = (wj*sj*sj).sum()
        rho_ = (wj*sj*(r_-1)).sum()/den if den>0 else 0.0
    tau = np.maximum(1+s*rho_, 1e-6)
    ql = pred*tau; latent[i] = ql/ql.sum()

def eval_rule(pick, name):
    """pick: (n,N_CELLS) bool. P&L par match + bootstrap par match."""
    for lab, m in [('train',tr),('OOS',te)]:
        pk = pick & m[:,None]
        nb = int(pk.sum())
        if nb == 0: print(f"  {name} [{lab}]: 0 pari"); continue
        pnl_m = ((Y*odds_grid - 1)*pk).sum(axis=1)   # net par match
        stake_m = pk.sum(axis=1)
        roi = pnl_m.sum()/stake_m.sum()
        # bootstrap par match
        mi = np.where(stake_m>0)[0]
        boots = np.empty(4000)
        for bidx in range(4000):
            sel = rng.choice(mi, len(mi), replace=True)
            boots[bidx] = pnl_m[sel].sum()/stake_m[sel].sum()
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pos = (boots<=0).mean()
        print(f"  {name} [{lab}]: n={nb} ROI={roi:+.4f} CI95=[{lo:+.4f},{hi:+.4f}] P(ROI<=0)={pos:.3f} "
              f"cote_moy={odds_grid[pk].mean():.2f}")

print("\n[W2-redo] EV latente DC > thr sur cotes CS (uncapped):")
ev = latent*odds_grid - 1
for thr in [0.03, 0.05]:
    eval_rule((ev>thr)&~capped, f"thr={thr}")

print("\n[W5] EV hybride > thr sur cotes CS (uncapped):")
evh = hyb*odds_grid - 1
for thr in [0.0, 0.03]:
    eval_rule((evh>thr)&~capped, f"thr={thr}")

print("\n[W6] anomalies split détectées sur TRAIN (q_fdr<10%) -> bet OOS:")
rows6 = []
for t_ in range(1,7):
    js = ROWS[t_]
    okr = ~capped[:, js].any(axis=1) & (tot_real==t_) & tr
    if okr.sum() < 50: continue
    s = imp[okr][:, js]; s = s/s.sum(axis=1)[:,None]
    for kk,j in enumerate(js):
        obs = (a_real[okr]==A[j]).sum(); p = s[:,kk]; E = p.sum(); V = (p*(1-p)).sum()
        z = (obs-E)/np.sqrt(V) if V>0 else 0
        rows6.append(dict(t=t_, j=j, z=z, p=2*stats.norm.sf(abs(z))))
t6 = pd.DataFrame(rows6); t6['q'] = bh_fdr(t6['p'].values)
sel6 = t6[(t6['q']<0.10)&(t6['z']>0)]
if len(sel6):
    pick = np.zeros((n,N_CELLS), bool)
    for _,r in sel6.iterrows():
        pick[:, int(r['j'])] = True
    pick &= ~capped
    print(f"  cellules train-significatives positives: {[LBL[int(j)] for j in sel6['j']]}")
    eval_rule(pick, "split-anom")
else:
    print("  aucune anomalie split positive sur train")

print("\nDONE")
