# -*- coding: utf-8 -*-
"""
WF3 — GRILLE DES SCORES, ITERATION 2 : RECONSTRUCTION DE LA GRILLE LATENTE
Hypothèse issue de l'itération 1 :
  le générateur = grille latente (famille Dixon-Coles tronquée à total<=6),
  les cotes Score exact = 1/(q_j) * marge, PLAFONNÉES à 100.0 (32% des cellules),
  les marchés dérivés (1X2, Total, +/-) sont pricés depuis la grille NON cappée.

Méthode : par match, WLS sur log(imp_brut) des cellules non cappées hors {0-0,1-0,0-1,1-1}
  log imp_ab = beta0 + a*log(lh) + b*log(la) - log(a!b!)   (Z de troncature + marge absorbés dans beta0)
  puis rho Dixon-Coles estimé sur les 4 cellules basses, grille latente = DC normalisée.

Tests:
  T1 réalité vs latente (z par cellule + FDR + chi2) — vs chi2=164.6 du devig proportionnel
  T2 latente->1X2 vs marché 1X2 (qui matche la réalité ?)
  T3 latente->Total vs marché 'Total de buts'
  T4 masse prédite des cellules cappées vs observé
  T5 structure de la marge (constante ? favorite-longshot ? lien n_capped)
  T6 paramètres moteur: distributions lh, la, rho ; lien 1X2/Total ; détection de lattice
  T7 résidu systématique par cellule (le moteur est-il EXACTEMENT DC ?)
  T8 WALK-FORWARD 70/30 : toutes stratégies candidates
"""
import sys, json, warnings, math
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 60)

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
df = df.sort_values('event_id').drop_duplicates(
    subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)

CELLS = [(a, b) for a in range(7) for b in range(7) if a + b <= 6]
N_CELLS = len(CELLS)
LBL = [f"{a}-{b}" for a, b in CELLS]
cell_idx = {c: j for j, c in enumerate(LBL)}
A = np.array([a for a, b in CELLS], float)
B = np.array([b for a, b in CELLS], float)
LOGFACT = np.array([math.lgamma(a + 1) + math.lgamma(b + 1) for a, b in CELLS])
DC_SET = {cell_idx['0-0'], cell_idx['1-0'], cell_idx['0-1'], cell_idx['1-1']}
dc_js = sorted(DC_SET)

ems = df['extra_markets'].map(lambda s: json.loads(s) if isinstance(s, str) else (s or {}))
n0 = len(df)
odds_grid = np.full((n0, N_CELLS), np.nan)
for i, em in enumerate(ems):
    cs = em.get('Score exact')
    if not cs: continue
    for j, c in enumerate(LBL):
        v = cs.get(c)
        if v and v > 1.0:
            odds_grid[i, j] = v
mask = ~np.isnan(odds_grid).any(axis=1)
df = df[mask].reset_index(drop=True)
odds_grid = odds_grid[mask]
ems = ems[mask].reset_index(drop=True)
n = len(df)
imp = 1.0 / odds_grid
capped = odds_grid >= 99.99
overround = imp.sum(axis=1)
devig_prop = imp / overround[:, None]
print(f"[0] N={n} | overround mean={overround.mean():.4f} | cellules cappées {capped.mean()*100:.1f}%")

res_key = df['score_a'].astype(int).astype(str) + '-' + df['score_b'].astype(int).astype(str)
y = res_key.map(cell_idx).values.astype(int)
Y = np.zeros((n, N_CELLS)); Y[np.arange(n), y] = 1.0
out3 = np.where(df['score_a'] > df['score_b'], 0, np.where(df['score_a'] == df['score_b'], 1, 2)).astype(int)
tot_real = (df['score_a'] + df['score_b']).values.astype(int)

imp3 = np.column_stack([1/df['odds_home'], 1/df['odds_draw'], 1/df['odds_away']])
p1x2 = imp3 / imp3.sum(axis=1)[:, None]

# ----------------------------------------------------------------------------
# FIT PAR MATCH
# ----------------------------------------------------------------------------
lh = np.zeros(n); la = np.zeros(n); rho = np.zeros(n); beta0 = np.zeros(n)
r2 = np.zeros(n); nfit = np.zeros(n, int)
latent = np.zeros((n, N_CELLS))
for i in range(n):
    ok = ~capped[i]
    fitm = ok.copy()
    for j in dc_js: fitm[j] = False
    js = np.where(fitm)[0]
    Xd = np.column_stack([np.ones(len(js)), A[js], B[js]])
    yv = np.log(imp[i, js]) + LOGFACT[js]
    w = imp[i, js]
    W = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(Xd * W[:, None], yv * W, rcond=None)
    beta0[i], llh, lla = coef
    lh[i], la[i] = np.exp(llh), np.exp(lla)
    nfit[i] = len(js)
    fit = Xd @ coef
    ssr = (w * (yv - fit)**2).sum(); sst = (w * (yv - (w*yv).sum()/w.sum())**2).sum()
    r2[i] = 1 - ssr/sst if sst > 0 else np.nan
    # pred poisson (avec marge+troncature absorbées)
    logpred = beta0[i] + A*llh + B*lla - LOGFACT
    pred = np.exp(logpred)
    # rho DC sur cellules basses non cappées
    s = np.zeros(N_CELLS)
    s[cell_idx['0-0']] = -lh[i]*la[i]; s[cell_idx['1-1']] = -1.0
    s[cell_idx['1-0']] = la[i];        s[cell_idx['0-1']] = lh[i]
    jd = [j for j in dc_js if ok[j]]
    if jd:
        r_ = imp[i, jd]/pred[jd]
        wj = imp[i, jd]; sj = s[jd]
        denom = (wj*sj*sj).sum()
        rho[i] = (wj*sj*(r_-1)).sum()/denom if denom > 0 else 0.0
    tau = 1 + s*rho[i]
    tau[tau < 1e-6] = 1e-6
    qlat = pred * tau
    latent[i] = qlat/qlat.sum()

print(f"[FIT] R2 médian={np.median(r2):.5f} | R2<0.99: {(r2<0.99).mean()*100:.2f}% des matchs")
print(f"[FIT] lh: med={np.median(lh):.3f} [{np.percentile(lh,5):.3f},{np.percentile(lh,95):.3f}] | "
      f"la: med={np.median(la):.3f} [{np.percentile(la,5):.3f},{np.percentile(la,95):.3f}] | "
      f"rho: med={np.median(rho):.4f} mean={rho.mean():.4f} sd={rho.std():.4f}")

def bh_fdr(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p)
    rk = p[o]*m/(np.arange(m)+1); rk = np.minimum.accumulate(rk[::-1])[::-1]
    out = np.empty(m); out[o] = np.clip(rk, 0, 1); return out

def grid_test(P, label):
    rows = []
    for j, c in enumerate(LBL):
        p = P[:, j]; k = Y[:, j].sum(); E = p.sum(); V = (p*(1-p)).sum()
        z = (k-E)/np.sqrt(V) if V > 0 else 0.0
        rows.append(dict(cell=c, obs=int(k), exp=E, z=z, p=2*stats.norm.sf(abs(z))))
    t = pd.DataFrame(rows); t['q_fdr'] = bh_fdr(t['p'].values)
    chi2 = ((t['obs']-t['exp'])**2/np.maximum(t['exp'],1e-9)).sum()
    pg = stats.chi2.sf(chi2, N_CELLS-1)
    ll = -np.log(np.clip(P[np.arange(n), y], 1e-12, 1)).mean()
    print(f"\n=== T1[{label}] chi2={chi2:.1f} (df=27) p={pg:.3e} | log-loss CS={ll:.4f} ===")
    sig = t[t['q_fdr'] < 0.05].sort_values('q_fdr')
    print(sig.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(sig) else "  aucune cellule significative (FDR 5%)")
    return t, chi2, ll

print("\n" + "="*100)
t_lat, chi_lat, ll_lat = grid_test(latent, "GRILLE LATENTE DC")
t_prop, chi_prop, ll_prop = grid_test(devig_prop, "devig proportionnel (rappel)")

# ----------------------------------------------------------------------------
# T2 latente -> 1X2
# ----------------------------------------------------------------------------
print("\n" + "="*100)
h_j = [j for j,(a,b) in enumerate(CELLS) if a > b]
d_j = [j for j,(a,b) in enumerate(CELLS) if a == b]
a_j = [j for j,(a,b) in enumerate(CELLS) if a < b]
l1x2 = np.column_stack([latent[:, h_j].sum(1), latent[:, d_j].sum(1), latent[:, a_j].sum(1)])
print("[T2] latente->1X2 vs 1X2 devig:")
for k, lab in enumerate(['H','D','A']):
    print(f"  {lab}: mean diff={(l1x2[:,k]-p1x2[:,k]).mean():+.4f} MAE={np.abs(l1x2[:,k]-p1x2[:,k]).mean():.4f} corr={np.corrcoef(l1x2[:,k],p1x2[:,k])[0,1]:.5f}")
ll_l = -np.log(np.clip(l1x2[np.arange(n), out3], 1e-12, 1))
ll_m = -np.log(np.clip(p1x2[np.arange(n), out3], 1e-12, 1))
print(f"  log-loss 1X2 réalité: latente={ll_l.mean():.5f} vs marché={ll_m.mean():.5f} (wilcoxon p={stats.wilcoxon(ll_l, ll_m).pvalue:.3g})")

# ----------------------------------------------------------------------------
# T3 latente -> Total
# ----------------------------------------------------------------------------
tot_cells = [[j for j,(a,b) in enumerate(CELLS) if a+b == t_] for t_ in range(7)]
l_tot = np.column_stack([latent[:, js].sum(1) for js in tot_cells])
tot_odds = np.full((n, 7), np.nan)
for i, em in enumerate(ems):
    tm = em.get('Total de buts')
    if not tm: continue
    for t_ in range(7):
        v = tm.get(str(t_))
        if v and v > 1.0: tot_odds[i, t_] = v
ok_t = ~np.isnan(tot_odds).any(axis=1)
tm_dv = (1/tot_odds[ok_t]) / (1/tot_odds[ok_t]).sum(axis=1)[:, None]
lt = l_tot[ok_t]; yt = tot_real[ok_t]; ntt = ok_t.sum()
print(f"\n[T3] latente->Total vs marché Total (n={ntt}):")
for t_ in range(7):
    print(f"  T={t_}: réel={(yt==t_).mean():.4f} latente={lt[:,t_].mean():.4f} marché={tm_dv[:,t_].mean():.4f} diff={(lt[:,t_]-tm_dv[:,t_]).mean():+.4f}")
ll_lt = -np.log(np.clip(lt[np.arange(ntt), yt], 1e-12, 1))
ll_mt = -np.log(np.clip(tm_dv[np.arange(ntt), yt], 1e-12, 1))
print(f"  log-loss Total: latente={ll_lt.mean():.5f} vs marché={ll_mt.mean():.5f} (wilcoxon p={stats.wilcoxon(ll_lt, ll_mt).pvalue:.3g})")
print(f"  MAE(latente,marché) par T: " + " ".join(f"T{t_}={np.abs(lt[:,t_]-tm_dv[:,t_]).mean():.4f}" for t_ in range(7)))

# ----------------------------------------------------------------------------
# T4 cellules cappées
# ----------------------------------------------------------------------------
print(f"\n[T4] cellules cappées: obs={int((Y*capped).sum())} | "
      f"attendu latente={latent[capped].sum():.1f} | attendu devig prop={devig_prop[capped].sum():.1f} | n_cellmatchs={capped.sum()}")
kc = int((Y*capped).sum())
pv = stats.binomtest(kc, int(capped.sum()), float(latent[capped].mean()))
print(f"  binomial vs latente: p={pv.pvalue:.3g} (freq obs={kc/capped.sum():.5f} vs latente moy={latent[capped].mean():.5f})")

# ----------------------------------------------------------------------------
# T5 marge
# ----------------------------------------------------------------------------
print("\n[T5] STRUCTURE DE LA MARGE")
m_cell = np.where(~capped, imp/np.maximum(latent, 1e-12), np.nan)
c_match = np.nansum(np.where(~capped, imp, 0), axis=1)/np.nansum(np.where(~capped, latent, 0), axis=1)
print(f"  facteur marge par match (uncapped): mean={c_match.mean():.4f} sd={c_match.std():.4f} "
      f"p5={np.percentile(c_match,5):.4f} p95={np.percentile(c_match,95):.4f}")
ncap = capped.sum(axis=1)
sl, icept, r_, p_, se = stats.linregress(ncap, overround)
print(f"  overround = {icept:.4f} + {sl:.5f}*n_capped (r={r_:.3f}, p={p_:.2g}) — si cap ajoute 0.01/cellule, pente attendue ~0.01-")
mc = np.nanmean(m_cell, axis=0)
tt = pd.DataFrame(dict(cell=LBL, marge_moy=mc, imp_moy=np.nanmean(np.where(~capped, imp, np.nan), axis=0),
                       n_uncap=(~capped).sum(axis=0))).sort_values('imp_moy', ascending=False)
print("  marge moyenne par cellule (imp/latente, uncapped) — favorite-longshot ?")
print(tt.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ----------------------------------------------------------------------------
# T6 paramètres moteur
# ----------------------------------------------------------------------------
print("\n[T6] PARAMETRES MOTEUR")
ltot = lh + la; ldif = np.log(lh) - np.log(la)
exp_tot_mkt = (tm_dv * np.arange(7)).sum(axis=1)
print(f"  corr(lh+la, E[T] marché Total) = {np.corrcoef(ltot[ok_t], exp_tot_mkt)[0,1]:.5f}")
print(f"  corr(log lh - log la, logit pH/pA 1X2) = {np.corrcoef(ldif, np.log(p1x2[:,0]/p1x2[:,2]))[0,1]:.5f}")
print(f"  E[T] réel={tot_real.mean():.4f} | E[T] latente={(l_tot*np.arange(7)).sum(1).mean():.4f} | lh+la mean={ltot.mean():.4f}")
print(f"  E[score_a] réel={df['score_a'].mean():.4f} vs lh*corr_trunc — lh mean={lh.mean():.4f}")
print(f"  E[score_b] réel={df['score_b'].mean():.4f} | la mean={la.mean():.4f}")
# lattice ?
for nm, v in [('lh', lh), ('la', la), ('lh+la', ltot)]:
    fr01 = (v/0.05) % 1.0
    ks = stats.kstest(fr01, 'uniform')
    print(f"  lattice 0.05 sur {nm}: KS p={ks.pvalue:.3f} (p<0.01 => quantifié)")
print(f"  rho: % matchs |rho|>0.05 = {(np.abs(rho)>0.05).mean()*100:.1f}% ; quantiles "
      + " ".join(f"q{q_}={np.percentile(rho,q_):.4f}" for q_ in [5,25,50,75,95]))
# rho vs equilibre
pm = np.maximum(p1x2[:,0], p1x2[:,2])
print(f"  corr(rho, pmax 1X2) = {np.corrcoef(rho, pm)[0,1]:.4f} ; corr(rho, lh+la) = {np.corrcoef(rho, ltot)[0,1]:.4f}")

# ----------------------------------------------------------------------------
# T7 le moteur est-il EXACTEMENT DC ? résidu systématique par cellule
# ----------------------------------------------------------------------------
print("\n[T7] RESIDU log(imp) - log(c_match*latente) par cellule (uncapped) — t-test + FDR")
resid = np.where(~capped, np.log(imp) - np.log(np.maximum(c_match[:, None]*latent, 1e-12)), np.nan)
rows = []
for j, c in enumerate(LBL):
    v = resid[:, j]; v = v[~np.isnan(v)]
    if len(v) < 100: rows.append(dict(cell=c, n=len(v), mean=np.nan, t=np.nan, p=np.nan)); continue
    t_, p_ = stats.ttest_1samp(v, 0)
    rows.append(dict(cell=c, n=len(v), mean=v.mean(), sd=v.std(), t=t_, p=p_))
t7 = pd.DataFrame(rows)
t7['q_fdr'] = bh_fdr(t7['p'].fillna(1).values)
print(t7.sort_values('q_fdr').to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ----------------------------------------------------------------------------
# T8 WALK-FORWARD 70/30
# ----------------------------------------------------------------------------
print("\n" + "="*100)
cut = int(n*0.7); tr = np.arange(n) < cut; te = ~tr
print(f"[T8] WALK-FORWARD train n={tr.sum()} / OOS n={te.sum()} "
      f"(coupure {df['expected_start'].iloc[cut]})")

def run_oos(pick_te, odds_mat, win_mat, name):
    npick = int(pick_te.sum())
    if npick == 0:
        print(f"  {name}: 0 pari OOS"); return None
    ret = float((win_mat * odds_mat * pick_te).sum())
    roi = (ret-npick)/npick
    co = float(odds_mat[pick_te].mean())
    wr = float((win_mat*pick_te).sum()/npick)
    se_roi = float(np.sqrt(((win_mat*odds_mat*pick_te - (win_mat*odds_mat*pick_te).sum()/npick)**2).sum()/npick)/np.sqrt(npick)) if npick>1 else 0
    print(f"  {name}: n_OOS={npick} ROI={roi:+.4f} WR={wr:.3f} cote_moy={co:.2f}")
    return dict(n=npick, roi=roi, wr=wr, cote=co)

# W1: cellule x bucket sélection sur train (z>2 et ROI>+3%)
print("\n[W1] cellule x bucket-pmax (sélection train: z>2 & ROI>3%):")
buckets = pd.Series(pd.cut(pm, [0, 0.40, 0.50, 0.62, 1.0], labels=['eq','leg','fort','ext']))
found = False
for b in buckets.cat.categories:
    mb = (buckets == b).values
    for j, c in enumerate(LBL):
        m_tr = mb & tr & ~capped[:, j]
        if m_tr.sum() < 200: continue
        k = Y[m_tr, j].sum(); p = devig_prop[m_tr, j]; E = p.sum(); V = (p*(1-p)).sum()
        z = (k-E)/np.sqrt(V) if V > 0 else 0
        roi_tr = (Y[m_tr, j]*odds_grid[m_tr, j]).sum()/m_tr.sum()-1
        if z > 2.0 and roi_tr > 0.03:
            found = True
            m_te = mb & te & ~capped[:, j]
            pick = np.zeros((n, N_CELLS), bool); pick[m_te, j] = True
            run_oos(pick, odds_grid, Y, f"{b}/{c} (train z={z:+.2f} ROI={roi_tr:+.3f})")
if not found: print("  aucune combinaison sélectionnée sur train")

# W2: EV latente vs cotes CS
print("\n[W2] EV = latente*cote_CS - 1 (cellules non cappées):")
ev = latent*odds_grid - 1
for thr in [0.0, 0.03, 0.05, 0.10]:
    pick = (ev > thr) & ~capped & te[:, None]
    run_oos(pick, odds_grid, Y, f"thr={thr}")

# W3: EV latente->Total vs cotes Total ; latente->1X2 vs cotes 1X2
print("\n[W3] EV latente->Total / latente->1X2:")
odds3 = df[['odds_home','odds_draw','odds_away']].values
ev_t = l_tot*tot_odds - 1
for thr in [0.0, 0.03, 0.05]:
    pick = (ev_t > thr) & ~np.isnan(tot_odds) & te[:, None]
    Yt = np.zeros((n, 7)); Yt[np.arange(n), tot_real] = 1
    run_oos(pick, np.nan_to_num(tot_odds), Yt, f"Total thr={thr}")
ev3 = l1x2*odds3 - 1
Y3 = np.zeros((n, 3)); Y3[np.arange(n), out3] = 1
for thr in [0.0, 0.02, 0.04]:
    pick = (ev3 > thr) & te[:, None]
    run_oos(pick, odds3, Y3, f"1X2 thr={thr}")

# W4: fréquences empiriques train par (bucket,cellule) comme proba -> EV vs cote CS
print("\n[W4] proba = freq empirique train par bucket -> EV vs cote CS:")
emp = {}
for b in buckets.cat.categories:
    mb = (buckets == b).values & tr
    emp[b] = (Y[mb].sum(axis=0) + 1) / (mb.sum() + N_CELLS)  # laplace
p_emp = np.array([emp[b] for b in buckets])
ev4 = p_emp*odds_grid - 1
for thr in [0.0, 0.05, 0.10]:
    pick = (ev4 > thr) & ~capped & te[:, None]
    run_oos(pick, odds_grid, Y, f"thr={thr}")

print("\nDONE")
