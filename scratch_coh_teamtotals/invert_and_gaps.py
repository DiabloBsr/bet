# coh_teamtotals — step 1: invert (lambda_h, lambda_a) from devig p|1X2| via independent Poisson,
# compute model-implied team-total / G-NG probabilities, gaps vs quoted devig p, calibration check.
import numpy as np, pandas as pd
from scipy.special import gammaln

CSV = r"D:\AGENTOVA\SAMY\virtual-sports-scraper\data\vfoot_ml\conjunctive_wide.csv"
OUT = r"D:\AGENTOVA\SAMY\virtual-sports-scraper\scratch_coh_teamtotals\gaps.csv"
K = 20
usecols = ['ts','sa','sb','phase','p|1X2|1','p|1X2|X','p|1X2|2',
 'p|Total equipe domicile|> 3.5','p|Total equipe domicile|< 3.5',
 'p|Total equipe extérieur|> 3.5','p|Total equipe extérieur|< 3.5',
 'p|G/NG equipe domicile|Oui','p|G/NG equipe domicile|Non',
 'p|G/NG equipe extérieur|Oui','p|G/NG equipe extérieur|Non']
df = pd.read_csv(CSV, usecols=usecols)
n = len(df)
t1 = df['p|1X2|1'].to_numpy(); t2 = df['p|1X2|2'].to_numpy()
kk = np.arange(K+1)
lgam = gammaln(kk+1)

def pmats(lh, la):
    Ph = np.exp(kk[None,:]*np.log(lh)[:,None] - lh[:,None] - lgam[None,:])
    Pa = np.exp(kk[None,:]*np.log(la)[:,None] - la[:,None] - lgam[None,:])
    return Ph, Pa

def p12(loglh, logla):
    lh = np.exp(loglh); la = np.exp(logla)
    Ph, Pa = pmats(lh, la)
    Fh = np.cumsum(Ph,1); Fa = np.cumsum(Pa,1)
    p1 = (Ph[:,1:]*Fa[:,:-1]).sum(1)
    p2 = (Pa[:,1:]*Fh[:,:-1]).sum(1)
    return p1, p2

# --- coarse grid init ---
g = np.exp(np.linspace(np.log(0.05), np.log(4.5), 80))
GH, GA = np.meshgrid(g, g, indexing='ij')
glh = np.log(GH.ravel()); gla = np.log(GA.ravel())
gp1, gp2 = p12(glh, gla)
init_lh = np.empty(n); init_la = np.empty(n)
for s in range(0, n, 2000):
    e = min(s+2000, n)
    d = (t1[s:e,None]-gp1[None,:])**2 + (t2[s:e,None]-gp2[None,:])**2
    idx = d.argmin(1)
    init_lh[s:e] = glh[idx]; init_la[s:e] = gla[idx]

# --- vectorized damped Newton on (log lh, log la) ---
x1, x2 = init_lh.copy(), init_la.copy()
eps = 1e-5
for it in range(40):
    p1, p2 = p12(x1, x2)
    r1 = p1 - t1; r2 = p2 - t2
    if max(np.abs(r1).max(), np.abs(r2).max()) < 1e-9: break
    p1a, p2a = p12(x1+eps, x2)
    p1b, p2b = p12(x1, x2+eps)
    J11 = (p1a-p1)/eps; J12 = (p1b-p1)/eps
    J21 = (p2a-p2)/eps; J22 = (p2b-p2)/eps
    det = J11*J22 - J12*J21
    det = np.where(np.abs(det) < 1e-14, np.sign(det)*1e-14 + (det==0)*1e-14, det)
    dx1 = (J22*r1 - J12*r2)/det
    dx2 = (-J21*r1 + J11*r2)/det
    dx1 = np.clip(dx1, -0.5, 0.5); dx2 = np.clip(dx2, -0.5, 0.5)
    x1 -= dx1; x2 -= dx2
p1, p2 = p12(x1, x2)
res = np.maximum(np.abs(p1-t1), np.abs(p2-t2))
lh = np.exp(x1); la = np.exp(x2)
print(f"Newton iters used: {it+1}; residual max={res.max():.2e} p99={np.quantile(res,0.99):.2e} median={np.median(res):.2e}")
print(f"lambda_h: min={lh.min():.3f} med={np.median(lh):.3f} max={lh.max():.3f} | lambda_a: min={la.min():.3f} med={np.median(la):.3f} max={la.max():.3f}")

# --- model-implied probabilities ---
Ph, Pa = pmats(lh, la)
Fh = np.cumsum(Ph,1); Fa = np.cumsum(Pa,1)
out = df[['ts','sa','sb','phase']].copy()
out['lam_h'] = lh; out['lam_a'] = la; out['fit_res'] = res
out['m_TTdom_gt'] = 1 - Fh[:,3]           # P(home goals >= 4)
out['m_TText_gt'] = 1 - Fa[:,3]
out['m_GNGdom_oui'] = 1 - np.exp(-lh)     # P(home scores)
out['m_GNGext_oui'] = 1 - np.exp(-la)
out['q_TTdom_gt'] = df['p|Total equipe domicile|> 3.5']
out['q_TText_gt'] = df['p|Total equipe extérieur|> 3.5']
out['q_TTdom_lt'] = df['p|Total equipe domicile|< 3.5']
out['q_TText_lt'] = df['p|Total equipe extérieur|< 3.5']
out['q_GNGdom_oui'] = df['p|G/NG equipe domicile|Oui']
out['q_GNGdom_non'] = df['p|G/NG equipe domicile|Non']
out['q_GNGext_oui'] = df['p|G/NG equipe extérieur|Oui']
out['q_GNGext_non'] = df['p|G/NG equipe extérieur|Non']
for mkt, q in [('TTdom_gt','q_TTdom_gt'),('TText_gt','q_TText_gt'),
               ('GNGdom_oui','q_GNGdom_oui'),('GNGext_oui','q_GNGext_oui')]:
    out['gap_'+mkt] = out['m_'+mkt] - out[q]
out.to_csv(OUT, index=False)

print("\n=== GAP model - quoted (reference side: >3.5 / Oui) ===")
for mkt in ['TTdom_gt','TText_gt','GNGdom_oui','GNGext_oui']:
    gcol = out['gap_'+mkt]
    for ph in ['train','test']:
        gph = gcol[out.phase==ph].dropna()
        print(f"{mkt:12s} {ph:5s} n={len(gph):6d} mean={gph.mean():+.5f} sd={gph.std():.5f} "
              f"q05={gph.quantile(.05):+.4f} q50={gph.quantile(.5):+.4f} q95={gph.quantile(.95):+.4f} "
              f"|gap|>0.02: {(gph.abs()>0.02).mean()*100:5.1f}%  >0.05: {(gph.abs()>0.05).mean()*100:5.1f}%")

print("\n=== CALIBRATION (who is right?) — Brier & logloss, quoted vs Poisson-model, per outcome (all rows w/ quote) ===")
ycols = {'TTdom_gt': (out.sa>=4).astype(float), 'TText_gt': (out.sb>=4).astype(float),
         'GNGdom_oui': (out.sa>=1).astype(float), 'GNGext_oui': (out.sb>=1).astype(float)}
for mkt, y in ycols.items():
    q = out['q_'+mkt]; m = out['m_'+mkt]; msk = q.notna()
    yv = y[msk].to_numpy(); qv = np.clip(q[msk].to_numpy(),1e-6,1-1e-6); mv = np.clip(m[msk].to_numpy(),1e-6,1-1e-6)
    bq = np.mean((qv-yv)**2); bm = np.mean((mv-yv)**2)
    lq = -np.mean(yv*np.log(qv)+(1-yv)*np.log(1-qv)); lm = -np.mean(yv*np.log(mv)+(1-yv)*np.log(1-mv))
    print(f"{mkt:12s} n={msk.sum():6d} freq={yv.mean():.4f} mean_q={qv.mean():.4f} mean_m={mv.mean():.4f} | Brier q={bq:.5f} m={bm:.5f} | logloss q={lq:.5f} m={lm:.5f} -> {'MODEL better' if bm<bq else 'QUOTED better'}")
