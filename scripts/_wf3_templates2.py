# -*- coding: utf-8 -*-
"""WF3b — le catalogue est au niveau MATCHUP (380 paires) + jitter.
1. Verif draw = 1/(1.06 - 1/H - 1/A) (overround exact)
2. Structure du jitter intra-paire: amplitude, correlation entre legs, distribution
3. Ratings: paire-mean = f(rating_a, rating_b)? (modele additif log-odds)
4. Stabilite temporelle des moyennes par paire (feedback loop?)
5. EXPLOIT: le jitter est-il du bruit d'affichage (resultats tires des probas de base)
   ou un vrai changement de proba? -> walk-forward
6. Mouvements entre snapshots: amplitude, ouverture vs cloture predictif?
"""
import sys, json
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from collections import Counter
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.id AS snap_id, o.captured_at,
       r.score_a, r.score_b
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
WHERE e.round_info != '0'
"""
raw = pd.read_sql(q, eng)
raw = raw.dropna(subset=['odds_home','odds_draw','odds_away','score_a','score_b'])

# opening + closing odds
raw = raw.sort_values('snap_id')
opening = raw.groupby('id', as_index=False).first()
closing = raw.groupby('id', as_index=False).last()
df = opening.drop_duplicates(['team_a','team_b','expected_start'], keep='first').copy()
df = df.sort_values('expected_start').reset_index(drop=True)
clos = closing.set_index('id')
df['c_home'] = df['id'].map(clos['odds_home'])
df['c_draw'] = df['id'].map(clos['odds_draw'])
df['c_away'] = df['id'].map(clos['odds_away'])
n = len(df)
print("matchs:", n)

H, D, A = df.odds_home.values, df.odds_draw.values, df.odds_away.values

# ---------- 1. draw determinisme via overround 1.06 ----------
print("\n=== 1. DRAW = 1/(1.06 - 1/H - 1/A) ?")
D_pred = 1/(1.06 - 1/H - 1/A)
res = D - D_pred
print(f"residus: mean={res.mean():.4f}, std={res.std():.4f}, max|res|={np.abs(res).max():.3f}")
print(f"|res|<=0.05: {(np.abs(res)<=0.05).mean()*100:.1f}%, <=0.10: {(np.abs(res)<=0.10).mean()*100:.1f}%")
# expected residual from rounding H and A to cents: dD = D^2 * (dH/H^2 + dA/A^2)
exp_std = (D_pred**2 * np.sqrt((0.005/H**2)**2 + (0.005/A**2)**2) / np.sqrt(3)).mean()
print(f"std attendu si seule source = arrondi au cent de H et A: ~{exp_std:.4f}")

# ---------- 2. structure du jitter intra-paire ----------
print("\n=== 2. JITTER INTRA-PAIRE")
df['iH'], df['iD'], df['iA'] = 1/H, 1/D, 1/A
s = df.iH + df.iD + df.iA
df['pH'], df['pD'], df['pA'] = df.iH/s, df.iD/s, df.iA/s
g = df.groupby(['team_a','team_b'])
df['pH_bar'] = g['pH'].transform('mean')
df['pD_bar'] = g['pD'].transform('mean')
df['pA_bar'] = g['pA'].transform('mean')
df['npair'] = g['pH'].transform('count')
df['jH'] = df.pH - df.pH_bar
df['jA'] = df.pA - df.pA_bar
df['jD'] = df.pD - df.pD_bar
m = df.npair >= 10
print(f"paires n>=10: {df[m].groupby(['team_a','team_b']).ngroups}, matchs: {m.sum()}")
print(f"jitter pH: std={df[m].jH.std():.4f} (en proba), pD: {df[m].jD.std():.4f}, pA: {df[m].jA.std():.4f}")
print(f"corr(jH, jA) = {np.corrcoef(df[m].jH, df[m].jA)[0,1]:.3f}  (-1 = un seul param de force)")
print(f"corr(jH, jD) = {np.corrcoef(df[m].jH, df[m].jD)[0,1]:.3f}, corr(jA,jD)={np.corrcoef(df[m].jA, df[m].jD)[0,1]:.3f}")
# jitter in log-odds space — relative
df['lH'] = np.log(df.pH/(1-df.pH))
df['jlH'] = df.lH - g['lH'].transform('mean')
print(f"jitter log-odds(pH): std={df[m].jlH.std():.4f}")
# normality
ks = stats.kstest((df[m].jlH - df[m].jlH.mean())/df[m].jlH.std(), 'norm')
print(f"KS normalite jitter log-odds: stat={ks.statistic:.4f}, p={ks.pvalue:.4f}")
# jitter autocorrelation within pair over time (is it a walk or iid?)
ac = []
for _, sub in df[m].groupby(['team_a','team_b']):
    sub = sub.sort_values('expected_start')
    if len(sub) >= 10:
        x = sub.jH.values
        ac.append(np.corrcoef(x[:-1], x[1:])[0,1])
ac = np.array(ac); ac = ac[~np.isnan(ac)]
t_ac = ac.mean()/(ac.std()/np.sqrt(len(ac)))
print(f"autocorr lag-1 du jitter intra-paire: mean={ac.mean():.3f} (t={t_ac:.2f}, n={len(ac)} paires)")

# ---------- 3. ratings additifs ----------
print("\n=== 3. MODELE RATINGS: logit(pH/(pH+pA)) ~ r_a - r_b + hfa")
teams = sorted(set(df.team_a) | set(df.team_b))
tidx = {t:i for i,t in enumerate(teams)}
pm = df.groupby(['team_a','team_b'])[['pH','pD','pA']].mean().reset_index()
y = np.log(pm.pH/pm.pA)   # log odds-ratio home vs away
X = np.zeros((len(pm), len(teams)))
for i, r in pm.iterrows():
    X[i, tidx[r.team_a]] += 1
    X[i, tidx[r.team_b]] -= 1
X = np.column_stack([X[:,1:], np.ones(len(pm))])  # drop 1 team (ref), add hfa
coef, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ coef
r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
print(f"R2 du modele additif sur les 380 paires (moyennes): {r2:.5f}, RMSE={np.sqrt(((y-pred)**2).mean()):.4f}")
print(f"home advantage (log-OR): {coef[-1]:.4f}")
ratings = np.concatenate([[0.0], coef[:-1]])
rk = sorted(zip(teams, ratings), key=lambda x: -x[1])
print("ratings (log-force, ref=%s):" % teams[0])
for t, r in rk: print(f"  {t:<16} {r:+.3f}")
# does pD follow from pH-pA gap?
pm['gap'] = abs(np.log(pm.pH/pm.pA))
cd = np.corrcoef(pm.gap, pm.pD)[0,1]
print(f"corr(|gap log|, pD) sur paires: {cd:.3f}")

# ---------- 4. stabilite temporelle des moyennes par paire ----------
print("\n=== 4. STABILITE TEMPORELLE DES PAIRES (1ere vs 2eme moitie)")
df['half'] = (np.arange(len(df)) >= len(df)//2).astype(int)
rows = []
for k, sub in df.groupby(['team_a','team_b']):
    a = sub[sub.half==0].pH; b = sub[sub.half==1].pH
    if len(a)>=5 and len(b)>=5:
        t, p = stats.ttest_ind(a, b)
        rows.append((k, len(a), len(b), a.mean(), b.mean(), p))
st = pd.DataFrame(rows, columns=['pair','n1','n2','m1','m2','p'])
print(f"paires testables: {len(st)}; p<0.05: {(st.p<0.05).sum()} (attendu sous H0: {0.05*len(st):.1f}); p<0.01: {(st.p<0.01).sum()}")
print(f"drift moyen |m2-m1|: {abs(st.m2-st.m1).mean():.4f} en proba")
worst = st.sort_values('p').head(5)
print(worst.to_string(index=False))

# ---------- 5. EXPLOIT: jitter = bruit d'affichage ? ----------
print("\n=== 5. LE JITTER PREDIT-IL LE RESULTAT ? (logistic check + walk-forward)")
df['win_h'] = (df.score_a > df.score_b).astype(int)
df['win_a'] = (df.score_a < df.score_b).astype(int)
# In-sample global check: regression du resultat sur pH_bar et jitter
# si resultats tires des probas JITTEREES -> coef(jH) ~ coef(pH_bar) ~ 1
# si tires des probas de BASE -> coef(jH) ~ 0
import numpy.linalg as la
def logit_fit(Xm, yv, iters=200):
    w = np.zeros(Xm.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-Xm@w))
        Wd = p*(1-p) + 1e-9
        grad = Xm.T@(yv-p)
        Hm = (Xm * Wd[:,None]).T @ Xm + 1e-9*np.eye(Xm.shape[1])
        w += la.solve(Hm, grad)
    p = 1/(1+np.exp(-Xm@w))
    Wd = p*(1-p)
    cov = la.inv((Xm * Wd[:,None]).T @ Xm)
    return w, np.sqrt(np.diag(cov))
sub = df[df.npair>=10].copy()
lo = lambda p: np.log(p/(1-p))
Xm = np.column_stack([lo(sub.pH_bar), lo(sub.pH) - lo(sub.pH_bar), np.ones(len(sub))])
w, se = logit_fit(Xm, sub.win_h.values)
print(f"logit(win_h) ~ b0 + b1*logit(pH_bar) + b2*jitter_logit:")
print(f"  b1 (base) = {w[0]:.3f} +/- {se[0]:.3f}")
print(f"  b2 (jitter)= {w[1]:.3f} +/- {se[1]:.3f}   (1=jitter reel, 0=bruit d'affichage)")

# walk-forward: train 70% -> pair means; OOS 30%: bet side dont la cote est gonflee vs pair mean
cut = int(n*0.7)
tr, te = df.iloc[:cut], df.iloc[cut:].copy()
pmH = tr.groupby(['team_a','team_b'])['pH'].agg(['mean','count'])
pmA = tr.groupby(['team_a','team_b'])['pA'].agg(['mean','count'])
te['key'] = list(zip(te.team_a, te.team_b))
pmH_d = {k:(v['mean'],v['count']) for k,v in pmH.iterrows()}
pmA_d = {k:(v['mean'],v['count']) for k,v in pmA.iterrows()}
results = []
for thr in [0.0, 0.005, 0.01, 0.015]:
    pnl_h, nb_h, pnl_a, nb_a, codds_h, codds_a = 0.0, 0, 0.0, 0, [], []
    for r in te.itertuples():
        k = (r.team_a, r.team_b)
        if k not in pmH_d or pmH_d[k][1] < 8: continue
        # edge home: base prob (train) vs implied now
        edge_h = pmH_d[k][0] - r.pH
        edge_a = pmA_d[k][0] - r.pA
        if edge_h > thr:
            nb_h += 1; codds_h.append(r.odds_home)
            pnl_h += (r.odds_home - 1) if r.win_h else -1
        if edge_a > thr:
            nb_a += 1; codds_a.append(r.odds_away)
            pnl_a += (r.odds_away - 1) if r.win_a else -1
    tot = pnl_h + pnl_a; nb = nb_h + nb_a
    print(f"  thr={thr:.3f}: HOME n={nb_h} roi={pnl_h/max(nb_h,1)*100:+.1f}% | "
          f"AWAY n={nb_a} roi={pnl_a/max(nb_a,1)*100:+.1f}% | TOT n={nb} roi={tot/max(nb,1)*100:+.1f}%")

# ---------- 6. snapshots: mouvement open->close ----------
print("\n=== 6. MOUVEMENT DES COTES OPEN -> CLOSE")
mv = df.dropna(subset=['c_home'])
moved = mv[(mv.odds_home != mv.c_home) | (mv.odds_away != mv.c_away)]
print(f"matchs avec close != open: {len(moved)} / {len(mv)}")
dh = (moved.c_home - moved.odds_home)
print(f"delta odds_home: mean={dh.mean():+.4f}, std={dh.std():.4f}, max|d|={dh.abs().max():.2f}")
# close implied probs
ic = 1/moved.c_home + 1/moved.c_draw + 1/moved.c_away
print(f"overround close: mean={ic.mean():.5f}, std={ic.std():.5f}")
# which predicts better? log-loss open vs close on moved subset
for nm, (oh, od, oa) in [('open',(moved.odds_home,moved.odds_draw,moved.odds_away)),
                          ('close',(moved.c_home,moved.c_draw,moved.c_away))]:
    si = 1/oh + 1/od + 1/oa
    ph, pd_, pa = (1/oh)/si, (1/od)/si, (1/oa)/si
    y = np.where(moved.score_a>moved.score_b, 0, np.where(moved.score_a==moved.score_b, 1, 2))
    P = np.column_stack([ph, pd_, pa])
    ll = -np.log(np.clip(P[np.arange(len(y)), y], 1e-9, 1)).mean()
    print(f"  log-loss {nm}: {ll:.5f}  (n={len(moved)})")
# le mouvement contient-il de l'info ? logit(win) ~ p_open + delta
mm = moved.copy()
si_o = 1/mm.odds_home + 1/mm.odds_draw + 1/mm.odds_away
si_c = 1/mm.c_home + 1/mm.c_draw + 1/mm.c_away
po = (1/mm.odds_home)/si_o; pc = (1/mm.c_home)/si_c
Xm2 = np.column_stack([lo(po), lo(pc)-lo(po), np.ones(len(mm))])
yv = (mm.score_a>mm.score_b).astype(int).values
w2, se2 = logit_fit(Xm2, yv)
print(f"logit(win_h) ~ logit(p_open) + delta_logit(close-open): b_delta={w2[1]:.3f} +/- {se2[1]:.3f}")
print("\nDONE")
