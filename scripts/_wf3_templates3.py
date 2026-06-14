# -*- coding: utf-8 -*-
"""WF3c — affiner le modele du generateur de cotes.
a) residus du modele ratings: idiosyncrasie par PAIRE? symetrie (A,B)<->(B,A)
b) mecanisme d'arrondi: simulation p*1.06 -> round vs distribution overround observee
c) calibration globale resultats vs pH_bar (template paire) — chi2 agrege + par bin
d) jitter et forme recente (feedback loop?)
e) validation statistique strategie 'fade le jitter' + EV theorique sous bruit d'affichage
f) info dans le mouvement open->close: test plus puissant
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.id AS snap_id,
       r.score_a, r.score_b
FROM events e
JOIN results r ON r.event_id = e.id
JOIN odds_snapshots o ON o.event_id = e.id
WHERE e.round_info != '0'
"""
raw = pd.read_sql(q, eng).dropna(subset=['odds_home','odds_draw','odds_away','score_a','score_b'])
raw = raw.sort_values('snap_id')
last = raw.groupby('id', as_index=False).last()
df = raw.groupby('id', as_index=False).first()
df = df.drop_duplicates(['team_a','team_b','expected_start'], keep='first')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').reset_index(drop=True)
lc = last.set_index('id')
for c in ['odds_home','odds_draw','odds_away']:
    df['c_'+c[5:]] = df['id'].map(lc[c])
n = len(df)
H, D, A = df.odds_home.values, df.odds_draw.values, df.odds_away.values
s = 1/H + 1/D + 1/A
df['pH'], df['pD'], df['pA'] = (1/H)/s, (1/D)/s, (1/A)/s
df['win_h'] = (df.score_a > df.score_b).astype(int)
df['draw'] = (df.score_a == df.score_b).astype(int)
df['win_a'] = (df.score_a < df.score_b).astype(int)
g = df.groupby(['team_a','team_b'])
df['pH_bar'] = g['pH'].transform('mean'); df['pD_bar'] = g['pD'].transform('mean')
df['pA_bar'] = g['pA'].transform('mean'); df['npair'] = g['pH'].transform('count')

# ---------- a) symetrie des residus ----------
print("=== a) RESIDUS RATINGS — IDIOSYNCRASIE PAR PAIRE ?")
teams = sorted(set(df.team_a) | set(df.team_b)); tidx = {t:i for i,t in enumerate(teams)}
pm = g[['pH','pD','pA']].mean().reset_index()
pm['y'] = np.log(pm.pH/pm.pA)
X = np.zeros((len(pm), len(teams)))
for i, r in pm.iterrows():
    X[i, tidx[r.team_a]] += 1; X[i, tidx[r.team_b]] -= 1
X = np.column_stack([X[:,1:], np.ones(len(pm))])
coef, *_ = np.linalg.lstsq(X, pm.y.values, rcond=None)
pm['res'] = pm.y - X @ coef
rev = pm.set_index(['team_a','team_b'])['res']
pairs_sym = []
for (a,b), r1 in rev.items():
    if (b,a) in rev.index and a < b:
        pairs_sym.append((r1, rev[(b,a)]))
ps = np.array(pairs_sym)
c_sym = np.corrcoef(ps[:,0], ps[:,1])[0,1]
print(f"corr(residu(A@home), residu(B@home)) sur {len(ps)} paires non-ordonnees: {c_sym:.3f}")
print("  (~ -1 = idiosyncrasie de FORCE relative h2h persistante; ~0 = bruit d'estimation independant)")
# residual magnitude vs sampling noise: jitter std / sqrt(npair)
npair_mean = g.size().mean()
jit_lo = 0.0348  # log-odds jitter std mesure
# y = log(pH/pA): jitter sur y ~ ? estimer directement
df['y_match'] = np.log(df.pH/df.pA)
jy = (df.y_match - g['y_match'].transform('mean'))[df.npair>=10].std()
exp_res_std = jy/np.sqrt(npair_mean)
print(f"std(residu) observe: {pm.res.std():.4f} vs bruit d'echantillonnage attendu ~{exp_res_std:.4f} (jitter {jy:.3f}/sqrt({npair_mean:.0f}))")

# refit avec terme par paire non-ordonnee? -> test si les residus h2h sont structurels:
# split temporel: moyennes paire 1ere moitie vs residus 2eme moitie
df['half'] = (np.arange(len(df)) >= n//2).astype(int)
pm1 = df[df.half==0].groupby(['team_a','team_b'])['y_match'].agg(['mean','count'])
pm2 = df[df.half==1].groupby(['team_a','team_b'])['y_match'].agg(['mean','count'])
common = pm1.index.intersection(pm2.index)
pm1c, pm2c = pm1.loc[common], pm2.loc[common]
ok = (pm1c['count']>=5) & (pm2c['count']>=5)
# residus vs modele additif fitte sur moitie 1
X1 = np.zeros((ok.sum(), len(teams)))
y1 = pm1c[ok]['mean'].values; y2 = pm2c[ok]['mean'].values
for i, (a,b) in enumerate(pm1c[ok].index):
    X1[i, tidx[a]] += 1; X1[i, tidx[b]] -= 1
X1 = np.column_stack([X1[:,1:], np.ones(ok.sum())])
c1, *_ = np.linalg.lstsq(X1, y1, rcond=None)
r1 = y1 - X1@c1; r2 = y2 - X1@c1
cc = np.corrcoef(r1, r2)[0,1]
print(f"persistance temporelle des residus h2h (corr res_half1, res_half2): {cc:.3f} sur {ok.sum()} paires")
print("  (eleve = le moteur stocke des probas PAR PAIRE, pas juste des ratings)")

# ---------- b) mecanisme d'arrondi ----------
print("\n=== b) MECANISME: simulation arrondi")
rng = np.random.default_rng(42)
# hypothese: p_fair (3 probas exactes), odds = round(1/(p*1.06), 2)
pH_s = df.pH_bar.values + rng.normal(0, 0.0076, n)
pA_s = df.pA_bar.values + rng.normal(0, 0.0067, n)
pD_s = 1 - pH_s - pA_s
for mode, f in [('round', np.round), ('floor', np.floor), ('ceil', np.ceil)]:
    oh = f(100/(pH_s*1.06))/100; od = f(100/(pD_s*1.06))/100; oa = f(100/(pA_s*1.06))/100
    ov = 1/oh + 1/od + 1/oa
    print(f"  {mode}: overround simule mean={ov.mean():.5f} std={ov.std():.5f}  "
          f"(observe: mean=1.05999 std=0.00119)")

# ---------- c) calibration globale vs pH_bar (template paire) ----------
print("\n=== c) CALIBRATION RESULTATS vs TEMPLATE PAIRE (pH_bar/pD_bar/pA_bar)")
m = df.npair >= 10
sub = df[m]
chi2_tot, dof = 0.0, 0
for k, ss in sub.groupby(['team_a','team_b']):
    nn = len(ss)
    obs = np.array([ss.win_h.sum(), ss.draw.sum(), ss.win_a.sum()])
    exp = np.array([ss.pH_bar.iloc[0], ss.pD_bar.iloc[0], ss.pA_bar.iloc[0]]) * nn
    if (exp > 1).all():
        chi2_tot += ((obs-exp)**2/exp).sum(); dof += 2
p_glob = 1 - stats.chi2.cdf(chi2_tot, dof)
print(f"chi2 agrege sur paires (n>=10): chi2={chi2_tot:.1f}, dof={dof}, p={p_glob:.4f}")
# par bin de pH
print("calibration home par decile de pH:")
sub2 = df.copy()
sub2['bin'] = pd.qcut(sub2.pH, 10, duplicates='drop')
cal = sub2.groupby('bin', observed=True).agg(n=('win_h','size'), obs=('win_h','mean'), imp=('pH','mean'))
cal['p_binom'] = [stats.binomtest(int(r.obs*r.n), int(r.n), r.imp).pvalue for r in cal.itertuples()]
print(cal.to_string(float_format='%.4f'))
print("calibration DRAW par quintile de pD:")
sub2['binD'] = pd.qcut(sub2.pD, 5, duplicates='drop')
calD = sub2.groupby('binD', observed=True).agg(n=('draw','size'), obs=('draw','mean'), imp=('pD','mean'))
calD['p_binom'] = [stats.binomtest(int(r.obs*r.n), int(r.n), r.imp).pvalue for r in calD.itertuples()]
print(calD.to_string(float_format='%.4f'))

# ---------- d) jitter vs forme recente ----------
print("\n=== d) LE JITTER DEPEND-IL DE LA FORME RECENTE ? (feedback loop)")
df['jlH'] = np.log(df.pH/(1-df.pH)) - g['pH'].transform(lambda x: np.log(x.mean()/(1-x.mean())))
# construire forme: pour chaque equipe, resultat du match precedent (points)
ev = []
for r in df.itertuples():
    ev.append((r.team_a, r.expected_start, 3 if r.win_h else (1 if r.draw else 0), r.Index))
    ev.append((r.team_b, r.expected_start, 3 if r.win_a else (1 if r.draw else 0), r.Index))
evdf = pd.DataFrame(ev, columns=['team','ts','pts','match_idx']).sort_values('ts')
evdf['prev_pts'] = evdf.groupby('team')['pts'].shift(1)
evdf['prev3'] = evdf.groupby('team')['pts'].transform(lambda x: x.shift(1).rolling(3).mean())
prev_a = evdf.set_index(['match_idx','team'])
df['form_a'] = [prev_a.loc[(i, t), 'prev3'] if (i,t) in prev_a.index else np.nan
                for i, t in zip(df.index, df.team_a)]
df['form_b'] = [prev_a.loc[(i, t), 'prev3'] if (i,t) in prev_a.index else np.nan
                for i, t in zip(df.index, df.team_b)]
mm = df.dropna(subset=['form_a','form_b','jlH'])
mm = mm[mm.npair>=10]
ca = np.corrcoef(mm.form_a, mm.jlH)[0,1]
cb = np.corrcoef(mm.form_b, mm.jlH)[0,1]
na = len(mm)
print(f"corr(forme equipe home [3 derniers], jitter logit pH): {ca:+.4f} (n={na}, p={2*(1-stats.norm.cdf(abs(ca)*np.sqrt(na))):.4f})")
print(f"corr(forme equipe away, jitter): {cb:+.4f} (p={2*(1-stats.norm.cdf(abs(cb)*np.sqrt(na))):.4f})")
# round dans la saison?
df['round_i'] = df.round_info.astype(int)
cr = np.corrcoef(df[df.npair>=10].round_i, df[df.npair>=10].jlH)[0,1]
print(f"corr(round, jitter): {cr:+.4f}")

# ---------- e) strategie fade-jitter: stats propres ----------
print("\n=== e) VALIDATION STRATEGIE FADE-JITTER (walk-forward 70/30)")
cut = int(n*0.7)
tr, te = df.iloc[:cut], df.iloc[cut:]
pmH = tr.groupby(['team_a','team_b'])['pH'].agg(['mean','count'])
pmA = tr.groupby(['team_a','team_b'])['pA'].agg(['mean','count'])
pmHd = {k:(v['mean'],v['count']) for k,v in pmH.iterrows()}
pmAd = {k:(v['mean'],v['count']) for k,v in pmA.iterrows()}
print("EV theorique sous H0 'bruit d'affichage': ROI = p_bar/(p_jit*1.06)-1")
for thr in [0.005, 0.010, 0.015]:
    bets = []
    for r in te.itertuples():
        k = (r.team_a, r.team_b)
        if k not in pmHd or pmHd[k][1] < 8: continue
        eh = pmHd[k][0] - r.pH; ea = pmAd[k][0] - r.pA
        if eh > thr:
            bets.append(('H', r.odds_home, r.win_h, pmHd[k][0]/(r.pH*1.06)-1))
        if ea > thr:
            bets.append(('A', r.odds_away, r.win_a, pmAd[k][0]/(r.pA*1.06)-1))
    if not bets: continue
    bdf = pd.DataFrame(bets, columns=['side','odds','win','ev_theo'])
    pnl = (bdf.win*(bdf.odds-1) - (1-bdf.win)).sum()
    roi = pnl/len(bdf)
    # p-value: nb wins vs implied-fair proba
    p_fair = (1/bdf.odds/1.06*1.06).mean()  # implied with margin = breakeven prob
    nwin = bdf.win.sum()
    pv = stats.binomtest(int(nwin), len(bdf), (1/bdf.odds).mean()).pvalue
    se_roi = bdf.assign(r=lambda x: x.win*(x.odds-1)-(1-x.win)).r.std()/np.sqrt(len(bdf))
    print(f"  thr={thr:.3f}: n={len(bdf)}, ROI={roi*100:+.1f}% (se={se_roi*100:.1f}%), "
          f"EV theorique bruit-affichage={bdf.ev_theo.mean()*100:+.1f}%, p_binom(win vs 1/odds)={pv:.3f}")

# ---------- f) mouvement open->close: test plus puissant ----------
print("\n=== f) INFO DANS LE MOUVEMENT OPEN->CLOSE")
mv = df[(df.odds_home != df.c_home) | (df.odds_away != df.c_away) | (df.odds_draw != df.c_draw)].copy()
sc = 1/mv.c_home + 1/mv.c_draw + 1/mv.c_away
mv['pH_c'] = (1/mv.c_home)/sc; mv['pA_c'] = (1/mv.c_away)/sc
mv['dlH'] = np.log(mv.pH_c/(1-mv.pH_c)) - np.log(mv.pH/(1-mv.pH))
print(f"n moved={len(mv)}, delta logit pH: std={mv.dlH.std():.4f}")
big_mv = mv[abs(mv.dlH) > mv.dlH.std()]
# le delta ressemble-t-il a un RE-TIRAGE du jitter ? corr(jitter_open, delta)
jo = mv.jlH; cd = np.corrcoef(jo[jo.notna()], mv.dlH[jo.notna()])[0,1]
print(f"corr(jitter_open, delta open->close): {cd:.3f}  (-0.5 attendu si close = nouveau tirage independant)")
# accuracy: brier open vs close sur big movers
for nm, col in [('open','pH'), ('close','pH_c')]:
    b = ((big_mv[col]-big_mv.win_h)**2).mean()
    print(f"  Brier {nm} (big movers, n={len(big_mv)}): {b:.5f}")

# logit(win_h) ~ logit(p_close) + delta : info au-dela du close ?
import numpy.linalg as la
def logit_fit(Xm, yv, iters=100):
    w = np.zeros(Xm.shape[1])
    for _ in range(iters):
        p = 1/(1+np.exp(-Xm@w)); Wd = p*(1-p) + 1e-9
        w += la.solve((Xm * Wd[:,None]).T @ Xm + 1e-9*np.eye(Xm.shape[1]), Xm.T@(yv-p))
    p = 1/(1+np.exp(-Xm@w)); Wd = p*(1-p)
    return w, np.sqrt(np.diag(la.inv((Xm * Wd[:,None]).T @ Xm)))
loF = lambda p: np.log(p/(1-p))
Xm = np.column_stack([loF(mv.pH_c), mv.dlH, np.ones(len(mv))])
w, se = logit_fit(Xm, mv.win_h.values)
print(f"logit(win_h) ~ logit(p_CLOSE) + delta: b_delta={w[1]:.3f} +/- {1.96*se[1]:.3f} "
      f"(>0 = le mouvement contient PLUS d'info que le close n'en price)")
Xm2 = np.column_stack([loF(mv.pH), mv.dlH, np.ones(len(mv))])
w2, se2 = logit_fit(Xm2, mv.win_h.values)
print(f"logit(win_h) ~ logit(p_OPEN)  + delta: b_delta={w2[1]:.3f} +/- {1.96*se2[1]:.3f} "
      f"(1 = close efficient; 0 = open efficient)")

# pari directionnel: si delta logit home > +thr, bet HOME au close; < -thr -> AWAY au close
print("\npari 'suis le mouvement' (au CLOSE), full sample + OOS 30%:")
cut70 = df.expected_start.quantile(0.70)
for thr in [0.05, 0.10, 0.15]:
    for scope, mvv in [('full', mv), ('OOS30', mv[mv.expected_start > cut70])]:
        bh = mvv[mvv.dlH > thr]; ba = mvv[mvv.dlH < -thr]
        nb = len(bh) + len(ba)
        if nb < 10:
            print(f"  thr={thr} [{scope}]: n={nb} (trop peu)"); continue
        pnl = (bh.win_h*(bh.c_home-1) - (1-bh.win_h)).sum() + \
              (ba.win_a*(ba.c_away-1) - (1-ba.win_a)).sum()
        wins = bh.win_h.sum() + ba.win_a.sum()
        per = np.concatenate([(bh.win_h*(bh.c_home-1) - (1-bh.win_h)).values,
                              (ba.win_a*(ba.c_away-1) - (1-ba.win_a)).values])
        se_roi = per.std()/np.sqrt(nb)
        print(f"  thr={thr} [{scope}]: n={nb}, WR={wins/nb*100:.1f}%, "
              f"ROI={pnl/nb*100:+.1f}% (se={se_roi*100:.1f}%)")
print("\nDONE")
