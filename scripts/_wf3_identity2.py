# -*- coding: utf-8 -*-
"""
WF3 IDENTITY - PART 2 : mecanique fine
a) structure du jitter de cotes intra-paire (forme, correlations, informatif ?)
b) base de paire = additive en 20 ratings (logit) ou table 380 ?
c) GLM venue-specific (80 params) vs partage (41) : LRT
d) sous-dispersion + generation des nuls (Poisson indep vs realise vs implicite)
e) stabilite des snapshots intra-event
"""
import sys, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 220)
eng = create_engine(load_settings().db_url)

with eng.connect() as c:
    ev = pd.read_sql(text("""
        SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start,
               r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE e.round_info != '0' AND r.score_a IS NOT NULL
    """), c)
    od = pd.read_sql(text("""
        SELECT id, event_id, odds_home, odds_draw, odds_away FROM odds_snapshots
        WHERE odds_home IS NOT NULL ORDER BY id
    """), c)

od_open = od.groupby('event_id', as_index=False).first()
df = ev.merge(od_open[['event_id', 'odds_home', 'odds_draw', 'odds_away']],
              left_on='id', right_on='event_id')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').drop_duplicates(
    subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['pH'] = (1/df.odds_home)/inv
df['pD'] = (1/df.odds_draw)/inv
df['pA'] = (1/df.odds_away)/inv
df['res'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
teams = sorted(set(df.team_a))
print(f"n={len(df)}")

# ---------------------------------------------------------------- e) snapshots intra-event
print("\n=== e) STABILITE DES SNAPSHOTS INTRA-EVENT ===")
od_f = od[od.event_id.isin(df.id)]
g = od_f.groupby('event_id').agg(n=('id', 'size'),
                                 nuh=('odds_home', 'nunique'),
                                 nud=('odds_draw', 'nunique'),
                                 nua=('odds_away', 'nunique'))
multi = g[g.n > 1]
print(f"events avec >1 snapshot: {len(multi)} / {len(g)}")
if len(multi):
    moved = ((multi.nuh > 1) | (multi.nud > 1) | (multi.nua > 1))
    print(f"events dont les cotes BOUGENT entre snapshots: {moved.sum()} ({moved.mean()*100:.2f}%)")

# ---------------------------------------------------------------- a) jitter
print("\n=== a) STRUCTURE DU JITTER INTRA-PAIRE ===")
for col in ['pH', 'pD', 'pA']:
    df[col+'_dm'] = df[col] - df.groupby(['team_a', 'team_b'])[col].transform('mean')
df['pH_pair'] = df.groupby(['team_a', 'team_b'])['pH'].transform('mean')
n_pair = df.groupby(['team_a', 'team_b'])['pH'].transform('count')
j = df[n_pair >= 10].copy()
# correction biais de demeaning: var observee = var_vraie*(1-1/n)
corr_f = math.sqrt(1 - 1/n_pair[n_pair >= 10].mean())
for col in ['pH_dm', 'pD_dm', 'pA_dm']:
    x = j[col].values
    sd = x.std()/corr_f
    kurt = stats.kurtosis(x, fisher=False)
    w = sd*math.sqrt(12)
    ks_u = stats.kstest(x, stats.uniform(loc=-w/2, scale=w).cdf)
    ks_n = stats.kstest(x, stats.norm(0, x.std()).cdf)
    print(f"{col}: sd={sd:.5f} (width_unif_equiv={w:.4f}) kurtosis={kurt:.2f} "
          f"[unif=1.8, norm=3] KS_unif p={ks_u.pvalue:.3f} KS_norm p={ks_n.pvalue:.2e}")
print("correlations du jitter: "
      f"corr(H,D)={np.corrcoef(j.pH_dm, j.pD_dm)[0,1]:+.3f} "
      f"corr(H,A)={np.corrcoef(j.pH_dm, j.pA_dm)[0,1]:+.3f} "
      f"corr(D,A)={np.corrcoef(j.pD_dm, j.pA_dm)[0,1]:+.3f}")
# le jitter depend-il du niveau de pH ? (multiplicatif vs additif)
buck = pd.cut(j.pH_pair, [0, .2, .35, .5, .65, .8, 1])
print("\nsd du jitter pH par niveau de pH_pair (additif => constant):")
print(j.groupby(buck, observed=True).pH_dm.std().round(5).to_string())
# en logit ?
j['lg_dm'] = np.log(j.pH/(1-j.pH)) - j.groupby(['team_a', 'team_b'])['pH'].transform(
    lambda s: np.log(s/(1-s)).mean())
print("\nsd du jitter LOGIT(pH) par niveau (multiplicatif-logit => constant):")
print(j.groupby(buck, observed=True).lg_dm.std().round(4).to_string())

# JITTER INFORMATIF ? y_dm ~ beta * p_dm (stack H/D/A)
print("\n--- jitter informatif ? (le resultat est-il tire de la proba jitteree ?) ---")
ys, xs = [], []
for out, p in [('H', 'pH'), ('D', 'pD'), ('A', 'pA')]:
    y = (df.res == out).astype(float)
    y_dm = y - y.groupby([df.team_a, df.team_b]).transform('mean')
    ys.append(y_dm.values); xs.append(df[p+'_dm'].values)
Y = np.concatenate(ys); X = np.concatenate(xs)
beta = (X*Y).sum()/(X*X).sum()
se = math.sqrt(((Y-beta*X)**2).mean()/(X*X).sum())
print(f"beta={beta:+.3f} se={se:.3f} (beta=1 si le tirage utilise la proba jitteree, 0 si jitter cosmetique)")
print(f"t(beta=0)={beta/se:+.2f} ; t(beta=1)={(beta-1)/se:+.2f}")

# ---------------------------------------------------------------- b) base paire additive en logit ?
print("\n=== b) BASE DE PAIRE: ADDITIVE EN 20 RATINGS (logit) OU TABLE 380 ? ===")
pair = df.groupby(['team_a', 'team_b']).agg(pH=('pH', 'mean'), pD=('pD', 'mean'),
                                            pA=('pA', 'mean'), n=('pH', 'size')).reset_index()
idx = {t: i for i, t in enumerate(teams)}
nT = len(teams)


def fit_additive(yvals, label):
    Xd = np.zeros((len(pair), 2*nT+1))
    for i, r in pair.iterrows():
        Xd[i, idx[r.team_a]] = 1
        Xd[i, nT+idx[r.team_b]] = 1
        Xd[i, -1] = 1
    # drop 2 colonnes pour identifiabilite
    keep = [c for c in range(2*nT+1) if c not in (nT-1, 2*nT-1)]
    sol, *_ = np.linalg.lstsq(Xd[:, keep], yvals, rcond=None)
    pred = Xd[:, keep] @ sol
    ss_res = ((yvals-pred)**2).sum()
    ss_tot = ((yvals-yvals.mean())**2).sum()
    resid = yvals - pred
    print(f"{label}: R2={1-ss_res/ss_tot:.5f} sd_resid={resid.std():.4f} max|resid|={np.abs(resid).max():.4f}")
    return resid


lgH = np.log(pair.pH/(1-pair.pH)).values
rH = fit_additive(lgH, "logit(pH) ~ alpha_home(a) + beta_away(b)")
lgD = np.log(pair.pD/(1-pair.pD)).values
rD = fit_additive(lgD, "logit(pD) ~ ...")
lgA = np.log(pair.pA/(1-pair.pA)).values
rA = fit_additive(lgA, "logit(pA) ~ ...")
# bruit d'echantillonnage attendu sur la moyenne de paire (jitter/sqrt(n))
sd_j = 0.0089  # approx jitter pH
exp_noise = (sd_j/np.sqrt(pair.n) / (pair.pH*(1-pair.pH))).mean()
print(f"(bruit attendu sur logit du a jitter/sqrt(n) ~ {exp_noise:.4f})")
pair['residH'] = rH
print("\npaires les + mal expliquees par le modele additif:")
print(pair.reindex(pair.residH.abs().sort_values(ascending=False).index)[
    ['team_a', 'team_b', 'n', 'pH', 'residH']].head(8).round(4).to_string(index=False))

# symetrie aller/retour: pH(a,b) vs pA(b,a) en logit
m = pair.merge(pair, left_on=['team_a', 'team_b'], right_on=['team_b', 'team_a'],
               suffixes=('', '_rev'))
r_sym = stats.pearsonr(np.log(m.pH/(1-m.pH)), np.log(m.pA_rev/(1-m.pA_rev)))
diff = np.log(m.pH/(1-m.pH)) - np.log(m.pA_rev/(1-m.pA_rev))
print(f"\nsymetrie: corr[logit pH(a,b), logit pA(b,a)] r={r_sym[0]:.4f} ; "
      f"delta moyen={diff.mean():.3f} (avantage domicile en logit) sd={diff.std():.3f}")

# ---------------------------------------------------------------- c) GLM venue-specific vs partage
print("\n=== c) GLM POISSON: 80 params venue-specific vs 41 partages (LRT) ===")
from sklearn.linear_model import PoissonRegressor


def pois_ll(y, mu):
    return (y*np.log(mu) - mu - np.array([math.lgamma(v+1) for v in y])).sum()


# modele partage
Xs_, ys_ = [], []
for _, r in df.iterrows():
    xh = np.zeros(2*nT+1); xh[idx[r.team_a]] = 1; xh[nT+idx[r.team_b]] = 1; xh[-1] = 1
    Xs_.append(xh); ys_.append(r.score_a)
    xa = np.zeros(2*nT+1); xa[idx[r.team_b]] = 1; xa[nT+idx[r.team_a]] = 1
    Xs_.append(xa); ys_.append(r.score_b)
Xs_ = np.array(Xs_); ys_ = np.array(ys_, float)
m1 = PoissonRegressor(alpha=1e-8, max_iter=3000).fit(Xs_, ys_)
ll1 = pois_ll(ys_, m1.predict(Xs_))
# venue-specific: atk_h, def_h, atk_a, def_a
Xv, yv = [], []
for _, r in df.iterrows():
    xh = np.zeros(4*nT); xh[idx[r.team_a]] = 1; xh[3*nT+idx[r.team_b]] = 1  # atk_h(a) + def_a(b)
    Xv.append(xh); yv.append(r.score_a)
    xa = np.zeros(4*nT); xa[2*nT+idx[r.team_b]] = 1; xa[nT+idx[r.team_a]] = 1  # atk_a(b) + def_h(a)
    Xv.append(xa); yv.append(r.score_b)
Xv = np.array(Xv); yv = np.array(yv, float)
m2 = PoissonRegressor(alpha=1e-8, max_iter=3000).fit(Xv, yv)
ll2 = pois_ll(yv, m2.predict(Xv))
lr = 2*(ll2-ll1)
ddl = 4*nT - (2*nT+1)
print(f"LL partage={ll1:.1f} LL venue-spec={ll2:.1f} LR={lr:.1f} ddl={ddl} p={1-stats.chi2.cdf(lr, ddl):.4f}")

# ---------------------------------------------------------------- d) dispersion + nuls
print("\n=== d) SOUS-DISPERSION ET GENERATION DES NULS ===")
mu_all = m1.predict(Xs_)
muH = mu_all[0::2]; muA = mu_all[1::2]
# Pearson par decile de mu
dec = pd.qcut(mu_all, 10, duplicates='drop')
t = pd.DataFrame({'y': ys_, 'mu': mu_all, 'dec': dec})
disp = t.groupby('dec', observed=True).apply(
    lambda x: pd.Series({'mu_mean': x.mu.mean(), 'pearson': ((x.y-x.mu)**2/x.mu).mean(), 'n': len(x)}),
    include_groups=False)
print("dispersion de Pearson par decile de mu (1=Poisson):")
print(disp.round(3).to_string())
# P(nul) Poisson indep vs realise vs implicite
ks = np.arange(0, 11)
pdraw_pois = np.zeros(len(df))
for i in range(len(df)):
    ph = stats.poisson.pmf(ks, muH[i]); pa = stats.poisson.pmf(ks, muA[i])
    pdraw_pois[i] = (ph*pa).sum()
realized_d = (df.res == 'D').mean()
print(f"\nP(nul): Poisson-indep predit={pdraw_pois.mean():.4f} | realise={realized_d:.4f} | implicite cotes={df.pD.mean():.4f}")
nD = int((df.res == 'D').sum())
print(f"binomial realise vs Poisson-indep: p={stats.binomtest(nD, len(df), pdraw_pois.mean()).pvalue:.4f}")
# distribution des buts home vs mixture Poisson
print("\ndistribution buts HOME: realise vs Poisson(mu_i) mixture:")
for k in range(0, 8):
    obs = (df.score_a == k).mean()
    pred = stats.poisson.pmf(k, muH).mean()
    print(f"  {k}: obs={obs:.4f} pred={pred:.4f} (n_obs={int((df.score_a==k).sum())})")
obs_counts = [(df.score_a == k).sum() for k in range(7)] + [(df.score_a >= 7).sum()]
pred_p = [stats.poisson.pmf(k, muH).mean() for k in range(7)]
pred_p.append(1-sum(pred_p))
chi2 = sum((o-len(df)*p)**2/(len(df)*p) for o, p in zip(obs_counts, pred_p))
print(f"chi2 global (0..6,7+) = {chi2:.1f} ddl=7 p={1-stats.chi2.cdf(chi2,7):.4f}")
# correlation buts home/away (Dixon-Coles ?)
rho = stats.pearsonr(df.score_a - muH, df.score_b - muA)
print(f"\ncorr(residus buts home, residus buts away) r={rho[0]:+.4f} p={rho[1]:.3f}")
# table (0,0),(1,0),(0,1),(1,1) obs vs Poisson indep
print("cases basses obs vs pred Poisson-indep:")
for sa, sb in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    obs = ((df.score_a == sa) & (df.score_b == sb)).mean()
    pred = (stats.poisson.pmf(sa, muH)*stats.poisson.pmf(sb, muA)).mean()
    print(f"  {sa}-{sb}: obs={obs:.4f} pred={pred:.4f} ratio={obs/pred:.3f}")

print("\nDONE")
