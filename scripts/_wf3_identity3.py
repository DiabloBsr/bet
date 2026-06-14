# -*- coding: utf-8 -*-
"""
WF3 IDENTITY - PART 3 : le flow complet du generateur
1) cap des scores: total <= 6 ? distribution realisee vs marche Score exact
2) le marche Score exact est-il un Poisson independant tronque ? (structure log-lineaire)
3) les cotes 1X2 sont-elles la somme des probas Score exact ? (coherence inter-marches)
4) calibration cell-par-cell du Score exact (le score est-il TIRE de cette table ?)
5) jitter: test OOS cosmetique vs informatif (ROI favorable vs defavorable)
"""
import sys, json, math
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
        SELECT e.id, e.team_a, e.team_b, e.expected_start, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE e.round_info != '0' AND r.score_a IS NOT NULL
    """), c)
    od = pd.read_sql(text("""
        SELECT id, event_id, odds_home, odds_draw, odds_away, extra_markets
        FROM odds_snapshots WHERE odds_home IS NOT NULL ORDER BY id
    """), c)

od_open = od.groupby('event_id', as_index=False).first()
df = ev.merge(od_open[['event_id', 'odds_home', 'odds_draw', 'odds_away', 'extra_markets']],
              left_on='id', right_on='event_id')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').drop_duplicates(
    subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['pH'] = (1/df.odds_home)/inv
df['pD'] = (1/df.odds_draw)/inv
df['pA'] = (1/df.odds_away)/inv
df['res'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))
df['tot'] = df.score_a + df.score_b
print(f"n={len(df)}")

# ---------------------------------------------------------------- 1) cap
print("\n=== 1) CAP DES SCORES ===")
print(f"max score_a={df.score_a.max()} max score_b={df.score_b.max()} max total={df.tot.max()}")
print("distribution du total de buts:")
print(df.tot.value_counts().sort_index().to_string())
n6 = (df.tot == 6).sum()
print(f"matchs a 7+ buts: {(df.tot > 6).sum()} / {len(df)} ; matchs a exactement 6: {n6}")

# ---------------------------------------------------------------- parse Score exact
cells = [(i, j) for i in range(7) for j in range(7) if i+j <= 6]
print(f"\ncellules attendues (i+j<=6): {len(cells)}")


def parse_cs(s):
    em = json.loads(s) if isinstance(s, str) else s
    if not em or 'Score exact' not in em:
        return None
    return em['Score exact']


cs_list = []
keep_idx = []
for i, r in df.iterrows():
    cs = parse_cs(r.extra_markets)
    if cs is None:
        continue
    q = {}
    for k, v in cs.items():
        try:
            a, b = k.split('-')
            q[(int(a), int(b))] = 1.0/float(v)
        except Exception:
            pass
    if len(q) < 20:
        continue
    cs_list.append(q)
    keep_idx.append(i)
print(f"events avec marche Score exact parse: {len(cs_list)}")
dfc = df.loc[keep_idx].reset_index(drop=True)

# matrice implied (events x 28), avec flag cap (odds==100 -> 0.01)
Q = np.zeros((len(cs_list), len(cells)))
CAP = np.zeros_like(Q, dtype=bool)
for n, q in enumerate(cs_list):
    for ci, c_ in enumerate(cells):
        v = q.get(c_, np.nan)
        Q[n, ci] = v if not np.isnan(v) else 0.0
        CAP[n, ci] = (abs(v - 0.01) < 1e-9)
overround_cs = Q.sum(axis=1)
print(f"somme 1/odds Score exact: mean={overround_cs.mean():.4f} std={overround_cs.std():.4f}")
Qn = Q / Q.sum(axis=1, keepdims=True)  # normalise

# ---------------------------------------------------------------- 3) coherence 1X2 vs Score exact
print("\n=== 3) COHERENCE 1X2 <- SCORE EXACT ===")
iH = [ci for ci, (a, b) in enumerate(cells) if a > b]
iD = [ci for ci, (a, b) in enumerate(cells) if a == b]
iA = [ci for ci, (a, b) in enumerate(cells) if a < b]
pH_cs = Qn[:, iH].sum(axis=1)
pD_cs = Qn[:, iD].sum(axis=1)
pA_cs = Qn[:, iA].sum(axis=1)
for nm, a, b in [('pH', pH_cs, dfc.pH.values), ('pD', pD_cs, dfc.pD.values), ('pA', pA_cs, dfc.pA.values)]:
    d = a - b
    print(f"{nm}: corr={np.corrcoef(a,b)[0,1]:.5f} mean|diff|={np.abs(d).mean():.5f} max|diff|={np.abs(d).max():.4f} bias={d.mean():+.5f}")

# ---------------------------------------------------------------- 2) structure log-lineaire du Score exact
print("\n=== 2) SCORE EXACT = POISSON INDEPENDANT TRONQUE ? ===")
# buckets de force pour moyenner les tables
dfc['bucket'] = pd.qcut(dfc.pH, 5, labels=False)
for bk in range(5):
    sel = (dfc.bucket == bk).values
    qm = Qn[sel].mean(axis=0)
    capped_frac = CAP[sel].mean(axis=0)
    use = capped_frac < 0.5  # exclure cellules majoritairement cappees
    # fit log q_ij = a_i + b_j sur cellules valides
    A = np.zeros((use.sum(), 14))
    yv = []
    rix = 0
    rows_used = []
    for ci, (i_, j_) in enumerate(cells):
        if not use[ci]:
            continue
        A[rix, i_] = 1; A[rix, 7+j_] = 1
        yv.append(math.log(qm[ci])); rows_used.append((i_, j_)); rix += 1
    keep = [c for c in range(14) if A[:, c].any()]
    keep = keep[:-1]  # drop one for identifiability
    sol, *_ = np.linalg.lstsq(A[:, keep], np.array(yv), rcond=None)
    pred = A[:, keep] @ sol
    resid = np.array(yv) - pred
    ss = 1 - (resid**2).sum()/((np.array(yv)-np.mean(yv))**2).sum()
    worst = sorted(zip(rows_used, resid), key=lambda x: -abs(x[1]))[:3]
    print(f"bucket pH~{dfc[dfc.bucket==bk].pH.mean():.2f}: cellules={use.sum()} R2(log q ~ a_i+b_j)={ss:.4f} "
          f"sd_resid={resid.std():.3f} worst={[(w[0], round(w[1],3)) for w in worst]}")
    # forme Poisson des marges: a_i ~ i*log(mu) - log(i!)
    ai = np.zeros(7); bi = np.zeros(7)
    full_sol = np.zeros(14); full_sol[keep] = sol
    ai[:7] = full_sol[:7]; bi[:7] = full_sol[7:]
    # test: a_i + log(i!) lineaire en i ?
    ii = np.arange(7)
    av = ai + np.array([math.lgamma(k+1) for k in ii])
    av_valid = ii[np.abs(ai) > 1e-12] if (np.abs(ai) > 1e-12).sum() >= 3 else ii
    sl = stats.linregress(ii[:6], av[:6])
    print(f"   marge home: a_i+log(i!) vs i -> r2={sl.rvalue**2:.4f} (1=Poisson exact), mu_h={math.exp(sl.slope):.2f}")

# ---------------------------------------------------------------- 4) calibration cell-par-cell
print("\n=== 4) CALIBRATION SCORE EXACT: realise vs implicite (cell par cell) ===")
sa = dfc.score_a.values; sb = dfc.score_b.values
rows = []
chi2_tot, nfree = 0.0, 0
for ci, (i_, j_) in enumerate(cells):
    obs = ((sa == i_) & (sb == j_)).sum()
    imp = Qn[:, ci].sum()  # somme des probas = nb attendu
    f_obs = obs/len(dfc); f_imp = imp/len(dfc)
    if imp > 5:
        chi2_tot += (obs-imp)**2/imp
        nfree += 1
    pv = stats.binomtest(int(obs), len(dfc), min(f_imp, 1)).pvalue if f_imp > 0 else np.nan
    rows.append((f"{i_}-{j_}", obs, round(imp, 1), f_obs, f_imp, pv, CAP[:, ci].mean()))
cal = pd.DataFrame(rows, columns=['cell', 'obs', 'exp_implied', 'f_obs', 'f_imp', 'p_binom', 'frac_capped'])
print(cal.round(4).to_string(index=False))
print(f"chi2 global={chi2_tot:.1f} ddl~{nfree-1} p={1-stats.chi2.cdf(chi2_tot, nfree-1):.4f}")
print(f"cellules p_binom<0.01: {(cal.p_binom < 0.01).sum()}/{len(cal)}")

# correlation realisee home/away vs predite par la table implicite moyenne
print("\ncorrelation buts H/A: realisee vs predite par les tables Score exact:")
r_real = stats.pearsonr(sa, sb)
# correlation predite: E[corr] sous la table moyenne par event -> simulons depuis Qn
rng = np.random.default_rng(0)
sim_a, sim_b = [], []
for n in range(len(dfc)):
    ci = rng.choice(len(cells), p=Qn[n])
    sim_a.append(cells[ci][0]); sim_b.append(cells[ci][1])
r_sim = stats.pearsonr(sim_a, sim_b)
print(f"corr(sa,sb) realise={r_real[0]:+.4f} ; simule depuis tables implicites={r_sim[0]:+.4f}")

# ---------------------------------------------------------------- 5) jitter OOS: cosmetique vs informatif
print("\n=== 5) JITTER: TEST OOS (base de paire estimee sur train 70%) ===")
df_s = df.sort_values('expected_start').reset_index(drop=True)
n_tr = int(0.7*len(df_s))
train, oos = df_s.iloc[:n_tr], df_s.iloc[n_tr:]
base = train.groupby(['team_a', 'team_b'])[['pH', 'pD', 'pA']].mean()
oo = oos.merge(base, left_on=['team_a', 'team_b'], right_index=True, suffixes=('', '_base'))
oo['jitH'] = oo.pH - oo.pH_base
oo['jitA'] = oo.pA - oo.pA_base
print(f"oos avec base: {len(oo)} ; sd jitter OOS vs base train: {oo.jitH.std():.4f}")
res_rows = []
for side, jcol, ocol, out in [('H', 'jitH', 'odds_home', 'H'), ('A', 'jitA', 'odds_away', 'A')]:
    fav = oo[oo[jcol] < -0.005]   # cote meilleure que la base => si cosmetique, EV ameliore
    unf = oo[oo[jcol] > 0.005]
    for nm, g in [('favorable', fav), ('defavorable', unf)]:
        if len(g) == 0:
            continue
        pnl = np.where(g.res == out, g[ocol]-1, -1)
        res_rows.append((side, nm, len(g), pnl.mean()*100, (g.res == out).mean()*100, g[ocol].mean()))
rr = pd.DataFrame(res_rows, columns=['side', 'jitter', 'n', 'ROI%', 'WR%', 'avg_odds'])
print(rr.round(2).to_string(index=False))
# difference de ROI attendue si cosmetique: ~2*|jit|/p ~ en %
expect = 2*0.0078/oo.pH_base.mean()*100
print(f"(si cosmetique, delta ROI attendu favorable-defavorable ~ {expect:.1f}% par cote)")

# beta plus puissant: goal-diff demeane vs jitter (toute la base)
df_s['gd'] = df_s.score_a - df_s.score_b
df_s['gd_dm'] = df_s.gd - df_s.groupby(['team_a', 'team_b']).gd.transform('mean')
df_s['pH_dm'] = df_s.pH - df_s.groupby(['team_a', 'team_b']).pH.transform('mean')
x = df_s.pH_dm.values; y = df_s.gd_dm.values
beta_gd = (x*y).sum()/(x*x).sum()
se_gd = math.sqrt(((y-beta_gd*x)**2).mean()/(x*x).sum())
# slope cross-pair attendu si informatif
pairm = df_s.groupby(['team_a', 'team_b'])[['pH', 'gd']].mean()
sl_cross = stats.linregress(pairm.pH, pairm.gd)
print(f"\nbeta(GD ~ jitter pH) = {beta_gd:+.2f} +- {se_gd:.2f} ; attendu si informatif ~ {sl_cross.slope:.2f}, si cosmetique 0")
print(f"t(beta=0)={beta_gd/se_gd:+.2f} ; t(beta=informatif)={(beta_gd-sl_cross.slope)/se_gd:+.2f}")

print("\nDONE")
