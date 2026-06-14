# -*- coding: utf-8 -*-
"""
WF3 — FACETTE "RNG ET TEMPS"
Q1: Indépendance intra-round — nb victoires home / journée vs Poisson-binomiale devig
Q2: Total de buts par round — régulation (under/overdispersion vs indépendance) ?
Q3: Autocorrélation round N -> N+1 (% favoris gagnants, total buts)
Q4: Heure du jour / numéro de saison — variations ? + CUSUM dérive temporelle
Q5: Upsets (cote >= 5 gagnante) en grappes ?
Q6: Score miroir aller/retour même saison — corrélation ?
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

rng = np.random.default_rng(42)
eng = create_engine(load_settings().db_url)

# ================================================================
# STEP 0 — CHARGEMENT
# ================================================================
print("=" * 80)
print("STEP 0 — CHARGEMENT & PRÉPARATION")
print("=" * 80)

SQL = """
SELECT e.id AS event_id, e.round_info, e.team_a, e.team_b, e.expected_start,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
FROM events e
JOIN (SELECT event_id, MIN(id) AS first_snap FROM odds_snapshots GROUP BY event_id) f
     ON f.event_id = e.id
JOIN odds_snapshots o ON o.id = f.first_snap
LEFT JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0'
"""
with eng.connect() as c:
    df = pd.read_sql(text(SQL), c)

df = df.drop_duplicates(subset=['team_a', 'team_b', 'expected_start'], keep='first')
df['round'] = df['round_info'].astype(int)
df['ts'] = pd.to_datetime(df['expected_start'])
df = df.sort_values(['ts', 'event_id']).reset_index(drop=True)

# Saisons: le round repart en arrière => nouvelle saison
season_id, prev_r, seasons = 0, None, []
for r in df['round']:
    if prev_r is not None and r < prev_r:
        season_id += 1
    seasons.append(season_id)
    prev_r = r
df['season'] = seasons

# Devig 1X2
inv = 1.0 / df[['odds_home', 'odds_draw', 'odds_away']].values
booksum = inv.sum(axis=1)
df['p_home'] = inv[:, 0] / booksum
df['p_draw'] = inv[:, 1] / booksum
df['p_away'] = inv[:, 2] / booksum
df['overround'] = booksum

# Score exact -> distribution devig, mu/var total buts, E[diff], proba par score
def parse_cs(em_raw):
    if em_raw is None:
        return None
    em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    cs = em.get('Score exact')
    if not cs:
        return None
    items = []
    for k, v in cs.items():
        try:
            a, b = k.split('-')
            items.append((int(a), int(b), 1.0 / float(v)))
        except Exception:
            continue
    if not items:
        return None
    tot = sum(p for _, _, p in items)
    return {(a, b): p / tot for a, b, p in items}

cs_list = [parse_cs(x) for x in df['extra_markets']]
mus, vars_, ediffs, vdiffs = [], [], [], []
for d in cs_list:
    if d is None:
        mus.append(np.nan); vars_.append(np.nan); ediffs.append(np.nan); vdiffs.append(np.nan)
        continue
    t = np.array([a + b for (a, b) in d]); g = np.array([a - b for (a, b) in d])
    p = np.array(list(d.values()))
    mu = (t * p).sum(); mus.append(mu); vars_.append(((t - mu) ** 2 * p).sum())
    ed = (g * p).sum(); ediffs.append(ed); vdiffs.append(((g - ed) ** 2 * p).sum())
df['cs_dist'] = cs_list
df['mu_goals'] = mus
df['var_goals'] = vars_
df['e_diff'] = ediffs
df['v_diff'] = vdiffs

fin = df[df['score_a'].notna()].copy()
fin['total'] = fin['score_a'] + fin['score_b']
fin['diff'] = fin['score_a'] - fin['score_b']
fin['home_win'] = (fin['score_a'] > fin['score_b']).astype(int)
fin['draw'] = (fin['score_a'] == fin['score_b']).astype(int)
fin['away_win'] = (fin['score_a'] < fin['score_b']).astype(int)
print(f"Matchs finis dédupliqués: {len(fin)} | saisons: {df['season'].nunique()} "
      f"| avec Score exact: {fin['mu_goals'].notna().sum()}")
print(f"Calibration globale: P(home) devig={fin['p_home'].mean():.4f} vs obs={fin['home_win'].mean():.4f} ; "
      f"E[buts] CS={fin['mu_goals'].mean():.3f} vs obs={fin['total'].mean():.3f}")

# Rounds complets (10 matchs finis avec cotes)
grp = fin.groupby(['season', 'round'])
full_keys = [k for k, g in grp if len(g) == 10]
print(f"Instances (saison, round) complètes 10/10 avec résultats: {len(full_keys)}")
fullmask = fin.set_index(['season', 'round']).index.isin(full_keys)
fr = fin[fullmask].copy()

NSIM = 4000

# ================================================================
# Q1 — INDÉPENDANCE INTRA-ROUND : VICTOIRES HOME PAR JOURNÉE
# ================================================================
print("\n" + "=" * 80)
print("Q1 — NB VICTOIRES HOME PAR JOURNÉE (10 matchs) vs POISSON-BINOMIALE DEVIG")
print("=" * 80)

g1 = fr.groupby(['season', 'round'])
W_obs, E_w, V_w, P_rounds = [], [], [], []
for k, g in g1:
    p = g['p_home'].values
    W_obs.append(g['home_win'].sum()); E_w.append(p.sum()); V_w.append((p * (1 - p)).sum())
    P_rounds.append(p)
W_obs = np.array(W_obs, float); E_w = np.array(E_w); V_w = np.array(V_w)
Zw = (W_obs - E_w) / np.sqrt(V_w)
n_r = len(W_obs)
print(f"n rounds={n_r} | mean(W)={W_obs.mean():.3f} vs E={E_w.mean():.3f}")
print(f"Z home-wins: mean={Zw.mean():.4f}  var={Zw.var(ddof=1):.4f} (1.0 attendu si indépendant)")

# MC Poisson-binomiale: null de var(Z) et chi2 de la distribution de W
P_mat = np.vstack(P_rounds)                       # (n_r, 10)
U = rng.random((NSIM, n_r, 10))
W_sim = (U < P_mat[None, :, :]).sum(axis=2).astype(float)   # (NSIM, n_r)
Z_sim = (W_sim - E_w[None, :]) / np.sqrt(V_w)[None, :]
var_sim = Z_sim.var(axis=1, ddof=1)
mean_sim = Z_sim.mean(axis=1)
pv_var = 2 * min((var_sim >= Zw.var(ddof=1)).mean(), (var_sim <= Zw.var(ddof=1)).mean())
pv_mean = 2 * min((mean_sim >= Zw.mean()).mean(), (mean_sim <= Zw.mean()).mean())
print(f"MC ({NSIM} sims): p(var)={pv_var:.4f}  p(mean)={pv_mean:.4f}")

# Chi2 sur l'histogramme de W (0..10)
obs_hist = np.bincount(W_obs.astype(int), minlength=11)
exp_hist = np.array([ (W_sim == k).mean(axis=0).sum() for k in range(11) ])
keep = exp_hist >= 5
chi2 = ((obs_hist[keep] - exp_hist[keep]) ** 2 / exp_hist[keep]).sum()
chi2_sim = np.array([(((np.bincount(W_sim[s].astype(int), minlength=11)[keep] - exp_hist[keep]) ** 2
                       / exp_hist[keep]).sum()) for s in range(min(NSIM, 2000))])
pv_chi2 = (chi2_sim >= chi2).mean()
print(f"Histogramme W: chi2={chi2:.2f}  p(MC)={pv_chi2:.4f}")
print("W: ", dict(zip(range(11), obs_hist.tolist())), " attendu:", np.round(exp_hist, 1).tolist())

# ================================================================
# Q2 — TOTAL DE BUTS PAR ROUND : RÉGULATION ?
# ================================================================
print("\n" + "=" * 80)
print("Q2 — TOTAL BUTS PAR JOURNÉE vs SOMME INDÉPENDANTE DES DISTRIBUTIONS 'SCORE EXACT'")
print("=" * 80)

ok2 = [k for k, g in g1 if g['mu_goals'].notna().all()]
T_obs, E_t, V_t, dists = [], [], [], []
for k in ok2:
    g = g1.get_group(k)
    T_obs.append(g['total'].sum()); E_t.append(g['mu_goals'].sum()); V_t.append(g['var_goals'].sum())
    dists.append(list(g['cs_dist']))
T_obs = np.array(T_obs, float); E_t = np.array(E_t); V_t = np.array(V_t)
Zt = (T_obs - E_t) / np.sqrt(V_t)
print(f"n rounds={len(T_obs)} | mean(T)={T_obs.mean():.2f} vs E={E_t.mean():.2f} | sd(T)={T_obs.std(ddof=1):.3f}")
print(f"Z total-buts: mean={Zt.mean():.4f}  var={Zt.var(ddof=1):.4f}")

# MC: tire chaque match depuis sa distribution Score exact devig
def presample_totals(dist_list, nsim):
    out = np.zeros((nsim, len(dist_list)))
    for j, d in enumerate(dist_list):
        vals = np.array([a + b for (a, b) in d]); p = np.array(list(d.values()))
        out[:, j] = rng.choice(vals, size=nsim, p=p / p.sum())
    return out
T_sim = np.zeros((NSIM, len(ok2)))
for i, dl in enumerate(dists):
    T_sim[:, i] = presample_totals(dl, NSIM).sum(axis=1)
Zt_sim = (T_sim - E_t[None, :]) / np.sqrt(V_t)[None, :]
var_t_sim = Zt_sim.var(axis=1, ddof=1); mean_t_sim = Zt_sim.mean(axis=1)
pv_var_t = 2 * min((var_t_sim >= Zt.var(ddof=1)).mean(), (var_t_sim <= Zt.var(ddof=1)).mean())
pv_mean_t = 2 * min((mean_t_sim >= Zt.mean()).mean(), (mean_t_sim <= Zt.mean()).mean())
print(f"MC: p(var)={pv_var_t:.4f}  p(mean)={pv_mean_t:.4f}")
print(f"  -> var(Z) null MC: [{np.percentile(var_t_sim,2.5):.3f}, {np.percentile(var_t_sim,97.5):.3f}]")

# KS de Z_t vs distribution MC poolée
ks = stats.ks_2samp(Zt, Zt_sim[:200].ravel())
print(f"KS Z_obs vs Z_MC: D={ks.statistic:.4f} p={ks.pvalue:.4f}")

# Corrélation paire-à-paire intra-round (résidus de buts) via permutation
fr2 = fr[fr['mu_goals'].notna()].copy()
fr2['gres'] = (fr2['total'] - fr2['mu_goals']) / np.sqrt(fr2['var_goals'])
def mean_pair_corr(values_by_round):
    s = n = 0
    for v in values_by_round:
        if len(v) < 2: continue
        m = v.mean(); ss = ((v - m) ** 2).sum()
        tot = v.sum()
        s += (tot ** 2 - (v ** 2).sum()); n += len(v) * (len(v) - 1)
    return s / n
groups = [g['gres'].values for _, g in fr2.groupby(['season', 'round'])]
obs_pc = mean_pair_corr(groups)
all_res = fr2['gres'].values
perm_pc = []
sizes = [len(v) for v in groups]
for _ in range(2000):
    perm = rng.permutation(all_res)
    out, i0 = [], 0
    for s_ in sizes:
        out.append(perm[i0:i0 + s_]); i0 += s_
    perm_pc.append(mean_pair_corr(out))
perm_pc = np.array(perm_pc)
pv_pc = 2 * min((perm_pc >= obs_pc).mean(), (perm_pc <= obs_pc).mean())
print(f"E[res_i * res_j] intra-round (proxy covariance paire): obs={obs_pc:.5f}  "
      f"perm null=[{np.percentile(perm_pc,2.5):.5f},{np.percentile(perm_pc,97.5):.5f}]  p={pv_pc:.4f}")

# ================================================================
# Q3 — AUTOCORRÉLATION ROUND N -> N+1
# ================================================================
print("\n" + "=" * 80)
print("Q3 — AUTOCORRÉLATION ENTRE JOURNÉES CONSÉCUTIVES (même saison)")
print("=" * 80)

# Métriques par round (sur rounds complets) : résidu favoris, Z buts
fr['p_fav'] = fr[['p_home', 'p_draw', 'p_away']].max(axis=1)
fav_out = np.where(fr['p_home'] >= fr[['p_draw', 'p_away']].max(axis=1), fr['home_win'],
            np.where(fr['p_away'] >= fr['p_draw'], fr['away_win'], fr['draw']))
fr['fav_win'] = fav_out
met = fr.groupby(['season', 'round']).agg(
    fav_res=('fav_win', 'mean'), p_fav=('p_fav', 'mean'),
).reset_index()
met['fav_resid'] = met['fav_res'] - met['p_fav']
zt_map = {k: z for k, z in zip(ok2, Zt)}
met['z_goals'] = [zt_map.get((s, r), np.nan) for s, r in zip(met['season'], met['round'])]

def lag1(metric):
    x, y = [], []
    for s, g in met.sort_values(['season', 'round']).groupby('season'):
        v = g[metric].values; rr = g['round'].values
        for i in range(len(v) - 1):
            if rr[i + 1] == rr[i] + 1 and np.isfinite(v[i]) and np.isfinite(v[i + 1]):
                x.append(v[i]); y.append(v[i + 1])
    x, y = np.array(x), np.array(y)
    r, p = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    return len(x), r, p, rs, ps
for m in ['fav_resid', 'z_goals']:
    n, r, p, rs, ps = lag1(m)
    print(f"{m:10s}: n_paires={n}  Pearson r={r:+.4f} (p={p:.4f})  Spearman={rs:+.4f} (p={ps:.4f})")

# ================================================================
# Q4 — HEURE DU JOUR / NUMÉRO DE SAISON + CUSUM
# ================================================================
print("\n" + "=" * 80)
print("Q4 — HEURE DU JOUR & NUMÉRO DE SAISON")
print("=" * 80)

fin['hres'] = fin['home_win'] - fin['p_home']
fing = fin[fin['mu_goals'].notna()].copy()
fing['gres'] = fing['total'] - fing['mu_goals']
fin['hour'] = fin['ts'].dt.hour
fing['hour'] = fing['ts'].dt.hour

# Heure: chi2 victoires home obs vs attendu devig, par heure
tab = fin.groupby('hour').agg(n=('home_win', 'size'), obs=('home_win', 'sum'), exp=('p_home', 'sum'),
                              vr=('p_home', lambda p: (p * (1 - p)).sum()))
tab = tab[tab['n'] >= 100]
tab['z'] = (tab['obs'] - tab['exp']) / np.sqrt(tab['vr'])
chi2_h = (tab['z'] ** 2).sum(); dof = len(tab)
print(f"Home wins par heure: chi2={chi2_h:.2f} dof={dof} p={1 - stats.chi2.cdf(chi2_h, dof):.4f}")
print(tab[['n', 'obs', 'exp', 'z']].round(2).to_string())
# Buts par heure (ANOVA Kruskal sur résidus)
groups_h = [g['gres'].values for _, g in fing.groupby('hour') if len(g) >= 100]
kw = stats.kruskal(*groups_h)
print(f"Résidus buts par heure: Kruskal H={kw.statistic:.2f} p={kw.pvalue:.4f}")

# Saison: regroupe les saisons en déciles temporels (assez de n par groupe)
fin['sgrp'] = pd.qcut(fin['season'], 10, labels=False, duplicates='drop')
fing['sgrp'] = pd.qcut(fing['season'], 10, labels=False, duplicates='drop')
tabs = fin.groupby('sgrp').agg(n=('home_win', 'size'), obs=('home_win', 'sum'), exp=('p_home', 'sum'),
                               vr=('p_home', lambda p: (p * (1 - p)).sum()))
tabs['z'] = (tabs['obs'] - tabs['exp']) / np.sqrt(tabs['vr'])
chi2_s = (tabs['z'] ** 2).sum()
print(f"\nHome wins par décile de saison: chi2={chi2_s:.2f} dof={len(tabs)} "
      f"p={1 - stats.chi2.cdf(chi2_s, len(tabs)):.4f}")
groups_s = [g['gres'].values for _, g in fing.groupby('sgrp')]
kw2 = stats.kruskal(*groups_s)
print(f"Résidus buts par décile de saison: Kruskal H={kw2.statistic:.2f} p={kw2.pvalue:.4f}")
print(f"E[buts] obs par décile: {fing.groupby('sgrp')['total'].mean().round(3).tolist()}")
print(f"E[buts] CS  par décile: {fing.groupby('sgrp')['mu_goals'].mean().round(3).tolist()}")

# CUSUM sur résidus dans l'ordre temporel (dérive du RNG ?)
for name, ser, sd in [('home_resid', fin.sort_values('ts')['hres'].values, None),
                      ('goals_resid', fing.sort_values('ts')['gres'].values, None)]:
    x = ser - ser.mean()
    s = np.abs(np.cumsum(x)).max() / (np.std(ser, ddof=1) * np.sqrt(len(ser)))
    mx = []
    for _ in range(1000):
        xp = rng.permutation(ser) - ser.mean()
        mx.append(np.abs(np.cumsum(xp)).max() / (np.std(ser, ddof=1) * np.sqrt(len(ser))))
    pv = (np.array(mx) >= s).mean()
    print(f"CUSUM {name}: stat={s:.3f}  p(perm)={pv:.4f}")

# ================================================================
# Q5 — UPSETS (COTE >= 5 GAGNANTE) EN GRAPPES ?
# ================================================================
print("\n" + "=" * 80)
print("Q5 — UPSETS (cote ouverture >= 5 qui gagne)")
print("=" * 80)

ODDS_TH = 5.0
def upset_info(row):
    p_up, won = 0.0, 0
    for oc, pc, res in [('odds_home', 'p_home', 'home_win'), ('odds_draw', 'p_draw', 'draw'),
                        ('odds_away', 'p_away', 'away_win')]:
        if row[oc] >= ODDS_TH:
            p_up += row[pc]
            if row[res] == 1:
                won = 1
    return p_up, won
ui = fin.apply(upset_info, axis=1, result_type='expand')
fin['p_upset'], fin['upset'] = ui[0], ui[1]
print(f"Matchs avec au moins un outcome cote>={ODDS_TH}: {(fin['p_upset'] > 0).sum()} "
      f"| upsets observés: {fin['upset'].sum()} vs attendus: {fin['p_upset'].sum():.1f}")

frU = fin[fullmask]
gU = frU.groupby(['season', 'round'])
U_obs = gU['upset'].sum().values.astype(float)
E_u = gU['p_upset'].sum().values
V_u = gU['p_upset'].apply(lambda p: (p * (1 - p)).sum()).values
keep = V_u > 0
Zu = (U_obs[keep] - E_u[keep]) / np.sqrt(V_u[keep])
print(f"Rounds avec exposition upset: {keep.sum()} | Z upsets: mean={Zu.mean():.4f} var={Zu.var(ddof=1):.4f}")
# MC
P_up = [g['p_upset'].values for _, g in gU]
P_up = np.vstack([p for p, k in zip(P_up, keep) if k])
Uu = rng.random((NSIM, P_up.shape[0], P_up.shape[1]))
U_sim = (Uu < P_up[None]).sum(axis=2).astype(float)
Zu_sim = (U_sim - E_u[keep][None]) / np.sqrt(V_u[keep])[None]
vu_sim = Zu_sim.var(axis=1, ddof=1)
pv_vu = 2 * min((vu_sim >= Zu.var(ddof=1)).mean(), (vu_sim <= Zu.var(ddof=1)).mean())
print(f"MC dispersion upsets/round: p={pv_vu:.4f}")

# Distribution du nb d'upsets par round vs MC
ohist = np.bincount(U_obs[keep].astype(int), minlength=8)[:8]
ehist = np.array([(U_sim == k).mean(axis=0).sum() for k in range(8)])
print("Upsets/round obs:", ohist.tolist(), " attendu MC:", np.round(ehist, 1).tolist())

# Grappes temporelles: autocorr lag-1 des résidus d'upsets entre rounds consécutifs
metU = gU.agg(u=('upset', 'sum'), e=('p_upset', 'sum')).reset_index()
metU['resid'] = metU['u'] - metU['e']
x, y = [], []
for s, g in metU.sort_values(['season', 'round']).groupby('season'):
    v = g['resid'].values; rr = g['round'].values
    for i in range(len(v) - 1):
        if rr[i + 1] == rr[i] + 1:
            x.append(v[i]); y.append(v[i + 1])
r5, p5 = stats.pearsonr(x, y)
print(f"Autocorr lag-1 résidus upsets (n={len(x)}): r={r5:+.4f} p={p5:.4f}")
# Conditionnel: P(upset | upset au round précédent même équipe?) -> niveau match: après un upset d'une équipe
fin_s = fin.sort_values(['season', 'round'])
ups_prev = []
for team_col in ['team_a', 'team_b']:
    pass  # couvert par autocorr round-level; granularité équipe traitée en Q6/standings ailleurs

# ================================================================
# Q6 — SCORE MIROIR ALLER/RETOUR MÊME SAISON
# ================================================================
print("\n" + "=" * 80)
print("Q6 — ALLER/RETOUR MÊME SAISON : INDÉPENDANCE DES DEUX LEGS ?")
print("=" * 80)

pairs = []
for s, g in fin.groupby('season'):
    seen = {}
    for _, row in g.iterrows():
        key = frozenset([row['team_a'], row['team_b']])
        if key in seen:
            r1 = seen[key]
            if r1['team_a'] == row['team_b']:   # vrai aller/retour
                pairs.append((r1, row))
        else:
            seen[key] = row
print(f"Paires aller/retour trouvées: {len(pairs)}")

# (a) Corrélation des résidus de différence de buts (perspective équipe A du leg 1)
e1, e2, w = [], [], []
for r1, r2 in pairs:
    if np.isfinite(r1['e_diff']) and np.isfinite(r2['e_diff']):
        d1 = (r1['diff'] - r1['e_diff']) / math.sqrt(r1['v_diff'])
        d2 = (-(r2['diff']) - (-(r2['e_diff']))) / math.sqrt(r2['v_diff'])  # du point de vue équipe A leg1
        e1.append(d1); e2.append(d2)
r6, p6 = stats.pearsonr(e1, e2)
rs6, ps6 = stats.spearmanr(e1, e2)
print(f"(a) Corr résidus diff-buts leg1 vs leg2 (n={len(e1)}): Pearson r={r6:+.4f} p={p6:.4f} | "
      f"Spearman {rs6:+.4f} p={ps6:.4f}")

# (b) Table 3x3 résultat leg1 x leg2 (perspective équipe A leg1) vs attendu devig indépendant
def res_persp(row, team):
    if row['team_a'] == team:
        return 0 if row['home_win'] else (1 if row['draw'] else 2)
    return 0 if row['away_win'] else (1 if row['draw'] else 2)
def probs_persp(row, team):
    if row['team_a'] == team:
        return np.array([row['p_home'], row['p_draw'], row['p_away']])
    return np.array([row['p_away'], row['p_draw'], row['p_home']])
obs33 = np.zeros((3, 3)); exp33 = np.zeros((3, 3))
P1, P2 = [], []
for r1, r2 in pairs:
    A = r1['team_a']
    o1, o2 = res_persp(r1, A), res_persp(r2, A)
    p1, p2 = probs_persp(r1, A), probs_persp(r2, A)
    obs33[o1, o2] += 1; exp33 += np.outer(p1, p2)
    P1.append(p1); P2.append(p2)
chi2_33 = ((obs33 - exp33) ** 2 / exp33).sum()
# Null MC (les attendus varient par paire)
P1 = np.vstack(P1); P2 = np.vstack(P2)
c1 = P1.cumsum(axis=1); c2 = P2.cumsum(axis=1)
chis = []
for _ in range(2000):
    u1 = rng.random(len(P1)); u2 = rng.random(len(P2))
    s1 = (u1[:, None] > c1).sum(axis=1); s2 = (u2[:, None] > c2).sum(axis=1)
    t = np.zeros((3, 3))
    np.add.at(t, (s1, s2), 1)
    chis.append(((t - exp33) ** 2 / exp33).sum())
pv33 = (np.array(chis) >= chi2_33).mean()
print(f"(b) Table 3x3 leg1 x leg2: chi2={chi2_33:.2f}  p(MC)={pv33:.4f}")
print("   obs:\n", obs33.astype(int))
print("   exp:\n", np.round(exp33, 1))

# (c) Score miroir exact: P(score2 == miroir(score1)) vs attendu indépendant (Score exact devig)
n_mir, e_mir, var_mir, n_used = 0, 0.0, 0.0, 0
for r1, r2 in pairs:
    d1, d2 = r1['cs_dist'], r2['cs_dist']
    if d1 is None or d2 is None:
        continue
    n_used += 1
    sc1 = (int(r1['score_a']), int(r1['score_b']))
    sc2 = (int(r2['score_a']), int(r2['score_b']))
    if sc2 == (sc1[1], sc1[0]):
        n_mir += 1
    pm = sum(p1 * d2.get((b, a), 0.0) for (a, b), p1 in d1.items())
    e_mir += pm; var_mir += pm * (1 - pm)
z_mir = (n_mir - e_mir) / math.sqrt(var_mir)
pv_mir = 2 * (1 - stats.norm.cdf(abs(z_mir)))
print(f"(c) Scores miroir exacts: obs={n_mir} attendu={e_mir:.1f} (n={n_used}) z={z_mir:+.2f} p={pv_mir:.4f}")

# (d) Même vainqueur (persp. A) les deux legs au-delà des cotes ?
same = sum(1 for r1, r2 in pairs if res_persp(r1, r1['team_a']) == res_persp(r2, r1['team_a']))
e_same = sum((probs_persp(r1, r1['team_a']) * probs_persp(r2, r1['team_a'])).sum() for r1, r2 in pairs)
v_same = sum((lambda p: p * (1 - p))((probs_persp(r1, r1['team_a']) * probs_persp(r2, r1['team_a'])).sum())
             for r1, r2 in pairs)
z_same = (same - e_same) / math.sqrt(v_same)
print(f"(d) Même résultat aller/retour: obs={same} attendu={e_same:.1f} z={z_same:+.2f} "
      f"p={2 * (1 - stats.norm.cdf(abs(z_same))):.4f}")

print("\nFIN _wf3_rng.py")
