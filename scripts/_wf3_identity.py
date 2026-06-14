# -*- coding: utf-8 -*-
"""
WF3 - FACETTE: IDENTITES D'EQUIPES
Le moteur a-t-il exactement 20 profils FIXES ?
1. Profils attack/defense home/away + WR avec IC
2. Stationnarite (split moitie + CUSUM)
3. Mapping paire ordonnee -> cotes (deterministe ?)
4. Edge exploitable pair-level (walk-forward 70/30)
5. Hierarchie / tiers
"""
import sys, json, math
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 200)

eng = create_engine(load_settings().db_url)

# ---------------------------------------------------------------- load
with eng.connect() as c:
    ev = pd.read_sql(text("""
        SELECT e.id, e.team_a, e.team_b, e.round_info, e.expected_start,
               r.score_a, r.score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        WHERE e.round_info != '0'
          AND r.score_a IS NOT NULL AND r.score_b IS NOT NULL
    """), c)
    od = pd.read_sql(text("""
        SELECT id, event_id, odds_home, odds_draw, odds_away
        FROM odds_snapshots
        WHERE odds_home IS NOT NULL AND odds_draw IS NOT NULL AND odds_away IS NOT NULL
        ORDER BY id
    """), c)

# opening odds = MIN(id) per event
od_open = od.groupby('event_id', as_index=False).first()
df = ev.merge(od_open[['event_id', 'odds_home', 'odds_draw', 'odds_away']],
              left_on='id', right_on='event_id', how='inner')
df['expected_start'] = pd.to_datetime(df['expected_start'])
df = df.sort_values('expected_start').drop_duplicates(
    subset=['team_a', 'team_b', 'expected_start'], keep='first').reset_index(drop=True)
df['round'] = df['round_info'].astype(int)
df['res'] = np.where(df.score_a > df.score_b, 'H', np.where(df.score_a < df.score_b, 'A', 'D'))

# implied probs (margin-normalized)
inv = 1/df.odds_home + 1/df.odds_draw + 1/df.odds_away
df['margin'] = inv - 1
df['pH'] = (1/df.odds_home)/inv
df['pD'] = (1/df.odds_draw)/inv
df['pA'] = (1/df.odds_away)/inv

print(f"=== DATASET: {len(df)} matchs dedupliques avec resultat + cote d'ouverture ===")
print(f"periode: {df.expected_start.min()} -> {df.expected_start.max()}")
print(f"marge bookmaker: mean={df.margin.mean()*100:.2f}% std={df.margin.std()*100:.2f}%")
teams = sorted(set(df.team_a) | set(df.team_b))
print(f"{len(teams)} equipes\n")

# ---------------------------------------------------------------- 0. calibration sanity
print("=== 0. SANITY: calibration globale 1X2 (cote d'ouverture) ===")
rows = []
for out, p in [('H', 'pH'), ('D', 'pD'), ('A', 'pA')]:
    y = (df.res == out).astype(int)
    rows.append((out, df[p].mean(), y.mean(), len(df)))
cal = pd.DataFrame(rows, columns=['outcome', 'implied_mean', 'freq', 'n'])
print(cal.to_string(index=False))
# binned calibration on pH
bins = pd.qcut(df.pH, 10, duplicates='drop')
g = df.groupby(bins, observed=True).apply(
    lambda x: pd.Series({'imp': x.pH.mean(), 'freq': (x.res == 'H').mean(), 'n': len(x)}),
    include_groups=False)
chi2_cal = (((g.freq - g.imp)**2 * g.n) / (g.imp*(1-g.imp))).sum()
print(f"chi2 calibration pH (10 bins) = {chi2_cal:.1f}, ddl=10, p={1-stats.chi2.cdf(chi2_cal,10):.4f}\n")

# ---------------------------------------------------------------- 1. profils equipes
print("=== 1. PROFILS PAR EQUIPE (toute la BDD, IC 95%) ===")


def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k/n
    z = 1.96
    d = 1 + z*z/n
    ctr = (p + z*z/(2*n))/d
    hw = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return p, ctr-hw, ctr+hw


prof = []
for t in teams:
    h = df[df.team_a == t]
    a = df[df.team_b == t]
    wrh, lo_h, hi_h = wilson((h.res == 'H').sum(), len(h))
    wra, lo_a, hi_a = wilson((a.res == 'A').sum(), len(a))
    ppg = ((h.res == 'H').sum()*3 + (h.res == 'D').sum() + (a.res == 'A').sum()*3 + (a.res == 'D').sum()) / (len(h)+len(a))
    prof.append({
        'team': t, 'nH': len(h), 'nA': len(a),
        'GF_h': h.score_a.mean(), 'GA_h': h.score_b.mean(),
        'GF_a': a.score_b.mean(), 'GA_a': a.score_a.mean(),
        'WR_h': wrh, 'WR_h_lo': lo_h, 'WR_h_hi': hi_h,
        'WR_a': wra, 'WR_a_lo': lo_a, 'WR_a_hi': hi_a,
        'PPG': ppg,
        'pH_imp': h.pH.mean(), 'pA_imp': a.pA.mean(),
    })
prof = pd.DataFrame(prof).sort_values('PPG', ascending=False).reset_index(drop=True)
print(prof.round(3).to_string(index=False))

# implied vs realized strength
r_imp = stats.pearsonr(prof.pH_imp, prof.WR_h)
print(f"\ncorr(pH implicite moyenne, WR home realise) r={r_imp[0]:.3f} p={r_imp[1]:.2e}")
r2 = stats.pearsonr(prof.pH_imp + prof.pA_imp, prof.PPG)
print(f"corr(force implicite h+a, PPG realise)      r={r2[0]:.3f} p={r2[1]:.2e}\n")

# ---------------------------------------------------------------- 1b. Poisson attack/defense GLM
print("=== 1b. MODELE POISSON attack/defense (GLM, toute la BDD) ===")
try:
    from sklearn.linear_model import PoissonRegressor
    idx = {t: i for i, t in enumerate(teams)}
    nT = len(teams)
    X, y = [], []
    for _, r in df.iterrows():
        xh = np.zeros(2*nT+1); xh[idx[r.team_a]] = 1; xh[nT+idx[r.team_b]] = 1; xh[-1] = 1
        X.append(xh); y.append(r.score_a)
        xa = np.zeros(2*nT+1); xa[idx[r.team_b]] = 1; xa[nT+idx[r.team_a]] = 1
        X.append(xa); y.append(r.score_b)
    X = np.array(X); y = np.array(y)
    glm = PoissonRegressor(alpha=1e-6, max_iter=2000).fit(X, y)
    atk = glm.coef_[:nT]; dfn = glm.coef_[nT:2*nT]; hadv = glm.coef_[-1]
    atk -= atk.mean(); dfn -= dfn.mean()
    pr = pd.DataFrame({'team': teams, 'attack': atk, 'defense': dfn}).sort_values('attack', ascending=False)
    print(pr.round(3).to_string(index=False))
    print(f"home_advantage (log) = {hadv:.4f} -> facteur x{math.exp(hadv):.3f}")
    print(f"intercept = {glm.intercept_:.4f} -> mu base = {math.exp(glm.intercept_):.3f}")
    mu = glm.predict(X)
    print(f"deviance ratio: var(y)/mean(y) global = {y.var()/y.mean():.3f} (1=Poisson pur)")
    # overdispersion conditionnelle: Pearson stat
    pearson = ((y-mu)**2/mu).sum()/(len(y)-X.shape[1])
    print(f"Pearson dispersion (cond. au modele) = {pearson:.3f} (1=Poisson)\n")
except Exception as e:
    print("GLM fail:", e)

# ---------------------------------------------------------------- 2. stationnarite
print("=== 2. STATIONNARITE DES PROFILS ===")
cut = df.expected_start.quantile(0.5)
half = np.where(df.expected_start <= cut, 1, 2)
df['half'] = half

print("--- 2a. split 1ere/2eme moitie : WR (points) et GF par equipe ---")
sig_wr, sig_gf = 0, 0
rows = []
for t in teams:
    m = df[(df.team_a == t) | (df.team_b == t)].copy()
    m['win'] = ((m.team_a == t) & (m.res == 'H')) | ((m.team_b == t) & (m.res == 'A'))
    m['gf'] = np.where(m.team_a == t, m.score_a, m.score_b)
    w1, n1 = m[m.half == 1].win.sum(), (m.half == 1).sum()
    w2, n2 = m[m.half == 2].win.sum(), (m.half == 2).sum()
    # two-prop z (chi2 2x2 exact-ish)
    tab = np.array([[w1, n1-w1], [w2, n2-w2]])
    p_wr = stats.chi2_contingency(tab, correction=True)[1] if tab.min() >= 0 else np.nan
    g1, g2 = m[m.half == 1].gf.sum(), m[m.half == 2].gf.sum()
    # conditional binomial test for Poisson rates
    p_gf = stats.binomtest(int(g1), int(g1+g2), n1/(n1+n2)).pvalue
    sig_wr += p_wr < 0.05; sig_gf += p_gf < 0.05
    rows.append((t, n1, n2, w1/n1, w2/n2, p_wr, g1/n1, g2/n2, p_gf))
st = pd.DataFrame(rows, columns=['team', 'n1', 'n2', 'WR1', 'WR2', 'p_WR', 'GF1', 'GF2', 'p_GF'])
print(st.round(3).to_string(index=False))
print(f"equipes avec p_WR<0.05 : {sig_wr}/20 (attendu ~1 sous H0)")
print(f"equipes avec p_GF<0.05 : {sig_gf}/20 (attendu ~1 sous H0)")

print("\n--- 2b. CUSUM sur la sequence chronologique de wins par equipe ---")


def cusum_pval(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 30 or x.std() == 0:
        return np.nan, np.nan
    s = np.cumsum(x - x.mean())
    stat = np.abs(s).max()/(x.std()*math.sqrt(n))
    # P(sup|B0(t)|>x) Kolmogorov
    p = 2*sum((-1)**(k-1)*math.exp(-2*k*k*stat*stat) for k in range(1, 101))
    return stat, min(max(p, 0), 1)


cus = []
for t in teams:
    m = df[(df.team_a == t) | (df.team_b == t)].sort_values('expected_start')
    win = (((m.team_a == t) & (m.res == 'H')) | ((m.team_b == t) & (m.res == 'A'))).astype(int).values
    gf = np.where(m.team_a == t, m.score_a, m.score_b)
    s1, p1 = cusum_pval(win)
    s2, p2 = cusum_pval(gf)
    cus.append((t, len(win), s1, p1, s2, p2))
cu = pd.DataFrame(cus, columns=['team', 'n', 'cusum_WR', 'p_WR', 'cusum_GF', 'p_GF'])
print(cu.round(4).to_string(index=False))
print(f"CUSUM WR p<0.05: {(cu.p_WR < 0.05).sum()}/20 ; CUSUM GF p<0.05: {(cu.p_GF < 0.05).sum()}/20")

# stationnarite des COTES elles-memes par paire
print("\n--- 2c. stationnarite des cotes: pH d'une meme paire, moitie 1 vs 2 ---")
pair_stab = []
for (a, b), g in df.groupby(['team_a', 'team_b']):
    if (g.half == 1).sum() >= 3 and (g.half == 2).sum() >= 3:
        d = g[g.half == 1].pH.mean() - g[g.half == 2].pH.mean()
        pair_stab.append(d)
pair_stab = np.array(pair_stab)
tt = stats.ttest_1samp(pair_stab, 0)
print(f"paires testables: {len(pair_stab)} ; delta pH moyen = {pair_stab.mean():+.4f} ; t-test p={tt.pvalue:.3f}")
print(f"|delta pH| moyen = {np.abs(pair_stab).mean():.4f}")

# ---------------------------------------------------------------- 3. mapping paire -> cotes
print("\n=== 3. MAPPING PAIRE ORDONNEE -> COTES ===")
pm = []
for (a, b), g in df.groupby(['team_a', 'team_b']):
    trip = set(zip(g.odds_home.round(2), g.odds_draw.round(2), g.odds_away.round(2)))
    pm.append({'pair': f"{a}|{b}", 'n': len(g), 'n_triplets': len(trip),
               'pH_min': g.pH.min(), 'pH_max': g.pH.max(), 'pH_range': g.pH.max()-g.pH.min(),
               'pH_std': g.pH.std()})
pm = pd.DataFrame(pm)
print(f"paires ordonnees observees: {len(pm)} / 380 possibles")
print(f"matchs par paire: mean={pm.n.mean():.1f} min={pm.n.min()} max={pm.n.max()}")
det = (pm.n_triplets == 1) & (pm.n > 1)
multi = (pm.n_triplets > 1)
print(f"paires (n>1) avec UN SEUL triplet de cotes: {det.sum()}/{(pm.n > 1).sum()}")
print(f"paires avec >1 triplet: {multi.sum()} ; ratio triplets/matchs moyen = {(pm.n_triplets/pm.n).mean():.3f}")
print(f"pH_range au sein d'une paire: mean={pm.pH_range.mean():.4f} max={pm.pH_range.max():.4f} median={pm.pH_range.median():.4f}")
print(pm.nlargest(8, 'pH_range')[['pair', 'n', 'n_triplets', 'pH_min', 'pH_max', 'pH_range']].round(3).to_string(index=False))

# triplets globaux: menu fini ?
trip_all = set(zip(df.odds_home.round(2), df.odds_draw.round(2), df.odds_away.round(2)))
print(f"\ntriplets distincts GLOBAUX: {len(trip_all)} pour {len(df)} matchs (ratio {len(trip_all)/len(df):.3f})")

# qu'est-ce qui explique la variation intra-paire ? round ?
d2 = df.copy()
d2['pH_dm'] = d2.pH - d2.groupby(['team_a', 'team_b']).pH.transform('mean')
d2['rd_dm'] = d2['round'] - d2.groupby(['team_a', 'team_b'])['round'].transform('mean')
ok = d2.groupby(['team_a', 'team_b']).pH.transform('count') > 3
r3 = stats.pearsonr(d2[ok].rd_dm, d2[ok].pH_dm)
print(f"corr intra-paire (pH demeaned vs round demeaned): r={r3[0]:.4f} p={r3[1]:.2e}")
# variance expliquee par la paire
from sklearn.metrics import r2_score
grp_mean = d2.groupby(['team_a', 'team_b']).pH.transform('mean')
ss_res = ((d2.pH - grp_mean)**2).sum(); ss_tot = ((d2.pH - d2.pH.mean())**2).sum()
print(f"R2 de pH explique par l'identite de la paire seule: {1-ss_res/ss_tot:.4f}")

# la cote intra-paire varie-t-elle avec la forme recente (proxy: resultat du match precedent de l'equipe home) ?
df_s = df.sort_values('expected_start').reset_index(drop=True)
last_res = {}
prev_win = np.full(len(df_s), np.nan)
for i, r in df_s.iterrows():
    if r.team_a in last_res:
        prev_win[i] = last_res[r.team_a]
    last_res[r.team_a] = 1.0 if r.res == 'H' else 0.0
    last_res[r.team_b] = 1.0 if r.res == 'A' else 0.0
df_s['prev_win_home'] = prev_win
d3 = df_s.dropna(subset=['prev_win_home']).copy()
d3['pH_dm'] = d3.pH - d3.groupby(['team_a', 'team_b']).pH.transform('mean')
g0 = d3[d3.prev_win_home == 0].pH_dm
g1 = d3[d3.prev_win_home == 1].pH_dm
tt2 = stats.ttest_ind(g0, g1)
print(f"pH_dm si home a perdu/nul son dernier match: {g0.mean():+.5f} (n={len(g0)}) ; si gagne: {g1.mean():+.5f} (n={len(g1)}) ; p={tt2.pvalue:.3f}")

# ---------------------------------------------------------------- 4. EXPLOITABLE walk-forward
print("\n=== 4. EDGE PAIR-LEVEL : WALK-FORWARD 70/30 ===")
df_s = df.sort_values('expected_start').reset_index(drop=True)
n_tr = int(len(df_s)*0.7)
train, oos = df_s.iloc[:n_tr], df_s.iloc[n_tr:]
print(f"train={len(train)} oos={len(oos)} (split chrono au {df_s.expected_start.iloc[n_tr]})")

# stats train par paire ordonnee
tr_stats = {}
for (a, b), g in train.groupby(['team_a', 'team_b']):
    tr_stats[(a, b)] = {
        'n': len(g),
        'fH': (g.res == 'H').mean(), 'fD': (g.res == 'D').mean(), 'fA': (g.res == 'A').mean(),
        'pH': g.pH.mean(), 'pD': g.pD.mean(), 'pA': g.pA.mean(),
    }

K = 6  # shrinkage prior strength vers la proba implicite


def bets_for(rule_ev, rule_n, sub):
    out = []
    for _, r in sub.iterrows():
        s = tr_stats.get((r.team_a, r.team_b))
        if s is None or s['n'] < rule_n:
            continue
        for o, fo, po, odd in [('H', 'fH', 'pH', r.odds_home), ('D', 'fD', 'pD', r.odds_draw), ('A', 'fA', 'pA', r.odds_away)]:
            shrunk = (s[fo]*s['n'] + s[po]*K)/(s['n']+K)
            ev = shrunk*odd
            if ev > rule_ev:
                won = (r.res == o)
                out.append((o, odd, won, ev))
    return out


# selection de la regle SUR TRAIN UNIQUEMENT (pseudo-walk-forward interne 70/30 du train)
n_tt = int(len(train)*0.7)
tr_in, tr_val = train.iloc[:n_tt], train.iloc[n_tt:]
tr_stats_in = {}
for (a, b), g in tr_in.groupby(['team_a', 'team_b']):
    tr_stats_in[(a, b)] = {'n': len(g), 'fH': (g.res == 'H').mean(), 'fD': (g.res == 'D').mean(),
                           'fA': (g.res == 'A').mean(), 'pH': g.pH.mean(), 'pD': g.pD.mean(), 'pA': g.pA.mean()}


def bets_for_g(stats_d, rule_ev, rule_n, sub):
    out = []
    for _, r in sub.iterrows():
        s = stats_d.get((r.team_a, r.team_b))
        if s is None or s['n'] < rule_n:
            continue
        for o, fo, po, odd in [('H', 'fH', 'pH', r.odds_home), ('D', 'fD', 'pD', r.odds_draw), ('A', 'fA', 'pA', r.odds_away)]:
            shrunk = (s[fo]*s['n'] + s[po]*K)/(s['n']+K)
            if shrunk*odd > rule_ev:
                out.append((o, odd, r.res == o))
    return out


print("\n--- selection de regle sur train interne (70/30 du train) ---")
best = None
for ev_th in [1.03, 1.06, 1.10, 1.15]:
    for n_th in [4, 6, 8]:
        b = bets_for_g(tr_stats_in, ev_th, n_th, tr_val)
        if len(b) < 20:
            print(f"EV>{ev_th} n>={n_th}: {len(b)} bets (trop peu)")
            continue
        roi = np.mean([(odd-1) if w else -1 for _, odd, w in b])
        print(f"EV>{ev_th} n>={n_th}: n={len(b)} ROI={roi*100:+.1f}%")
        if best is None or roi > best[2]:
            best = (ev_th, n_th, roi, len(b))
if best:
    print(f"--> regle retenue: EV>{best[0]}, n_train>={best[1]} (ROI val interne {best[2]*100:+.1f}% sur {best[3]} bets)")
    b_oos = bets_for(best[0], best[1], oos)
    if b_oos:
        roi = np.mean([(odd-1) if w else -1 for _, odd, w, _ in b_oos])
        wr = np.mean([w for _, _, w, _ in b_oos])
        ao = np.mean([odd for _, odd, _, _ in b_oos])
        # test binomial: WR vs 1/odds moyen attendu
        p_exp = np.mean([1/odd for _, odd, _, _ in b_oos])
        pv = stats.binomtest(int(sum(w for _, _, w, _ in b_oos)), len(b_oos), p_exp).pvalue
        print(f"==> OOS: n={len(b_oos)} ROI={roi*100:+.1f}% WR={wr*100:.1f}% (implied {p_exp*100:.1f}%) avg_odds={ao:.2f} p_binom={pv:.3f}")
    else:
        print("==> OOS: 0 bets")

# baseline: parier TOUT en OOS pour contexte
for o, odd_c in [('H', 'odds_home'), ('D', 'odds_draw'), ('A', 'odds_away')]:
    won = (oos.res == o)
    roi = np.mean(np.where(won, oos[odd_c]-1, -1))
    print(f"baseline OOS bet-all {o}: ROI={roi*100:+.1f}% (n={len(oos)})")

# ---------------- 4b. residus TEAM-level (plus de stat power que pair-level)
print("\n--- 4b. residus team-level (train) -> ROI OOS ---")
flags = []
for t in teams:
    for venue, mask_tr, mask_oos, pcol, ocol, out in [
            ('home', train.team_a == t, oos.team_a == t, 'pH', 'odds_home', 'H'),
            ('away', train.team_b == t, oos.team_b == t, 'pA', 'odds_away', 'A')]:
        g = train[mask_tr]
        if len(g) < 50:
            continue
        resid = (g.res == out).astype(float) - g[pcol]
        t_st, p = stats.ttest_1samp(resid, 0)
        if p < 0.05:
            flags.append((t, venue, resid.mean(), p, len(g), t_st > 0, ocol, out, mask_oos))
print(f"flags train p<0.05: {len(flags)} / 40 tests (attendu ~2 sous H0)")
tot_pnl, tot_n = 0, 0
for t, venue, rm, p, n, pos, ocol, out, mask_oos in flags:
    g = oos[mask_oos]
    if pos:  # under-priced -> back
        pnl = np.where(g.res == out, g[ocol]-1, -1).sum()
        print(f"  BACK {t} {venue} (resid train {rm:+.3f}, p={p:.3f}, n_tr={n}): OOS n={len(g)} PnL={pnl:+.1f}u ROI={pnl/max(len(g),1)*100:+.1f}%")
        tot_pnl += pnl; tot_n += len(g)
    else:
        print(f"  (over-priced {t} {venue} resid {rm:+.3f} p={p:.3f} -> pas de back direct)")
if tot_n:
    print(f"  TOTAL backs OOS: n={tot_n} ROI={tot_pnl/tot_n*100:+.1f}%")

# ---------------------------------------------------------------- 5. hierarchie / tiers
print("\n=== 5. HIERARCHIE: TIERS ===")
s = prof[['team', 'PPG', 'pH_imp', 'pA_imp']].copy()
s['strength_imp'] = (s.pH_imp + s.pA_imp)/2
s = s.sort_values('strength_imp', ascending=False).reset_index(drop=True)
s['gap'] = s.strength_imp.diff(-1)
print(s.round(3).to_string(index=False))
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
Xs = s[['strength_imp']].values
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    print(f"k={k}: silhouette={sil:.3f}")
km_best = None
best_sil = -1
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    if sil > best_sil:
        best_sil, km_best, kk = sil, km, k
order = np.argsort(-km_best.cluster_centers_.ravel())
remap = {old: new for new, old in enumerate(order)}
s['tier'] = [remap[l]+1 for l in km_best.labels_]
print(f"\nmeilleur k={kk} (sil={best_sil:.3f}):")
for tier in sorted(s.tier.unique()):
    tt = s[s.tier == tier]
    print(f"  TIER {tier}: {list(tt.team)} (strength {tt.strength_imp.min():.3f}-{tt.strength_imp.max():.3f})")

print("\nDONE")
