# -*- coding: utf-8 -*-
"""WF3 - LE SCRIPT DE SAISON : feedback classement -> cotes.

Q1. Paires jouees plusieurs fois : variance des cotes d'ouverture entre occurrences.
    ~0 -> figees ; sinon regression de la deviation sur classement/forme/journee.
Q2. La deviation vs cote moyenne de paire est-elle predictive ? (moteur change vraiment
    ses probas vs ajustement cosmetique). Walk-forward 70/30.
Q3. Champion scripte : le vainqueur final sur-performait-il ses cotes en debut de saison ?
    (champion defini sur J11-38 pour eviter le biais de selection, teste sur J1-10)
    + Monte Carlo : dispersion des points finaux vs tirage i.i.d. des cotes.
Q4. Drift intra-event entre snapshots : informatif ? logloss(open) vs logloss(last),
    coefficient logistique du drift, walk-forward ROI.
"""
import sys, json, math
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scraper.config import load_settings
from sqlalchemy import create_engine, text

rng = np.random.default_rng(42)
SEP = "=" * 88

def parse_t(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

def devig(oh, od, oa):
    inv = 1/oh + 1/od + 1/oa
    return (1/oh)/inv, (1/od)/inv, (1/oa)/inv

def logit(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return math.log(p/(1-p))

# ------------------------------------------------------------------ logistic helpers
def fit_logistic(X, y):
    """X (n,k) sans colonne intercept; retourne beta (k+1,), loglik."""
    Xd = np.column_stack([np.ones(len(X)), X])
    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z)) - np.sum(y * z)
    r = minimize(nll, np.zeros(Xd.shape[1]), method='BFGS')
    return r.x, -r.fun

def lr_test(ll_full, ll_red, df_):
    lr = 2 * (ll_full - ll_red)
    return lr, stats.chi2.sf(max(lr, 0), df_)

def logistic_se(X, beta):
    Xd = np.column_stack([np.ones(len(X)), X])
    z = Xd @ beta
    p = 1/(1+np.exp(-z))
    H = Xd.T @ (Xd * (p*(1-p))[:, None])
    try:
        return np.sqrt(np.diag(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        return np.full(len(beta), np.nan)

def ols_pvals(X, y):
    Xd = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n, k = Xd.shape
    sigma2 = resid @ resid / (n - k)
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - resid @ resid / ss_tot if ss_tot > 0 else 0.0
    return beta, t, p, r2

# ================================================================== LOAD
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    evs = c.execute(text(
        "SELECT e.id, CAST(e.round_info AS INT) rd, e.team_a, e.team_b, e.expected_start, "
        "r.score_a, r.score_b "
        "FROM events e LEFT JOIN results r ON r.event_id=e.id "
        "ORDER BY e.expected_start, e.id")).fetchall()
    snaps = c.execute(text(
        "SELECT event_id, id, odds_home, odds_draw, odds_away, captured_at "
        "FROM odds_snapshots ORDER BY event_id, id")).fetchall()

snaps_by_ev = defaultdict(list)
for r in snaps:
    if r[2] and r[3] and r[4] and r[2] > 1.0 and r[3] > 1.0 and r[4] > 1.0:
        snaps_by_ev[r[0]].append((r[1], r[2], r[3], r[4], parse_t(r[5])))

open_odds = {}   # MIN(id) snapshot = cote d'ouverture
for eid, lst in snaps_by_ev.items():
    lst.sort(key=lambda x: x[0])
    open_odds[eid] = lst[0]

# dedup (team_a, team_b, expected_start), garder celui avec result; drop round 0/None
seen = {}
for r in evs:
    if r[1] is None or r[1] == 0:
        continue
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))
print(f"events dedup={len(evs2)}  avec result={sum(1 for r in evs2 if r[5] is not None)}")

# ------------------------------------------------------------------ saisons
seasons = []
cur, last_rd, last_t = [], None, None
for r in evs2:
    rd, t = r[1], parse_t(r[4])
    new = False
    if last_rd is not None:
        if rd < last_rd - 4: new = True
        if last_t is not None and (t - last_t).total_seconds() > 45 * 60: new = True
    if new and cur:
        seasons.append(cur); cur = []; last_rd = None
    cur.append(r)
    last_rd = rd if last_rd is None else max(last_rd, rd)
    last_t = t
if cur: seasons.append(cur)
print(f"saisons reconstruites={len(seasons)}  "
      f"tailles top12={sorted([len(s) for s in seasons], reverse=True)[:12]}")

# ------------------------------------------------------------------ match rows + standings
rows = []
for sid, seg in enumerate(seasons):
    by_rd = defaultdict(list)
    for r in seg:
        by_rd[r[1]].append(r)
    table = defaultdict(lambda: {"pts": 0, "gd": 0, "played": 0, "hist": []})
    for rd in sorted(by_rd):
        standing = sorted(table.items(), key=lambda kv: (-kv[1]["pts"], -kv[1]["gd"]))
        pos_map = {t: i + 1 for i, (t, d) in enumerate(standing)}
        snap_tbl = {t: dict(d, hist=list(d["hist"])) for t, d in table.items()}
        for r in by_rd[rd]:
            eid, _, ta, tb, est, sa, sb = r
            if sa is not None and eid in open_odds:
                _, oh, od, oa, t_cap = open_odds[eid]
                ph, pd_, pa = devig(oh, od, oa)
                d = dict(eid=eid, sid=sid, rd=rd, ta=ta, tb=tb, t=parse_t(est),
                         oh=oh, od=od, oa=oa, ph=ph, pd=pd_, pa=pa,
                         y=(0 if sa > sb else (1 if sa == sb else 2)))
                fa, fb = snap_tbl.get(ta), snap_tbl.get(tb)
                if fa and fb and fa["played"] >= 3 and fb["played"] >= 3:
                    d["pos_diff"] = pos_map[tb] - pos_map[ta]      # >0 = home mieux classe
                    d["pts_diff"] = fa["pts"] - fb["pts"]
                    f5a = np.mean(fa["hist"][-5:]) if fa["hist"] else np.nan
                    f5b = np.mean(fb["hist"][-5:]) if fb["hist"] else np.nan
                    d["form_diff"] = f5a - f5b
                rows.append(d)
            # update table apres la journee
            if sa is not None:
                pa_, pb_ = (3, 0) if sa > sb else ((1, 1) if sa == sb else (0, 3))
                for tm, p_, gd_ in ((ta, pa_, sa - sb), (tb, pb_, sb - sa)):
                    table[tm]["pts"] += p_; table[tm]["gd"] += gd_
                    table[tm]["played"] += 1; table[tm]["hist"].append(p_)

rows.sort(key=lambda d: d["t"])
print(f"matchs finis + cotes ouverture = {len(rows)}")

t_split = rows[int(len(rows) * 0.70)]["t"]
train = [d for d in rows if d["t"] < t_split]
test = [d for d in rows if d["t"] >= t_split]
print(f"split temporel 70/30 : train={len(train)} test={len(test)}  (split @ {t_split})")

# ================================================================== Q1
print(f"\n{SEP}\nQ1. COTES D'OUVERTURE PAR PAIRE : FIGEES OU MODULEES ?\n{SEP}")
by_pair = defaultdict(list)
for d in rows:
    by_pair[(d["ta"], d["tb"])].append(d)

stds_between, cvs, n_identical = [], [], 0
for pair, lst in by_pair.items():
    if len(lst) < 4:
        continue
    phs = np.array([d["ph"] for d in lst])
    ohs = np.array([d["oh"] for d in lst])
    stds_between.append(phs.std(ddof=1))
    cvs.append(ohs.std(ddof=1) / ohs.mean())
    if len(set(ohs.tolist())) == 1:
        n_identical += 1
    for d in lst:
        d["pair_mean_ph"] = phs.mean()
stds_between = np.array(stds_between)
print(f"paires (>=4 occ.) = {len(stds_between)}  dont cotes home STRICTEMENT identiques : {n_identical}")
print(f"std INTER-occurrences de p(home) devigge : "
      f"mediane={np.median(stds_between):.4f}  moy={stds_between.mean():.4f}  "
      f"p90={np.percentile(stds_between,90):.4f}")
print(f"CV des cotes home : mediane={np.median(cvs)*100:.2f}%  moy={np.mean(cvs)*100:.2f}%")

# bruit INTRA-event comme reference de jitter
intra_stds = []
for eid, lst in snaps_by_ev.items():
    if len(lst) >= 3:
        phs = np.array([devig(o[1], o[2], o[3])[0] for o in lst])
        intra_stds.append(phs.std(ddof=1))
intra_stds = np.array(intra_stds)
print(f"std INTRA-event (jitter snapshots, n={len(intra_stds)} events) : "
      f"mediane={np.median(intra_stds):.4f}  moy={intra_stds.mean():.4f}")
mw = stats.mannwhitneyu(stds_between, intra_stds, alternative='greater')
print(f"Mann-Whitney inter > intra : U={mw.statistic:.0f}  p={mw.pvalue:.2e}")
print(f"ratio mediane inter/intra = {np.median(stds_between)/max(np.median(intra_stds),1e-9):.2f}")

print("\n-- Regression FE (demean par paire) : dev p(home) ~ pos_diff + form_diff + journee --")
feat = [d for d in rows if "pair_mean_ph" in d and "pos_diff" in d
        and not (isinstance(d.get("form_diff"), float) and math.isnan(d["form_diff"]))]
grp = defaultdict(list)
for d in feat:
    grp[(d["ta"], d["tb"])].append(d)
Xl, yl = [], []
for pair, lst in grp.items():
    if len(lst) < 3:
        continue
    pos = np.array([d["pos_diff"] for d in lst], float)
    frm = np.array([d["form_diff"] for d in lst], float)
    rdv = np.array([d["rd"] for d in lst], float)
    ph = np.array([d["ph"] for d in lst], float)
    for i in range(len(lst)):
        Xl.append([pos[i]-pos.mean(), frm[i]-frm.mean(), rdv[i]-rdv.mean()])
        yl.append(ph[i]-ph.mean())
Xl, yl = np.array(Xl), np.array(yl)
beta, tv, pv, r2 = ols_pvals(Xl, yl)
for nm, b, t_, p_ in zip(["intercept", "pos_diff(dm)", "form_diff(dm)", "journee(dm)"], beta, tv, pv):
    print(f"  {nm:14s} beta={b:+.5f}  t={t_:+6.2f}  p={p_:.2e}")
print(f"  n={len(yl)}  R2={r2:.4f}")
# regressions univariees pour lisibilite
for j, nm in enumerate(["pos_diff", "form_diff", "journee"]):
    b1, t1, p1, r21 = ols_pvals(Xl[:, [j]], yl)
    print(f"  univarie {nm:10s} : beta={b1[1]:+.5f} t={t1[1]:+6.2f} p={p1[1]:.2e} R2={r21:.4f}")

ph_all = np.array([d["ph"] for d in rows if "pair_mean_ph" in d])
pm_all = np.array([d["pair_mean_ph"] for d in rows if "pair_mean_ph" in d])
ss_tot = np.sum((ph_all - ph_all.mean())**2)
ss_within = np.sum((ph_all - pm_all)**2)
print(f"\nANOVA : l'identite de la paire explique {(1-ss_within/ss_tot)*100:.2f}% "
      f"de la variance de p(home) ; reste within-pair = {ss_within/ss_tot*100:.2f}%")

# ================================================================== Q2
print(f"\n{SEP}\nQ2. DEVIATION vs MOYENNE DE PAIRE : VRAIE INFO OU COSMETIQUE ? (WF 70/30)\n{SEP}")
pair_train = defaultdict(list)
for d in train:
    pair_train[(d["ta"], d["tb"])].append(d["ph"])
pair_mu = {k: float(np.mean(v)) for k, v in pair_train.items() if len(v) >= 3}

test_q2 = [d for d in test if (d["ta"], d["tb"]) in pair_mu]
print(f"test avec moyenne de paire calculee sur train (>=3 occ.) : n={len(test_q2)}")

y_home = np.array([1 if d["y"] == 0 else 0 for d in test_q2])
lg_cur = np.array([logit(d["ph"]) for d in test_q2])
lg_mu = np.array([logit(pair_mu[(d["ta"], d["tb"])]) for d in test_q2])
dev = lg_cur - lg_mu
p_cur, p_mu = 1/(1+np.exp(-lg_cur)), 1/(1+np.exp(-lg_mu))
ll_cur = -np.mean(y_home*np.log(p_cur) + (1-y_home)*np.log(1-p_cur))
ll_mu = -np.mean(y_home*np.log(p_mu) + (1-y_home)*np.log(1-p_mu))
diffs = (y_home*np.log(p_cur)+(1-y_home)*np.log(1-p_cur)) - (y_home*np.log(p_mu)+(1-y_home)*np.log(1-p_mu))
tt = stats.ttest_1samp(diffs, 0)
print(f"logloss(home) OOS : cote courante={ll_cur:.4f}  moyenne paire={ll_mu:.4f}  "
      f"delta={ll_mu-ll_cur:+.4f}  t apparie p={tt.pvalue:.2e}")
print("  (delta>0 = la cote du jour bat l'historique de la paire -> le moteur change VRAIMENT)")

tr_q2 = [d for d in train if (d["ta"], d["tb"]) in pair_mu]
ytr = np.array([1 if d["y"] == 0 else 0 for d in tr_q2])
lgc_tr = np.array([logit(d["ph"]) for d in tr_q2])
lgm_tr = np.array([logit(pair_mu[(d["ta"], d["tb"])]) for d in tr_q2])
dev_tr = lgc_tr - lgm_tr
bB, llB = fit_logistic(np.column_stack([lgm_tr, dev_tr]), ytr)
bA, llA = fit_logistic(lgm_tr.reshape(-1, 1), ytr)
lr, p_lr = lr_test(llB, llA, 1)
se = logistic_se(np.column_stack([lgm_tr, dev_tr]), bB)
print(f"\nTRAIN  y_home ~ logit(mu_paire) + dev  (n={len(ytr)}, std(dev)={dev_tr.std():.3f})")
print(f"  coef(mu)={bB[1]:+.3f}+-{se[1]:.3f}  coef(dev)={bB[2]:+.3f}+-{se[2]:.3f}  LR={lr:.1f} p={p_lr:.2e}")
bBt, llBt = fit_logistic(np.column_stack([lg_mu, dev]), y_home)
bAt, llAt = fit_logistic(lg_mu.reshape(-1, 1), y_home)
lrt_, p_lrt = lr_test(llBt, llAt, 1)
set_ = logistic_se(np.column_stack([lg_mu, dev]), bBt)
print(f"TEST   coef(mu)={bBt[1]:+.3f}+-{set_[1]:.3f}  coef(dev)={bBt[2]:+.3f}+-{set_[2]:.3f}  "
      f"LR={lrt_:.1f} p={p_lrt:.2e}")
print("  coef(dev) ~ coef(mu) -> probas reellement modulees ; coef(dev)~0 -> cosmetique")

print("\n-- Strategies deviation (signal choisi sur TRAIN, ROI sur OOS) --")
def eval_strat(ds, side, mode, thr, mu_map):
    st, ret, w, ol = 0, 0.0, 0, []
    for d in ds:
        mu = mu_map.get((d["ta"], d["tb"]))
        if mu is None: continue
        dv = logit(d["ph"]) - logit(mu)
        if side == "home":
            o, won = d["oh"], d["y"] == 0
            sig = (dv < -thr) if mode == "fade" else (dv > thr)
        else:
            o, won = d["oa"], d["y"] == 2
            sig = (dv > thr) if mode == "fade" else (dv < -thr)
        if sig:
            st += 1; ol.append(o)
            if won: ret += o; w += 1
    return st, ((ret-st)/st*100 if st else float('nan')), \
           (w/st*100 if st else float('nan')), (np.mean(ol) if ol else float('nan')), w

best = None
for side in ("home", "away"):
    for mode in ("fade", "follow"):
        for thr in (0.05, 0.10, 0.15, 0.20):
            n, roi, wr, ao, w = eval_strat(train, side, mode, thr, pair_mu)
            mark = ""
            if n >= 80 and (best is None or roi > best[0]):
                best = (roi, side, mode, thr); mark = "  <-"
            print(f"  TRAIN {side:4s} {mode:6s} thr={thr:.2f} : n={n:4d} roi={roi:+6.2f}% wr={wr:5.1f}% cote={ao:5.2f}{mark}")
if best:
    _, side, mode, thr = best
    n, roi, wr, ao, w = eval_strat(test_q2, side, mode, thr, pair_mu)
    pbin = stats.binomtest(w, n, 1/ao, alternative='greater').pvalue if n and not math.isnan(ao) else float('nan')
    print(f"\n  MEILLEUR TRAIN = {side}/{mode}/thr={thr}")
    print(f"  >>> OOS : n={n} roi={roi:+.2f}% wr={wr:.1f}% cote_moy={ao:.2f}  binom p(WR>1/cote)={pbin:.3f}")

# ================================================================== Q3
print(f"\n{SEP}\nQ3. CHAMPION SCRIPTE ? (selection J11-38, test J1-10 + Monte Carlo)\n{SEP}")
full_seasons = []
for sid, seg in enumerate(seasons):
    rds = set(r[1] for r in seg)
    fin = [r for r in seg if r[5] is not None and r[0] in open_odds]
    if len(rds) >= 35 and len(fin) >= 330:
        full_seasons.append((sid, fin))
print(f"saisons quasi completes (>=35 journees, >=330 matchs finis+cotes) : {len(full_seasons)}")

def season_points(fin, rd_min, rd_max):
    pts = defaultdict(float)
    for r in fin:
        _, rd, ta, tb, _, sa, sb = r
        if not (rd_min <= rd <= rd_max): continue
        if sa > sb: pts[ta] += 3
        elif sa == sb: pts[ta] += 1; pts[tb] += 1
        else: pts[tb] += 3
    return pts

def early_perf(fin, team, rd_max=10):
    dp, vp, nm, wa, we = 0.0, 0.0, 0, 0, 0.0
    for r in fin:
        eid, rd, ta, tb, _, sa, sb = r
        if rd > rd_max or team not in (ta, tb): continue
        _, oh, od, oa, _ = open_odds[eid]
        ph, pd_, pa = devig(oh, od, oa)
        if ta == team:
            pw, pdr = ph, pd_; apts = 3 if sa > sb else (1 if sa == sb else 0)
        else:
            pw, pdr = pa, pd_; apts = 3 if sb > sa else (1 if sa == sb else 0)
        epts = 3*pw + pdr
        dp += apts - epts; vp += 9*pw + pdr - epts**2
        nm += 1; wa += 1 if apts == 3 else 0; we += pw
    return dp, vp, nm, wa, we

agg = dict(dp=0.0, vp=0.0, nm=0, wa=0, we=0.0)
per_season = []
for sid, fin in full_seasons:
    late = season_points(fin, 11, 38)
    if not late: continue
    champ = max(late, key=late.get)
    dp, vp, nm, wa, we = early_perf(fin, champ)
    for k, v in zip(("dp", "vp", "nm", "wa", "we"), (dp, vp, nm, wa, we)):
        agg[k] += v
    per_season.append((sid, champ, nm, dp))
z = agg["dp"]/math.sqrt(agg["vp"]) if agg["vp"] > 0 else float('nan')
print(f"\nChampion(J11-38) sur J1-10 : {agg['nm']} matchs, delta points (obs-attendu) = {agg['dp']:+.1f}"
      f"  z={z:+.2f}  p(2s)={2*stats.norm.sf(abs(z)):.3f}")
print(f"  victoires : {agg['wa']} obs vs {agg['we']:.1f} attendues ({agg['wa']-agg['we']:+.1f})")
for sid, ch, nm, dp in per_season:
    print(f"   saison {sid:3d}  champ={ch:18s} J1-10 n={nm:2d}  delta_pts={dp:+5.1f}")

aggb = dict(dp=0.0, vp=0.0, nm=0)
for sid, fin in full_seasons:
    late = season_points(fin, 11, 38)
    if not late: continue
    teams_late = {t for r in fin if r[1] > 10 for t in (r[2], r[3])}
    bottom = min(teams_late, key=lambda t: late.get(t, 0))
    dp, vp, nm, _, _ = early_perf(fin, bottom)
    aggb["dp"] += dp; aggb["vp"] += vp; aggb["nm"] += nm
zb = aggb["dp"]/math.sqrt(aggb["vp"]) if aggb["vp"] > 0 else float('nan')
print(f"\nLanterne rouge(J11-38) sur J1-10 : {aggb['nm']} matchs, delta_pts={aggb['dp']:+.1f}"
      f"  z={zb:+.2f}  p={2*stats.norm.sf(abs(zb)):.3f}")

print("\n-- Monte Carlo (2000 sims/saison) : points du champion + dispersion finale --")
NSIM = 2000
champ_pctl, std_pctl = [], []
for sid, fin in full_seasons:
    probs, homes, aways = [], [], []
    for r in fin:
        eid = r[0]
        _, oh, od, oa, _ = open_odds[eid]
        probs.append(devig(oh, od, oa)); homes.append(r[2]); aways.append(r[3])
    probs = np.array(probs)
    teams = sorted(set(homes) | set(aways))
    tidx = {t: i for i, t in enumerate(teams)}
    hi = np.array([tidx[t] for t in homes]); ai = np.array([tidx[t] for t in aways])
    act = season_points(fin, 1, 38)
    act_v = np.array([act.get(t, 0) for t in teams], float)
    u = rng.random((NSIM, len(fin)))
    out = np.where(u < probs[:, 0], 0, np.where(u < probs[:, 0] + probs[:, 1], 1, 2))
    sim_pts = np.zeros((NSIM, len(teams)))
    for k in range(len(fin)):
        o = out[:, k]
        sim_pts[:, hi[k]] += np.where(o == 0, 3, np.where(o == 1, 1, 0))
        sim_pts[:, ai[k]] += np.where(o == 2, 3, np.where(o == 1, 1, 0))
    champ_pctl.append((sim_pts.max(axis=1) < act_v.max()).mean())
    std_pctl.append((sim_pts.std(axis=1) < act_v.std()).mean())
champ_pctl, std_pctl = np.array(champ_pctl), np.array(std_pctl)
print(f"percentile points du champion reel vs sims : moy={champ_pctl.mean():.3f} (0.5 si i.i.d.)")
print(f"percentile du std des points finaux        : moy={std_pctl.mean():.3f}")
ks1 = stats.kstest(champ_pctl, 'uniform'); ks2 = stats.kstest(std_pctl, 'uniform')
tt1 = stats.ttest_1samp(champ_pctl, 0.5); tt2 = stats.ttest_1samp(std_pctl, 0.5)
print(f"KS uniformite : champion D={ks1.statistic:.3f} p={ks1.pvalue:.3f} | "
      f"std D={ks2.statistic:.3f} p={ks2.pvalue:.3f}")
print(f"t-test vs 0.5 : champion t={tt1.statistic:+.2f} p={tt1.pvalue:.3f} | "
      f"std t={tt2.statistic:+.2f} p={tt2.pvalue:.3f}")

# ================================================================== Q4
print(f"\n{SEP}\nQ4. DRIFT INTRA-EVENT (1er vs dernier snapshot avant coup d'envoi)\n{SEP}")
ev_t = {d["eid"]: d for d in rows}
drift_rows = []
for eid, lst in snaps_by_ev.items():
    d = ev_t.get(eid)
    if d is None: continue
    pre = [s for s in lst if s[4] <= d["t"]]
    if len(pre) < 2: continue
    first, last = pre[0], pre[-1]
    gap_min = (last[4] - first[4]).total_seconds() / 60
    if gap_min < 5: continue
    ph_f = devig(first[1], first[2], first[3])[0]
    ph_l = devig(last[1], last[2], last[3])[0]
    drift_rows.append(dict(t=d["t"], y=d["y"], gap=gap_min,
                           lg_f=logit(ph_f), lg_l=logit(ph_l),
                           oh_l=last[1], oa_l=last[3]))
drift_rows.sort(key=lambda d: d["t"])
print(f"events avec >=2 snapshots pre-match (gap>=5min) et result : n={len(drift_rows)}")
if len(drift_rows) >= 100:
    yh = np.array([1 if d["y"] == 0 else 0 for d in drift_rows])
    lf = np.array([d["lg_f"] for d in drift_rows])
    ll_ = np.array([d["lg_l"] for d in drift_rows])
    dr = ll_ - lf
    g = np.array([d["gap"] for d in drift_rows])
    print(f"drift logit(p_home) : moy={dr.mean():+.4f} std={dr.std():.4f}  "
          f"|drift|>0.05 : {(np.abs(dr)>0.05).mean()*100:.1f}%")
    rho = stats.spearmanr(g, np.abs(dr))
    print(f"corr(gap_min, |drift|) Spearman rho={rho.statistic:+.3f} p={rho.pvalue:.2e}"
          f"  (rho>0 = marche aleatoire, ~0 = jitter stationnaire)")
    p_f, p_l = 1/(1+np.exp(-lf)), 1/(1+np.exp(-ll_))
    lo_f = -(yh*np.log(p_f) + (1-yh)*np.log(1-p_f))
    lo_l = -(yh*np.log(p_l) + (1-yh)*np.log(1-p_l))
    tt = stats.ttest_rel(lo_f, lo_l)
    print(f"logloss(home) : 1er snap={lo_f.mean():.4f}  dernier={lo_l.mean():.4f}  "
          f"delta={lo_f.mean()-lo_l.mean():+.4f}  t apparie p={tt.pvalue:.3f}")
    b2, ll2 = fit_logistic(np.column_stack([ll_, dr]), yh)
    b1, ll1 = fit_logistic(ll_.reshape(-1, 1), yh)
    lr_, plr_ = lr_test(ll2, ll1, 1)
    se2 = logistic_se(np.column_stack([ll_, dr]), b2)
    print(f"y ~ logit(last) + drift : coef(drift)={b2[2]:+.3f}+-{se2[2]:.3f}  LR={lr_:.2f} p={plr_:.3f}")
    cut = int(len(drift_rows) * 0.7)
    trn, tst = drift_rows[:cut], drift_rows[cut:]
    def drift_strat(ds, thr, follow=True):
        st, ret, w, ol = 0, 0.0, 0, []
        for d in ds:
            dv = d["lg_l"] - d["lg_f"]
            if dv > thr: side = "home" if follow else "away"
            elif dv < -thr: side = "away" if follow else "home"
            else: continue
            o = d["oh_l"] if side == "home" else d["oa_l"]
            won = (d["y"] == 0) if side == "home" else (d["y"] == 2)
            st += 1; ol.append(o)
            if won: ret += o; w += 1
        return st, ((ret-st)/st*100 if st else float('nan')), \
               (w/st*100 if st else float('nan')), (np.mean(ol) if ol else float('nan')), w
    bestd = None
    for fol in (True, False):
        for thr in (0.03, 0.05, 0.10):
            n, roi, wr, ao, w = drift_strat(trn, thr, fol)
            tag = "follow" if fol else "fade"
            mark = ""
            if n >= 50 and (bestd is None or roi > bestd[0]):
                bestd = (roi, fol, thr); mark = "  <-"
            print(f"  TRAIN drift {tag:6s} thr={thr:.2f} : n={n:4d} roi={roi:+6.2f}% wr={wr:5.1f}%{mark}")
    if bestd:
        _, fol, thr = bestd
        n, roi, wr, ao, w = drift_strat(tst, thr, fol)
        pbin = stats.binomtest(w, n, 1/ao, alternative='greater').pvalue if n and ao and not math.isnan(ao) else float('nan')
        print(f"  >>> OOS drift {'follow' if fol else 'fade'} thr={thr} : "
              f"n={n} roi={roi:+.2f}% wr={wr:.1f}% cote={ao:.2f} binom p={pbin:.3f}")
else:
    print("echantillon insuffisant")

print(f"\n{SEP}\nFIN\n{SEP}")
