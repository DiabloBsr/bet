# -*- coding: utf-8 -*-
"""WF3 - SCRIPT DE SAISON, iteration 3 : verdict causal pleine puissance.

Moyenne de paire CAUSALE (occurrences passees uniquement, >=3) sur tout l'echantillon :
 1. logloss(open) vs logloss(mu_causal) apparie, 3 classes + binaire home
 2. logistique y_home ~ logit(mu_causal) + dev : coef(dev) avec IC (0=cosmetique, 1=reel)
 3. fade-open toutes donnees, seuils 0.03/0.05/0.07, par cote, MC p sous 2 nulls (pas de
    selection de variante : tout est affiche)
 4. follow/fade drift intra-event pleine periode, MC p sous truth=last
 5. decomposition de variance : jitter publication vs variance vraie entre occurrences
"""
import sys, math
from collections import defaultdict
from datetime import datetime
sys.path.insert(0, '.')
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scraper.config import load_settings
from sqlalchemy import create_engine, text

rng = np.random.default_rng(11)
SEP = "=" * 88

def parse_t(s):
    return datetime.fromisoformat(str(s).replace('Z', ''))

def devig(oh, od, oa):
    inv = 1/oh + 1/od + 1/oa
    return (1/oh)/inv, (1/od)/inv, (1/oa)/inv

def logit(p):
    p = min(max(p, 1e-6), 1-1e-6)
    return math.log(p/(1-p))

def fit_logistic(X, y):
    Xd = np.column_stack([np.ones(len(X)), X])
    def nll(b):
        z = Xd @ b
        return np.sum(np.logaddexp(0, z)) - np.sum(y * z)
    r = minimize(nll, np.zeros(Xd.shape[1]), method='BFGS')
    return r.x, -r.fun

def logistic_se(X, beta):
    Xd = np.column_stack([np.ones(len(X)), X])
    z = Xd @ beta
    p = 1/(1+np.exp(-z))
    H = Xd.T @ (Xd * (p*(1-p))[:, None])
    return np.sqrt(np.diag(np.linalg.inv(H)))

# ================================================================== LOAD (identique iter2)
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
for lst in snaps_by_ev.values():
    lst.sort(key=lambda x: x[0])

seen = {}
for r in evs:
    if r[1] is None or r[1] == 0:
        continue
    k = (r[2], r[3], str(r[4]))
    if k not in seen or (seen[k][5] is None and r[5] is not None):
        seen[k] = r
evs2 = sorted(seen.values(), key=lambda r: (str(r[4]), r[0]))

rows = []
for r in evs2:
    eid, rd, ta, tb, est, sa, sb = r
    if sa is None or eid not in snaps_by_ev:
        continue
    t_start = parse_t(est)
    lst = snaps_by_ev[eid]
    pre = [s for s in lst if s[4] <= t_start]
    open_s, last_s = lst[0], (pre[-1] if pre else lst[0])
    rows.append(dict(eid=eid, ta=ta, tb=tb, t=t_start,
                     o_open=(open_s[1], open_s[2], open_s[3]),
                     o_last=(last_s[1], last_s[2], last_s[3]),
                     p_open=devig(open_s[1], open_s[2], open_s[3]),
                     p_last=devig(last_s[1], last_s[2], last_s[3]),
                     gap=(last_s[4] - open_s[4]).total_seconds() / 60,
                     y=(0 if sa > sb else (1 if sa == sb else 2))))
rows.sort(key=lambda d: d["t"])
print(f"matchs finis+cotes = {len(rows)}")

# moyenne de paire CAUSALE : occurrences passees uniquement (>=3)
hist = defaultdict(list)
for d in rows:
    k = (d["ta"], d["tb"])
    if len(hist[k]) >= 3:
        d["mu"] = tuple(np.mean(np.array(hist[k]), axis=0))
        d["k_hist"] = len(hist[k])
    hist[k].append(d["p_open"])
sub = [d for d in rows if "mu" in d]
print(f"matchs avec mu causale (>=3 occurrences passees) = {len(sub)}")

# ================================================================== 1. LOGLOSS APPARIE
print(f"\n{SEP}\n1. LOGLOSS (pleine periode, mu causale)\n{SEP}")
def ll3(d, p): return -math.log(max(p[d["y"]], 1e-9))
Lo = np.array([ll3(d, d["p_open"]) for d in sub])
Lm = np.array([ll3(d, d["mu"]) for d in sub])
Ll = np.array([ll3(d, d["p_last"]) for d in sub])
for nm, v in (("open", Lo), ("last", Ll), ("mu_causal", Lm)):
    print(f"  logloss3 {nm:9s} = {v.mean():.4f}")
for a, va, b, vb in (("open", Lo, "mu", Lm), ("last", Ll, "mu", Lm), ("open", Lo, "last", Ll)):
    tt = stats.ttest_rel(va, vb)
    print(f"  {a} vs {b} : delta={va.mean()-vb.mean():+.5f}  t={tt.statistic:+.2f} p={tt.pvalue:.3f}")

# ================================================================== 2. LOGISTIQUE DEV
print(f"\n{SEP}\n2. y_home ~ logit(mu) + dev (pleine periode)\n{SEP}")
yh = np.array([1 if d["y"] == 0 else 0 for d in sub])
lgm = np.array([logit(d["mu"][0]) for d in sub])
dev = np.array([logit(d["p_open"][0]) - logit(d["mu"][0]) for d in sub])
b, ll = fit_logistic(np.column_stack([lgm, dev]), yh)
se = logistic_se(np.column_stack([lgm, dev]), b)
print(f"n={len(sub)}  std(dev)={dev.std():.4f}")
print(f"coef(mu) ={b[1]:+.3f} +- {se[1]:.3f}")
print(f"coef(dev)={b[2]:+.3f} +- {se[2]:.3f}  IC95=[{b[2]-1.96*se[2]:+.2f},{b[2]+1.96*se[2]:+.2f}]")
print("  0 = jitter cosmetique ; 1 = le resultat est tire des cotes affichees du jour")
# meme chose avec dev du DERNIER snapshot
dev_l = np.array([logit(d["p_last"][0]) - logit(d["mu"][0]) for d in sub])
b2, _ = fit_logistic(np.column_stack([lgm, dev_l]), yh)
se2 = logistic_se(np.column_stack([lgm, dev_l]), b2)
print(f"variante dev_last : coef={b2[2]:+.3f} +- {se2[2]:.3f}")

# ================================================================== 3. FADE-OPEN PLEIN
print(f"\n{SEP}\n3. FADE deviation vs mu causale (pleine periode, AUCUNE selection)\n{SEP}")
def mc_pval(o, q, roi_obs, B=20000):
    o, q = np.array(o), np.array(q)
    wins = rng.random((B, len(o))) < q
    roi_sim = (wins * o).sum(axis=1) / len(o) - 1
    return float((roi_sim >= roi_obs).mean()), float(roi_sim.mean()), float(roi_sim.std())

def fade_eval(ds, thr, use_last_exec=False, sides=(0, 2)):
    bo, bq_self, bq_mu, ret, w = [], [], [], 0.0, 0
    for d in ds:
        pv = d["p_last"] if use_last_exec else d["p_open"]
        ov = d["o_last"] if use_last_exec else d["o_open"]
        dv = logit(pv[0]) - logit(d["mu"][0])
        side = None
        if dv < -thr and 0 in sides: side = 0
        elif dv > thr and 2 in sides: side = 2
        if side is None: continue
        o = ov[0] if side == 0 else ov[2]
        bo.append(o)
        bq_self.append(pv[0] if side == 0 else pv[2])
        bq_mu.append(d["mu"][0] if side == 0 else d["mu"][2])
        if d["y"] == side: ret += o; w += 1
    n = len(bo)
    if n < 20:
        print(f"   thr={thr} sides={sides} exec={'last' if use_last_exec else 'open'} : n={n} (trop petit)")
        return
    roi = (ret - n) / n
    p1, e1, s1 = mc_pval(bo, bq_self, roi)
    p2, e2, s2 = mc_pval(bo, bq_mu, roi)
    print(f"   thr={thr:.2f} sides={sides} exec={'last' if use_last_exec else 'open'} : "
          f"n={n:4d} roi={roi*100:+7.2f}% wr={w/n*100:5.1f}% cote={np.mean(bo):5.2f}")
    print(f"      null[truth=cotes]={e1*100:+6.2f}%+-{s1*100:.1f} p={p1:.4f} | "
          f"null[truth=mu]={e2*100:+6.2f}%+-{s2*100:.1f} p={p2:.4f}")

for thr in (0.03, 0.05, 0.07):
    fade_eval(sub, thr)
print()
for thr in (0.05,):
    fade_eval(sub, thr, sides=(0,))
    fade_eval(sub, thr, sides=(2,))
    fade_eval(sub, thr, use_last_exec=True)

# stabilite temporelle du fade-open 0.05 : ROI cumule par tiers
print("\n   stabilite par tiers (fade thr=0.05, 2 cotes) :")
n3 = len(sub) // 3
for i, part in enumerate((sub[:n3], sub[n3:2*n3], sub[2*n3:])):
    fade_eval(part, 0.05)

# ================================================================== 4. DRIFT PLEIN
print(f"\n{SEP}\n4. DRIFT INTRA-EVENT pleine periode (bet aux dernieres cotes)\n{SEP}")
dr_rows = [d for d in rows if d["gap"] >= 5]
print(f"events gap>=5min : {len(dr_rows)}")
def drift_eval(ds, thr, follow=True):
    bo, bq, ret, w = [], [], 0.0, 0
    for d in ds:
        dv = logit(d["p_last"][0]) - logit(d["p_open"][0])
        if dv > thr: side = 0 if follow else 2
        elif dv < -thr: side = 2 if follow else 0
        else: continue
        o = d["o_last"][0] if side == 0 else d["o_last"][2]
        bo.append(o); bq.append(d["p_last"][0] if side == 0 else d["p_last"][2])
        if d["y"] == side: ret += o; w += 1
    n = len(bo)
    if n < 20:
        print(f"   {'follow' if follow else 'fade  '} thr={thr} : n={n} (trop petit)"); return
    roi = (ret - n) / n
    p1, e1, s1 = mc_pval(bo, bq, roi)
    print(f"   {'follow' if follow else 'fade  '} thr={thr:.2f} : n={n:4d} roi={roi*100:+7.2f}% "
          f"wr={w/n*100:5.1f}% cote={np.mean(bo):5.2f}  null[truth=last] E={e1*100:+5.2f}% p={p1:.4f}")
for fol in (True, False):
    for thr in (0.03, 0.05):
        drift_eval(dr_rows, thr, fol)
print("\n   stabilite par tiers (follow thr=0.03) :")
n3 = len(dr_rows) // 3
for part in (dr_rows[:n3], dr_rows[n3:2*n3], dr_rows[2*n3:]):
    drift_eval(part, 0.03, True)

# ================================================================== 5. DECOMPOSITION VARIANCE
print(f"\n{SEP}\n5. DECOMPOSITION : jitter publication vs variance vraie inter-occurrences\n{SEP}")
# jitter publication (intra-event, en logit p_home)
jit = []
for eid, lst in snaps_by_ev.items():
    if len(lst) >= 5:
        x = np.array([logit(devig(s[1], s[2], s[3])[0]) for s in lst])
        jit.append(x.var(ddof=1))
sig2_jit = float(np.mean(jit))
# variance inter-occurrences (cotes d'ouverture, within-pair, en logit)
grp = defaultdict(list)
for d in rows:
    grp[(d["ta"], d["tb"])].append(logit(d["p_open"][0]))
btw = [np.var(v, ddof=1) for v in grp.values() if len(v) >= 4]
sig2_btw = float(np.mean(btw))
print(f"var jitter intra-event (logit p_home)        = {sig2_jit:.6f}  (std={math.sqrt(sig2_jit):.4f})")
print(f"var inter-occurrences within-pair (openings) = {sig2_btw:.6f}  (std={math.sqrt(sig2_btw):.4f})")
res = sig2_btw - sig2_jit
print(f"variance vraie residuelle (btw - jit)        = {res:+.6f}  "
      f"(std={math.sqrt(max(res,0)):.4f})")
print(f"part de la variance inter-occurrence expliquee par le jitter = "
      f"{min(sig2_jit/sig2_btw,1)*100:.1f}%")

print(f"\n{SEP}\nFIN\n{SEP}")
