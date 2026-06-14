# -*- coding: utf-8 -*-
"""WF4 - mu factorization by pair (8035).
Phase 1: load opening 1X2 odds, invert (lh, la), fit attack/defense factor model
(Dixon-Coles without correlation), diagnostics on pair residuals.
Caches per-event data to exports/_wf4_mufactor_data.pkl for phase 2.
READ-ONLY on DB.
"""
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from scraper.config import load_settings
from sqlalchemy import create_engine, text

MAXG = 15  # Poisson grid size

def poisson_vec(lam, maxg=MAXG):
    k = np.arange(maxg + 1)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in range(maxg + 1)])
    return np.exp(logp)

def probs_1x2(lh, la):
    ph_ = poisson_vec(lh); pa_ = poisson_vec(la)
    grid = np.outer(ph_, pa_)
    p_home = np.tril(grid, -1).sum()
    p_draw = np.trace(grid)
    p_away = np.triu(grid, 1).sum()
    return p_home, p_draw, p_away

_inv_cache = {}
def invert_lambdas(oh, od, oa):
    """Invert (lh, la) from 1X2 odds. Normalize implied probs to sum 1 (flat 6% margin)."""
    key = (round(oh, 4), round(od, 4), round(oa, 4))
    if key in _inv_cache:
        return _inv_cache[key]
    inv = np.array([1.0 / oh, 1.0 / od, 1.0 / oa])
    p = inv / inv.sum()
    ph_t, pa_t = p[0], p[2]
    # 2D Newton on (log lh, log la)
    x = np.array([math.log(1.5), math.log(1.1)])
    ok = False
    for _ in range(60):
        lh, la = math.exp(x[0]), math.exp(x[1])
        Ph, Pd, Pa = probs_1x2(lh, la)
        f = np.array([Ph - ph_t, Pa - pa_t])
        if abs(f[0]) < 1e-10 and abs(f[1]) < 1e-10:
            ok = True
            break
        eps = 1e-5
        J = np.zeros((2, 2))
        for j in range(2):
            x2 = x.copy(); x2[j] += eps
            Ph2, _, Pa2 = probs_1x2(math.exp(x2[0]), math.exp(x2[1]))
            J[0, j] = (Ph2 - Ph) / eps
            J[1, j] = (Pa2 - Pa) / eps
        try:
            dx = np.linalg.solve(J, f)
        except np.linalg.LinAlgError:
            break
        dx = np.clip(dx, -1.0, 1.0)
        x = x - dx
        x = np.clip(x, math.log(0.05), math.log(6.0))
    lh, la = math.exp(x[0]), math.exp(x[1])
    res = (lh, la, ok)
    _inv_cache[key] = res
    return res

def main():
    corrupted = set()
    with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
        d = json.load(f)
        corrupted = set(int(k) for k in d["events"].keys())
    print(f"corrupted ids: {len(corrupted)}")

    e = create_engine(load_settings().db_url)
    q = """
    SELECT ev.id, ev.team_a, ev.team_b, ev.expected_start, ev.round_info,
           r.score_a, r.score_b,
           o.odds_home, o.odds_draw, o.odds_away
    FROM events ev
    JOIN results r ON r.event_id = ev.id
    JOIN odds_snapshots o ON o.event_id = ev.id
      AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
    WHERE ev.competition = 'InstantLeague-8035'
    ORDER BY ev.expected_start, ev.id
    """
    rows = []
    with e.connect() as c:
        for r in c.execute(text(q)):
            if r[0] in corrupted:
                continue
            if r[7] is None or r[8] is None or r[9] is None:
                continue
            if r[7] <= 1.0 or r[8] <= 1.0 or r[9] <= 1.0:
                continue
            rows.append(dict(id=r[0], home=r[1], away=r[2], start=r[3], rnd=r[4],
                             sa=int(r[5]), sb=int(r[6]),
                             oh=float(r[7]), od=float(r[8]), oa=float(r[9])))
    print(f"clean matches with opening odds: {len(rows)}")

    # invert lambdas
    nfail = 0
    for m in rows:
        lh, la, ok = invert_lambdas(m["oh"], m["od"], m["oa"])
        m["lh"], m["la"], m["inv_ok"] = lh, la, ok
        if not ok:
            nfail += 1
    print(f"inversion failures: {nfail} / cache size {len(_inv_cache)}")
    rows = [m for m in rows if m["inv_ok"]]

    # sanity: re-derive 1X2 probs from inverted lambdas vs normalized implied
    errs = []
    for m in rows[:300]:
        Ph, Pd, Pa = probs_1x2(m["lh"], m["la"])
        inv = np.array([1 / m["oh"], 1 / m["od"], 1 / m["oa"]]); p = inv / inv.sum()
        errs.append(abs(Ph - p[0]) + abs(Pd - p[1]) + abs(Pa - p[2]))
    print(f"inversion sanity max abs err (300): {max(errs):.2e}")

    teams = sorted(set(m["home"] for m in rows) | set(m["away"] for m in rows))
    tidx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    print(f"teams: {T}")

    # 70/30 temporal split
    n = len(rows)
    ncut = int(n * 0.7)
    train, test = rows[:ncut], rows[ncut:]
    print(f"train {len(train)} (last start {train[-1]['start']}), test {len(test)} (first start {test[0]['start']})")

    # ---- fit factor model on train: least squares on log lambda ----
    # params: [c, h, att_0..att_T-1, def_0..def_T-1]
    def fit_factor(matches):
        nr = 2 * len(matches)
        npar = 2 + 2 * T
        A = np.zeros((nr, npar)); y = np.zeros(nr)
        for i, m in enumerate(matches):
            hi, ai = tidx[m["home"]], tidx[m["away"]]
            # log lh = c + h + att_home - def_away
            A[2 * i, 0] = 1; A[2 * i, 1] = 1
            A[2 * i, 2 + hi] = 1; A[2 * i, 2 + T + ai] = -1
            y[2 * i] = math.log(m["lh"])
            # log la = c + att_away - def_home
            A[2 * i + 1, 0] = 1
            A[2 * i + 1, 2 + ai] = 1; A[2 * i + 1, 2 + T + hi] = -1
            y[2 * i + 1] = math.log(m["la"])
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ sol
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return sol, 1 - ss_res / ss_tot, math.sqrt(ss_res / nr)

    sol_tr, r2_tr, rmse_tr = fit_factor(train)
    print(f"TRAIN factor fit: R2={r2_tr:.5f} rmse(log-lambda)={rmse_tr:.4f}")
    c0, hadv = sol_tr[0], sol_tr[1]
    att = sol_tr[2:2 + T]; deff = sol_tr[2 + T:]
    att -= att.mean()  # display only
    print(f"base c={c0:.4f} home_adv={hadv:.4f}")

    def factor_lambdas(sol, home, away):
        hi, ai = tidx[home], tidx[away]
        llh = sol[0] + sol[1] + sol[2 + hi] - sol[2 + T + ai]
        lla = sol[0] + sol[2 + ai] - sol[2 + T + hi]
        return math.exp(llh), math.exp(lla)

    # per-match residuals (train, in log space)
    for m in rows:
        flh, fla = factor_lambdas(sol_tr, m["home"], m["away"])
        m["flh"], m["fla"] = flh, fla
        m["res_h"] = math.log(m["lh"]) - math.log(flh)
        m["res_a"] = math.log(m["la"]) - math.log(fla)

    # ---- pair-level residual stability (train) ----
    from collections import defaultdict
    pair_tr = defaultdict(list)
    for m in train:
        pair_tr[(m["home"], m["away"])].append(m)
    print(f"ordered pairs in train: {len(pair_tr)}, occurrences/pair median "
          f"{np.median([len(v) for v in pair_tr.values()]):.0f}")

    # variance decomposition of residual: between-pair vs within-pair (jitter)
    res_h_all = np.array([m["res_h"] for m in train])
    res_a_all = np.array([m["res_a"] for m in train])
    within_h, within_a, between_h, between_a = [], [], [], []
    pair_stats = {}
    for k, v in pair_tr.items():
        rh = np.array([m["res_h"] for m in v]); ra = np.array([m["res_a"] for m in v])
        pair_stats[k] = dict(n=len(v), mrh=rh.mean(), mra=ra.mean(),
                             sdh=rh.std(ddof=1) if len(v) > 1 else np.nan)
        if len(v) > 1:
            within_h.append(rh.std(ddof=1)); within_a.append(ra.std(ddof=1))
        between_h.append(rh.mean()); between_a.append(ra.mean())
    print(f"residual log-lh: total std {res_h_all.std():.4f} | within-pair (jitter) "
          f"{np.nanmean(within_h):.4f} | between-pair std {np.std(between_h):.4f}")
    print(f"residual log-la: total std {res_a_all.std():.4f} | within-pair "
          f"{np.nanmean(within_a):.4f} | between-pair std {np.std(between_a):.4f}")

    # split-half reliability of pair residual within TRAIN (is it persistent?)
    rng = np.random.default_rng(42)
    h1h, h2h, h1a, h2a = [], [], [], []
    for k, v in pair_tr.items():
        if len(v) < 4:
            continue
        idx = rng.permutation(len(v))
        half = len(v) // 2
        g1 = [v[i] for i in idx[:half]]; g2 = [v[i] for i in idx[half:]]
        h1h.append(np.mean([m["res_h"] for m in g1])); h2h.append(np.mean([m["res_h"] for m in g2]))
        h1a.append(np.mean([m["res_a"] for m in g1])); h2a.append(np.mean([m["res_a"] for m in g2]))
    if len(h1h) > 10:
        ch = np.corrcoef(h1h, h2h)[0, 1]; ca = np.corrcoef(h1a, h2a)[0, 1]
        print(f"split-half reliability of pair residual (train, {len(h1h)} pairs): "
              f"res_h r={ch:.3f}, res_a r={ca:.3f}")

    # ---- KEY DIAGNOSTIC: do actual goals follow published mu or factorized mu? ----
    # pair-level regression on TEST matches using TRAIN-estimated pair residual:
    # actual goals_h ~ a + b1*log(flh) + b2*pair_res_h  -> b2~1: engine sims published mu
    #                                                      b2~0: engine sims factorized mu
    Xh, Yh, Xa, Ya = [], [], [], []
    for m in test:
        k = (m["home"], m["away"])
        if k not in pair_stats or pair_stats[k]["n"] < 3:
            continue
        ps = pair_stats[k]
        Xh.append([1.0, math.log(m["flh"]), ps["mrh"]]); Yh.append(m["sa"])
        Xa.append([1.0, math.log(m["fla"]), ps["mra"]]); Ya.append(m["sb"])
    Xh = np.array(Xh); Yh = np.array(Yh, dtype=float)
    Xa = np.array(Xa); Ya = np.array(Ya, dtype=float)
    print(f"\nKEY TEST (poisson-ish OLS on log link approx), n_test={len(Yh)}")

    # Poisson GLM via IRLS, log link: E[goals] = exp(b0 + b1*log(flh) + b2*res_pair)
    def poisson_glm(X, y, niter=50):
        b = np.zeros(X.shape[1]); b[0] = math.log(max(y.mean(), 0.1))
        for _ in range(niter):
            eta = X @ b
            mu = np.exp(np.clip(eta, -10, 5))
            W = mu
            z = eta + (y - mu) / mu
            XtW = X.T * W
            b_new = np.linalg.solve(XtW @ X, XtW @ z)
            if np.max(np.abs(b_new - b)) < 1e-10:
                b = b_new; break
            b = b_new
        eta = X @ b; mu = np.exp(np.clip(eta, -10, 5))
        XtW = X.T * mu
        cov = np.linalg.inv(XtW @ X)
        se = np.sqrt(np.diag(cov))
        return b, se

    bh, seh = poisson_glm(Xh, Yh)
    ba, sea = poisson_glm(Xa, Ya)
    print(f"HOME goals: b(log flh)={bh[1]:.3f}+-{seh[1]:.3f}  b(pair_res)={bh[2]:.3f}+-{seh[2]:.3f}")
    print(f"AWAY goals: b(log fla)={ba[1]:.3f}+-{sea[1]:.3f}  b(pair_res)={ba[2]:.3f}+-{sea[2]:.3f}")
    print("interpretation: b(pair_res)~1 => results follow PUBLISHED mu; ~0 => follow FACTORIZED mu")

    # same on TRAIN (in-sample, pair residual leave-self caveat) for power check
    Xh2, Yh2 = [], []
    for m in train:
        k = (m["home"], m["away"])
        ps = pair_stats[k]
        if ps["n"] < 3:
            continue
        # leave-one-out residual to avoid self-contamination
        loo = (ps["mrh"] * ps["n"] - m["res_h"]) / (ps["n"] - 1)
        Xh2.append([1.0, math.log(m["flh"]), loo]); Yh2.append(m["sa"])
    bh2, seh2 = poisson_glm(np.array(Xh2), np.array(Yh2, dtype=float))
    print(f"TRAIN(LOO) HOME goals: b(pair_res)={bh2[2]:.3f}+-{seh2[2]:.3f} (n={len(Yh2)})")

    with open("exports/_wf4_mufactor_data.pkl", "wb") as f:
        pickle.dump(dict(rows=rows, ncut=ncut, teams=teams, sol_tr=sol_tr.tolist(),
                         pair_stats={f"{k[0]}|{k[1]}": v for k, v in pair_stats.items()},
                         r2_tr=r2_tr, rmse_tr=rmse_tr), f)
    print("\nsaved exports/_wf4_mufactor_data.pkl")

if __name__ == "__main__":
    main()
