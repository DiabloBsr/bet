# -*- coding: utf-8 -*-
"""WF4 - mu factorization phase 4: rigorous inference.
 1. Monte-Carlo p-values under null p_true=1/odds (proper for longshot portfolios)
    for the 70/30 test backtest and the rolling-origin pooled backtest.
 2. LR test b2 == b1 (is the residual really LESS followed than the factor part,
    vs global lambda compression).
 3. Robustness: exclude Sunderland (top deviant team) from shrink estimate + backtest.
Outputs exports/wf4_mufactor_phase4.json
"""
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict

MAXG = 15

def poisson_vec(lam):
    k = np.arange(MAXG + 1)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in range(MAXG + 1)])
    return np.exp(logp)

def probs_1x2(lh, la):
    grid = np.outer(poisson_vec(lh), poisson_vec(la))
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())

def poisson_glm(X, y, niter=80):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = np.zeros(X.shape[1]); b[0] = math.log(max(y.mean(), 0.1))
    for _ in range(niter):
        eta = np.clip(X @ b, -10, 5); mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-12:
            b = bn; break
        b = bn
    eta = np.clip(X @ b, -10, 5); mu = np.exp(eta)
    ll = float(np.sum(y * eta - mu))  # up to constant
    cov = np.linalg.inv((X.T * mu) @ X)
    return b, np.sqrt(np.diag(cov)), ll

def fit_factor(matches, teams):
    tidx = {t: i for i, t in enumerate(teams)}; T = len(teams)
    A = np.zeros((2 * len(matches), 2 + 2 * T)); y = np.zeros(2 * len(matches))
    for i, m in enumerate(matches):
        hi, ai = tidx[m["home"]], tidx[m["away"]]
        A[2 * i, 0] = 1; A[2 * i, 1] = 1; A[2 * i, 2 + hi] = 1; A[2 * i, 2 + T + ai] = -1
        y[2 * i] = math.log(m["lh"])
        A[2 * i + 1, 0] = 1; A[2 * i + 1, 2 + ai] = 1; A[2 * i + 1, 2 + T + hi] = -1
        y[2 * i + 1] = math.log(m["la"])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    return sol, tidx, T

def factor_lambdas(sol, tidx, T, home, away):
    hi, ai = tidx[home], tidx[away]
    return (math.exp(sol[0] + sol[1] + sol[2 + hi] - sol[2 + T + ai]),
            math.exp(sol[0] + sol[2 + ai] - sol[2 + T + hi]))

def pair_residuals(matches, sol, tidx, T):
    pr = defaultdict(lambda: [0.0, 0.0, 0])
    for m in matches:
        flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
        pr[(m["home"], m["away"])][0] += math.log(m["lh"]) - math.log(flh)
        pr[(m["home"], m["away"])][1] += math.log(m["la"]) - math.log(fla)
        pr[(m["home"], m["away"])][2] += 1
    return {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in pr.items()}

def mc_pvalue(bets, nsim=200000, seed=7):
    """bets: list of (odds, won). Null: P(win)=1/odds -> E[return]=0.
    Returns observed ROI, MC one-sided p, and win-count z."""
    rng = np.random.default_rng(seed)
    o = np.array([b[0] for b in bets]); w = np.array([b[1] for b in bets])
    p0 = 1.0 / o
    obs_roi = float((w * o - 1).mean())
    sims = (rng.random((nsim, len(o))) < p0) * o[None, :]
    roi_sims = (sims - 1).mean(axis=1)
    p_mc = float((roi_sims >= obs_roi).mean())
    ew, vw = p0.sum(), (p0 * (1 - p0)).sum()
    z_wins = (w.sum() - ew) / math.sqrt(vw)
    return obs_roi, p_mc, float(z_wins), int(w.sum()), float(ew)

def run_backtest(rows, train_frac_pairs, test_slice, teams, SH, SA, ev_min,
                 exclude_team=None):
    """train on rows[:cut], bet on test_slice."""
    tr = rows[:train_frac_pairs]
    sol, tidx, T = fit_factor(tr, teams)
    pres = {k: v for k, v in pair_residuals(tr, sol, tidx, T).items() if v[2] >= 3}
    bets = []
    for m in test_slice:
        if exclude_team and exclude_team in (m["home"], m["away"]):
            continue
        k = (m["home"], m["away"])
        if k not in pres:
            continue
        mrh, mra, _ = pres[k]
        flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
        lh = math.exp(math.log(flh) + SH * mrh); la = math.exp(math.log(fla) + SA * mra)
        pb = np.array(probs_1x2(lh, la))
        odds = np.array([m["oh"], m["od"], m["oa"]])
        evs = pb * odds - 1
        j = int(np.argmax(evs))
        if evs[j] >= ev_min:
            o = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
            bets.append((float(odds[j]), int(o == j)))
    return bets

def main():
    out = {}
    d = pickle.load(open("exports/_wf4_mufactor_data.pkl", "rb"))
    rows, ncut, teams = d["rows"], d["ncut"], d["teams"]
    train, test = rows[:ncut], rows[ncut:]
    pair_stats = {tuple(k.split("|")): v for k, v in d["pair_stats"].items()}
    SH, SA = 0.738, 0.762
    n = len(rows)

    # ---------- 1. MC p-values ----------
    print("=== 1. Monte-Carlo p-values (null: p_true = 1/odds) ===")
    for ev_min in [0.04, 0.06, 0.08]:
        bets = run_backtest(rows, ncut, test, teams, SH, SA, ev_min)
        roi, pmc, zw, W, EW = mc_pvalue(bets)
        avg_o = np.mean([b[0] for b in bets])
        print(f"70/30 TEST ev>={ev_min}: n={len(bets)} roi={roi*100:+.2f}% wins={W} vs E0={EW:.1f} "
              f"z={zw:+.2f} MC p={pmc:.5f} avgodds={avg_o:.2f}")
        out[f"mc_7030_ev{ev_min}"] = dict(n=len(bets), roi_pct=round(roi * 100, 2),
                                          wins=W, exp_wins_null=round(EW, 1),
                                          z_wins=round(zw, 2), p_mc=pmc,
                                          avg_odds=round(float(avg_o), 3))
    # rolling-origin pooled
    folds = [(0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.0)]
    pooled = []
    for ftr, fte in folds:
        pooled += run_backtest(rows, int(n * ftr), rows[int(n * ftr):int(n * fte)],
                               teams, SH, SA, 0.04)
    roi, pmc, zw, W, EW = mc_pvalue(pooled)
    avg_o = np.mean([b[0] for b in pooled])
    wr = np.mean([b[1] for b in pooled])
    print(f"ROLLING pooled ev>=0.04: n={len(pooled)} roi={roi*100:+.2f}% wr={wr:.3f} wins={W} vs E0={EW:.1f} "
          f"z={zw:+.2f} MC p={pmc:.5f} avgodds={avg_o:.2f}")
    out["mc_rolling_ev0.04"] = dict(n=len(pooled), roi_pct=round(roi * 100, 2), wr=round(float(wr), 4),
                                    wins=W, exp_wins_null=round(EW, 1), z_wins=round(zw, 2),
                                    p_mc=pmc, avg_odds=round(float(avg_o), 3))

    # ---------- 2. LR test b2 == b1 ----------
    print("\n=== 2. LR test: does the residual carry LESS weight than the factor part? ===")
    for side, lamk, resk, sk in [("home", "flh", "res_h", "mrh"), ("away", "fla", "res_a", "mra")]:
        Xu, Xc, Y = [], [], []
        for m in train:
            ps = pair_stats[(m["home"], m["away"])]
            if ps["n"] < 3:
                continue
            loo = (ps[sk] * ps["n"] - m[resk]) / (ps["n"] - 1)
            lf = math.log(m[lamk])
            Xu.append([1.0, lf, loo]); Xc.append([1.0, lf + loo])
            Y.append(m["sa"] if side == "home" else m["sb"])
        bu, seu, llu = poisson_glm(Xu, Y)
        bc, sec, llc = poisson_glm(Xc, Y)
        lr = 2 * (llu - llc)
        from math import erf
        p_lr = math.exp(-lr / 2) if lr > 0 else 1.0  # chi2_1 upper tail approx via survival
        # better chi2_1 survival:
        p_lr = 1 - math.erf(math.sqrt(max(lr, 0) / 2))
        print(f"{side}: unconstrained b1={bu[1]:.3f} b2={bu[2]:.3f} | constrained b={bc[1]:.3f} | "
              f"LR={lr:.2f} p(b2==b1)={p_lr:.4f}")
        out[f"lr_{side}"] = dict(b1=float(bu[1]), b2=float(bu[2]), b_constr=float(bc[1]),
                                 LR=float(lr), p=float(p_lr))

    # ---------- 3. robustness: exclude Sunderland ----------
    print("\n=== 3. Robustness: exclude Sunderland ===")
    # shrink re-estimate without Sunderland
    Xh, Yh = [], []
    for m in train:
        if "Sunderland" in (m["home"], m["away"]):
            continue
        ps = pair_stats[(m["home"], m["away"])]
        if ps["n"] < 3:
            continue
        loo = (ps["mrh"] * ps["n"] - m["res_h"]) / (ps["n"] - 1)
        Xh.append([1.0, math.log(m["flh"]), loo]); Yh.append(m["sa"])
    b, se, _ = poisson_glm(Xh, Yh)
    print(f"shrink home ex-Sunderland: b2={b[2]:.3f}+-{se[2]:.3f} (n={len(Yh)})")
    out["shrink_home_ex_sunderland"] = dict(b2=float(b[2]), se=float(se[2]), n=len(Yh))

    bets_ex = run_backtest(rows, ncut, test, teams, SH, SA, 0.04, exclude_team="Sunderland")
    if len(bets_ex) >= 10:
        roi, pmc, zw, W, EW = mc_pvalue(bets_ex)
        print(f"70/30 TEST ev>=0.04 ex-Sunderland: n={len(bets_ex)} roi={roi*100:+.2f}% "
              f"wins={W} vs E0={EW:.1f} z={zw:+.2f} MC p={pmc:.5f}")
        out["mc_7030_ev0.04_ex_sunderland"] = dict(n=len(bets_ex), roi_pct=round(roi * 100, 2),
                                                   wins=W, exp_wins_null=round(EW, 1),
                                                   z_wins=round(zw, 2), p_mc=pmc)
    pooled_ex = []
    for ftr, fte in folds:
        pooled_ex += run_backtest(rows, int(n * ftr), rows[int(n * ftr):int(n * fte)],
                                  teams, SH, SA, 0.04, exclude_team="Sunderland")
    roi, pmc, zw, W, EW = mc_pvalue(pooled_ex)
    print(f"ROLLING pooled ev>=0.04 ex-Sunderland: n={len(pooled_ex)} roi={roi*100:+.2f}% "
          f"wins={W} vs E0={EW:.1f} z={zw:+.2f} MC p={pmc:.5f}")
    out["mc_rolling_ev0.04_ex_sunderland"] = dict(n=len(pooled_ex), roi_pct=round(roi * 100, 2),
                                                  wins=W, exp_wins_null=round(EW, 1),
                                                  z_wins=round(zw, 2), p_mc=pmc)

    out["n_tests_scanned_phase4"] = 3 + 1 + 2 + 2
    with open("exports/wf4_mufactor_phase4.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nsaved exports/wf4_mufactor_phase4.json")

if __name__ == "__main__":
    main()
