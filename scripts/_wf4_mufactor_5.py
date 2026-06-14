# -*- coding: utf-8 -*-
"""WF4 - mu factorization phase 5: ATTRIBUTION.
Same betting machinery (Poisson grid, EV>=0.04 vs opening 1X2 odds), rolling-origin
folds, but different lambda models, all train-fit only:
 V1 pairmu   : lambda = pair-mean published lambda (s=1)        -> pure jitter edge
 V2 factor   : lambda = factor lambda (s=0)
 V3 blend    : log lam = log flh + 0.74/0.76 * res_pair         -> phase-2/4 strategy
 V4 recal_pub: log lam = a + c*(log flh + res_pair)             -> global recalibration of pair mu
 V5 recal_bl : log lam = a + c*log flh + s*res_pair (a,c,s GLM) -> full statistical model
Each also run ex-Sunderland. MC p-values under p_true=1/odds.
Outputs exports/wf4_mufactor_phase5.json
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
    return b

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

def mc_pvalue(bets, nsim=200000, seed=7):
    rng = np.random.default_rng(seed)
    o = np.array([b[0] for b in bets]); w = np.array([b[1] for b in bets])
    p0 = 1.0 / o
    obs_roi = float((w * o - 1).mean())
    hits = 0
    chunk = max(1, int(2e7 / max(len(o), 1)))
    done = 0
    while done < nsim:
        k = min(chunk, nsim - done)
        sims = (rng.random((k, len(o))) < p0) * o[None, :]
        hits += int(((sims - 1).mean(axis=1) >= obs_roi).sum())
        done += k
    p_mc = hits / nsim
    z = (w.sum() - p0.sum()) / math.sqrt((p0 * (1 - p0)).sum())
    return obs_roi, p_mc, float(z)

def main():
    d = pickle.load(open("exports/_wf4_mufactor_data.pkl", "rb"))
    rows, teams = d["rows"], d["teams"]
    n = len(rows)
    folds = [(0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.0)]
    EVMIN = 0.04

    variants = ["V1_pairmu", "V2_factor", "V3_blend", "V4_recal_pub", "V5_recal_blend"]
    book = {v: [] for v in variants}
    book_ex = {v: [] for v in variants}
    bet_log = {v: [] for v in variants}

    for ftr, fte in folds:
        tr = rows[:int(n * ftr)]; te = rows[int(n * ftr):int(n * fte)]
        sol, tidx, T = fit_factor(tr, teams)
        # pair stats on train
        pr = defaultdict(lambda: [0.0, 0.0, 0])
        per_match = {}
        for m in tr:
            flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
            per_match[m["id"]] = (flh, fla)
            k = (m["home"], m["away"])
            pr[k][0] += math.log(m["lh"]) - math.log(flh)
            pr[k][1] += math.log(m["la"]) - math.log(fla)
            pr[k][2] += 1
        pres = {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in pr.items() if v[2] >= 3}
        # GLMs on train (LOO residual) for V4/V5 coefs and V3 shrink
        Xh5, Xa5, Yh, Ya = [], [], [], []
        for m in tr:
            k = (m["home"], m["away"])
            if k not in pres or pres[k][2] < 3:
                continue
            mrh, mra, cnt = pres[k]
            flh, fla = per_match[m["id"]]
            looh = (mrh * cnt - (math.log(m["lh"]) - math.log(flh))) / (cnt - 1)
            looa = (mra * cnt - (math.log(m["la"]) - math.log(fla))) / (cnt - 1)
            Xh5.append([1.0, math.log(flh), looh]); Yh.append(m["sa"])
            Xa5.append([1.0, math.log(fla), looa]); Ya.append(m["sb"])
        bh5 = poisson_glm(Xh5, Yh)   # [a, c, s] home
        ba5 = poisson_glm(Xa5, Ya)
        bh4 = poisson_glm([[x[0], x[1] + x[2]] for x in Xh5], Yh)  # [a, c] on full pair mu
        ba4 = poisson_glm([[x[0], x[1] + x[2]] for x in Xa5], Ya)

        for m in te:
            k = (m["home"], m["away"])
            if k not in pres:
                continue
            mrh, mra, cnt = pres[k]
            flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
            lfh, lfa = math.log(flh), math.log(fla)
            lam = {
                "V1_pairmu": (math.exp(lfh + mrh), math.exp(lfa + mra)),
                "V2_factor": (flh, fla),
                "V3_blend": (math.exp(lfh + 0.738 * mrh), math.exp(lfa + 0.762 * mra)),
                "V4_recal_pub": (math.exp(bh4[0] + bh4[1] * (lfh + mrh)),
                                 math.exp(ba4[0] + ba4[1] * (lfa + mra))),
                "V5_recal_blend": (math.exp(bh5[0] + bh5[1] * lfh + bh5[2] * mrh),
                                   math.exp(ba5[0] + ba5[1] * lfa + ba5[2] * mra)),
            }
            odds = np.array([m["oh"], m["od"], m["oa"]])
            o = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
            for v in variants:
                pb = np.array(probs_1x2(*lam[v]))
                evs = pb * odds - 1
                j = int(np.argmax(evs))
                if evs[j] >= EVMIN:
                    bet = (float(odds[j]), int(o == j))
                    book[v].append(bet)
                    bet_log[v].append(dict(pair=k, sel=j, odd=float(odds[j]), won=int(o == j)))
                    if "Sunderland" not in k:
                        book_ex[v].append(bet)

    out = {}
    print(f"=== rolling-origin pooled, EV>={EVMIN}, 1X2 opening odds ===")
    for v in variants:
        for tag, b in [("all", book[v]), ("exSund", book_ex[v])]:
            if len(b) < 10:
                print(f"{v:15s} [{tag}] n={len(b)} (too few)")
                continue
            roi, pmc, z = mc_pvalue(b)
            avg_o = float(np.mean([x[0] for x in b]))
            wr = float(np.mean([x[1] for x in b]))
            npairs = len(set((bl["pair"], bl["sel"]) for bl in bet_log[v])) if tag == "all" else None
            print(f"{v:15s} [{tag:6s}] n={len(b):4d} roi={roi*100:+7.2f}% wr={wr:.3f} "
                  f"avgodds={avg_o:5.2f} z={z:+5.2f} MCp={pmc:.5f}" +
                  (f" distinct(pair,sel)={npairs}" if npairs else ""))
            out[f"{v}_{tag}"] = dict(n=len(b), roi_pct=round(roi * 100, 2), wr=round(wr, 4),
                                     avg_odds=round(avg_o, 3), z=round(z, 2), p_mc=pmc)

    # selection overlap V3 vs V1 (does blend just re-find jitter bets?)
    s3 = set((bl["pair"], bl["sel"]) for bl in bet_log["V3_blend"])
    s1 = set((bl["pair"], bl["sel"]) for bl in bet_log["V1_pairmu"])
    print(f"\noverlap (pair,sel): V3 {len(s3)} vs V1 {len(s1)}, common {len(s3 & s1)}")
    out["overlap_V3_V1"] = dict(v3=len(s3), v1=len(s1), common=len(s3 & s1))
    out["n_tests_scanned_phase5"] = len(variants) * 2

    with open("exports/wf4_mufactor_phase5.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("saved exports/wf4_mufactor_phase5.json")

if __name__ == "__main__":
    main()
