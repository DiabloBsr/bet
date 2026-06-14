# -*- coding: utf-8 -*-
"""WF4 - mu factorization phase 2: walk-forward backtest on 8035.
Strategy: fade the pair residual. True lambdas estimated as
   log lam = log(factor lambda) + shrink * pair_residual(train)
with shrink estimated on TRAIN (LOO Poisson GLM). Bet 1X2 selections with
positive EV under blended probs at offered OPENING odds. TEST metrics only.
Outputs exports/wf4_mufactor.json
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
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()

def poisson_glm(X, y, niter=60):
    b = np.zeros(X.shape[1]); b[0] = math.log(max(y.mean(), 0.1))
    for _ in range(niter):
        eta = np.clip(X @ b, -10, 5); mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        b_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(b_new - b)) < 1e-10:
            b = b_new; break
        b = b_new
    mu = np.exp(np.clip(X @ b, -10, 5))
    cov = np.linalg.inv((X.T * mu) @ X)
    return b, np.sqrt(np.diag(cov))

def tstat_pvalue(returns):
    r = np.array(returns)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return None
    t = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))
    # one-sided p via normal approx
    from math import erf
    p = 1 - 0.5 * (1 + erf(t / math.sqrt(2)))
    return float(p)

def main():
    d = pickle.load(open("exports/_wf4_mufactor_data.pkl", "rb"))
    rows, ncut = d["rows"], d["ncut"]
    train, test = rows[:ncut], rows[ncut:]
    pair_stats = {tuple(k.split("|")): v for k, v in d["pair_stats"].items()}

    # ---- estimate shrink on TRAIN (LOO pair residual) for home and away ----
    def train_shrink(side):
        X, Y = [], []
        for m in train:
            ps = pair_stats.get((m["home"], m["away"]))
            if ps is None or ps["n"] < 3:
                continue
            if side == "h":
                loo = (ps["mrh"] * ps["n"] - m["res_h"]) / (ps["n"] - 1)
                X.append([1.0, math.log(m["flh"]), loo]); Y.append(m["sa"])
            else:
                loo = (ps["mra"] * ps["n"] - m["res_a"]) / (ps["n"] - 1)
                X.append([1.0, math.log(m["fla"]), loo]); Y.append(m["sb"])
        b, se = poisson_glm(np.array(X), np.array(Y, dtype=float))
        return b, se, len(Y)

    bh, seh, nh = train_shrink("h")
    ba, sea, na = train_shrink("a")
    print(f"TRAIN shrink home: b2={bh[2]:.3f}+-{seh[2]:.3f} (b0={bh[0]:.3f}, b1={bh[1]:.3f}, n={nh})")
    print(f"TRAIN shrink away: b2={ba[2]:.3f}+-{sea[2]:.3f} (b0={ba[0]:.3f}, b1={ba[1]:.3f}, n={na})")
    shrink_h, shrink_a = float(bh[2]), float(ba[2])

    # ---- build per-test-match probabilities ----
    # published probs (margin removed)
    # blended probs: lambda = exp(log flh + shrink*pair_res) ; pure factor: shrink=0
    test_use = []
    for m in test:
        ps = pair_stats.get((m["home"], m["away"]))
        if ps is None or ps["n"] < 3:
            continue
        inv = np.array([1 / m["oh"], 1 / m["od"], 1 / m["oa"]])
        ppub = inv / inv.sum()
        lh_b = math.exp(math.log(m["flh"]) + shrink_h * ps["mrh"])
        la_b = math.exp(math.log(m["fla"]) + shrink_a * ps["mra"])
        pb = probs_1x2(lh_b, la_b)
        lf, lfa = m["flh"], m["fla"]
        pf = probs_1x2(lf, lfa)
        out = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
        test_use.append(dict(m=m, ppub=ppub, pblend=np.array(pb), pfact=np.array(pf),
                             out=out, res_pair=(ps["mrh"], ps["mra"]),
                             odds=np.array([m["oh"], m["od"], m["oa"]])))
    print(f"test matches usable (pair seen >=3x in train): {len(test_use)} / {len(test)}")

    # ---- calibration: log-loss published vs blended vs factor on test ----
    ll_pub = -np.mean([math.log(max(t["ppub"][t["out"]], 1e-9)) for t in test_use])
    ll_bl = -np.mean([math.log(max(t["pblend"][t["out"]], 1e-9)) for t in test_use])
    ll_fa = -np.mean([math.log(max(t["pfact"][t["out"]], 1e-9)) for t in test_use])
    print(f"TEST 1X2 log-loss: published={ll_pub:.5f}  blended={ll_bl:.5f}  factor={ll_fa:.5f}")
    # paired diff significance (published - blended per match)
    diffs = [math.log(max(t["pblend"][t["out"]], 1e-9)) - math.log(max(t["ppub"][t["out"]], 1e-9))
             for t in test_use]
    pcal = tstat_pvalue(diffs)
    print(f"blended beats published per-match: mean dll={np.mean(diffs):+.5f}, one-sided p={pcal:.4g}")

    # ---- backtest grid ----
    results = []
    configs = []
    for model in ["blend", "factor"]:
        for ev_min in [0.00, 0.02, 0.04, 0.06, 0.08]:
            for omin in [1.0, 1.6, 2.0]:
                configs.append((model, ev_min, omin))
    n_tests = len(configs)

    for model, ev_min, omin in configs:
        rets, odds_l, wins, sels = [], [], 0, []
        for t in test_use:
            p = t["pblend"] if model == "blend" else t["pfact"]
            evs = p * t["odds"] - 1
            j = int(np.argmax(evs))
            if evs[j] >= ev_min and t["odds"][j] >= omin:
                won = (t["out"] == j)
                rets.append(t["odds"][j] - 1 if won else -1.0)
                odds_l.append(t["odds"][j]); wins += won; sels.append(j)
        n = len(rets)
        if n == 0:
            continue
        roi = float(np.mean(rets)) * 100
        pv = tstat_pvalue(rets)
        results.append(dict(model=model, ev_min=ev_min, odds_min=omin, n=n,
                            roi_pct=round(roi, 2), wr=round(wins / n, 4),
                            avg_odds=round(float(np.mean(odds_l)), 3),
                            pvalue=(round(pv, 6) if pv is not None else None),
                            sel_mix={s: sels.count(s) for s in set(sels)}))
    print(f"\n--- backtest grid (n_tests_scanned={n_tests}) ---")
    for r in sorted(results, key=lambda x: -x["roi_pct"]):
        print(f"{r['model']:6s} ev>={r['ev_min']:.2f} odds>={r['odds_min']:.1f} | n={r['n']:4d} "
              f"roi={r['roi_pct']:+6.2f}% wr={r['wr']:.3f} avgodds={r['avg_odds']:.2f} p={r['pvalue']} mix={r['sel_mix']}")

    with open("exports/wf4_mufactor.json", "w", encoding="utf-8") as f:
        json.dump(dict(shrink_h=shrink_h, shrink_a=shrink_a,
                       shrink_h_se=float(seh[2]), shrink_a_se=float(sea[2]),
                       logloss=dict(published=ll_pub, blended=ll_bl, factor=ll_fa,
                                    p_blend_beats_pub=pcal),
                       n_test_usable=len(test_use), n_tests_scanned=n_tests,
                       grid=results), f, indent=1)
    print("\nsaved exports/wf4_mufactor.json")

if __name__ == "__main__":
    main()
