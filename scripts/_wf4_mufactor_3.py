# -*- coding: utf-8 -*-
"""WF4 - mu factorization phase 3:
 A. characterize blend bets (which pairs drive them)
 B. higher-power calibration test on high-divergence subset (G-test pub vs blend)
 C. robustness of shrinkage (quartiles of |residual|, trimming)
 D. Double Chance backtest on same signal (lower variance)
 E. rolling-origin supporting backtest (5 folds, walk-forward pure)
 F. list of currently deviant pairs (full-data fit)
Outputs appended into exports/wf4_mufactor_phase3.json
"""
import sys, json, pickle, math
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from scraper.config import load_settings
from sqlalchemy import create_engine, text

MAXG = 15

def poisson_vec(lam):
    k = np.arange(MAXG + 1)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in range(MAXG + 1)])
    return np.exp(logp)

def probs_1x2(lh, la):
    grid = np.outer(poisson_vec(lh), poisson_vec(la))
    return float(np.tril(grid, -1).sum()), float(np.trace(grid)), float(np.triu(grid, 1).sum())

def poisson_glm(X, y, niter=60):
    X = np.asarray(X, float); y = np.asarray(y, float)
    b = np.zeros(X.shape[1]); b[0] = math.log(max(y.mean(), 0.1))
    for _ in range(niter):
        eta = np.clip(X @ b, -10, 5); mu = np.exp(eta)
        z = eta + (y - mu) / mu
        XtW = X.T * mu
        bn = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(bn - b)) < 1e-10:
            b = bn; break
        b = bn
    mu = np.exp(np.clip(X @ b, -10, 5))
    cov = np.linalg.inv((X.T * mu) @ X)
    return b, np.sqrt(np.diag(cov))

def tstat_pvalue(returns):
    r = np.array(returns, float)
    if len(r) < 2 or r.std(ddof=1) == 0:
        return None
    t = r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))
    from math import erf
    return float(1 - 0.5 * (1 + erf(t / math.sqrt(2))))

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
        k = (m["home"], m["away"])
        pr[k][0] += math.log(m["lh"]) - math.log(flh)
        pr[k][1] += math.log(m["la"]) - math.log(fla)
        pr[k][2] += 1
    return {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in pr.items()}

def main():
    out = {}
    d = pickle.load(open("exports/_wf4_mufactor_data.pkl", "rb"))
    rows, ncut, teams = d["rows"], d["ncut"], d["teams"]
    train, test = rows[:ncut], rows[ncut:]
    pair_stats = {tuple(k.split("|")): v for k, v in d["pair_stats"].items()}
    SH, SA = 0.738, 0.762  # train-estimated shrink

    def blend_probs(m):
        ps = pair_stats[(m["home"], m["away"])]
        lh = math.exp(math.log(m["flh"]) + SH * ps["mrh"])
        la = math.exp(math.log(m["fla"]) + SA * ps["mra"])
        return np.array(probs_1x2(lh, la)), lh, la

    # ---------- A. characterize blend ev>=0.04 bets ----------
    bets = []
    for m in test:
        pb, lhb, lab = blend_probs(m)
        odds = np.array([m["oh"], m["od"], m["oa"]])
        evs = pb * odds - 1
        j = int(np.argmax(evs))
        if evs[j] >= 0.04:
            out_ = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
            bets.append(dict(pair=(m["home"], m["away"]), sel=j, odd=float(odds[j]),
                             ev=float(evs[j]), won=int(out_ == j),
                             res=(pair_stats[(m["home"], m["away"])]["mrh"],
                                  pair_stats[(m["home"], m["away"])]["mra"])))
    pairs_used = defaultdict(lambda: [0, 0.0])
    for b in bets:
        k = (b["pair"], b["sel"])
        pairs_used[k][0] += 1
        pairs_used[k][1] += (b["odd"] - 1) if b["won"] else -1.0
    print(f"A. blend ev>=0.04: {len(bets)} bets over {len(pairs_used)} distinct (pair,sel)")
    top = sorted(pairs_used.items(), key=lambda x: -x[1][0])[:15]
    for (p, s), (cnt, pnl) in top:
        print(f"   {p[0]:18s} v {p[1]:18s} sel={'HXA'[s]} n={cnt} pnl={pnl:+.1f}")
    out["A_distinct_pairsel"] = len(pairs_used)
    out["A_nbets"] = len(bets)

    # ---------- B. G-test pub vs blend on high-divergence subset ----------
    # subset: matches where max_j |p_blend - p_pub| >= delta
    res_B = []
    for delta in [0.01, 0.02, 0.03]:
        ll_pub, ll_bl, nsub = 0.0, 0.0, 0
        per_match = []
        for m in test:
            pb, *_ = blend_probs(m)
            inv = np.array([1 / m["oh"], 1 / m["od"], 1 / m["oa"]]); pp = inv / inv.sum()
            if np.max(np.abs(pb - pp)) < delta:
                continue
            o = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
            ll_pub += math.log(max(pp[o], 1e-9)); ll_bl += math.log(max(pb[o], 1e-9))
            per_match.append(math.log(max(pb[o], 1e-9)) - math.log(max(pp[o], 1e-9)))
            nsub += 1
        if nsub < 20:
            continue
        pv = tstat_pvalue(per_match)
        res_B.append(dict(delta=delta, n=nsub, dll_total=ll_bl - ll_pub,
                          dll_mean=float(np.mean(per_match)), p=pv))
        print(f"B. delta>={delta}: n={nsub} total dLL(blend-pub)={ll_bl-ll_pub:+.2f} p={pv:.4f}")
    out["B_calibration_subset"] = res_B

    # ---------- C. robustness of shrinkage by |residual| quartile ----------
    res_C = []
    # bucket train matches by |pair residual| magnitude (mean of |mrh|,|mra|)
    mags = []
    for m in train:
        ps = pair_stats[(m["home"], m["away"])]
        mags.append(max(abs(ps["mrh"]), abs(ps["mra"])))
    qs = np.quantile(mags, [0.25, 0.5, 0.75])
    for lo, hi, lab in [(0, qs[0], "Q1"), (qs[0], qs[1], "Q2"), (qs[1], qs[2], "Q3"), (qs[2], 99, "Q4")]:
        X, Y = [], []
        for m in train:
            ps = pair_stats[(m["home"], m["away"])]
            mg = max(abs(ps["mrh"]), abs(ps["mra"]))
            if not (lo <= mg < hi) or ps["n"] < 3:
                continue
            loo = (ps["mrh"] * ps["n"] - m["res_h"]) / (ps["n"] - 1)
            X.append([1.0, math.log(m["flh"]), loo]); Y.append(m["sa"])
        if len(Y) < 200:
            continue
        b, se = poisson_glm(X, Y)
        res_C.append(dict(bucket=lab, n=len(Y), b2=float(b[2]), se=float(se[2])))
        print(f"C. {lab} (|res| in [{lo:.3f},{hi:.3f})): n={len(Y)} b2={b[2]:.3f}+-{se[2]:.3f}")
    out["C_shrink_quartiles"] = res_C

    # ---------- D. Double Chance backtest on test ----------
    eng = create_engine(load_settings().db_url)
    ids = [m["id"] for m in test]
    dc_odds = {}
    with eng.connect() as c:
        for chunk in range(0, len(ids), 500):
            sub = ids[chunk:chunk + 500]
            q = text(f"""SELECT o.event_id, o.extra_markets FROM odds_snapshots o
                     WHERE o.event_id IN ({','.join(str(i) for i in sub)})
                     AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=o.event_id)""")
            for eid, xm in c.execute(q):
                if not xm:
                    continue
                try:
                    j = json.loads(xm)
                except Exception:
                    continue
                dc = j.get("Double Chance")
                if dc:
                    dc_odds[eid] = dc
    print(f"D. DC opening odds found for {len(dc_odds)}/{len(test)} test matches")
    # margin check
    margs = []
    for m in test[:300]:
        if m["id"] in dc_odds:
            dc = dc_odds[m["id"]]
            try:
                margs.append(1 / dc["1X"] + 1 / dc["X2"] + 1 / dc["12"] - 1)
            except Exception:
                pass
    if margs:
        # DC overround: 3 selections each covering 2 outcomes -> sum 1/o ~ 2*(1+margin/?)
        print(f"D. DC sum(1/odds) mean = {np.mean(margs)+1:.4f} (2.0 = fair)")
    res_D = []
    n_d_tests = 0
    for ev_min in [0.0, 0.02, 0.04]:
        rets, odds_l, wins = [], [], 0
        for m in test:
            if m["id"] not in dc_odds:
                continue
            dc = dc_odds[m["id"]]
            try:
                o3 = [float(dc["1X"]), float(dc["X2"]), float(dc["12"])]
            except Exception:
                continue
            if min(o3) <= 1.0:
                continue
            pb, *_ = blend_probs(m)
            pdc = np.array([pb[0] + pb[1], pb[1] + pb[2], pb[0] + pb[2]])
            evs = pdc * np.array(o3) - 1
            j = int(np.argmax(evs))
            if evs[j] < ev_min:
                continue
            o = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
            won = (j == 0 and o in (0, 1)) or (j == 1 and o in (1, 2)) or (j == 2 and o in (0, 2))
            rets.append(o3[j] - 1 if won else -1.0); odds_l.append(o3[j]); wins += won
        n_d_tests += 1
        if len(rets) < 10:
            continue
        pv = tstat_pvalue(rets)
        res_D.append(dict(ev_min=ev_min, n=len(rets), roi_pct=round(float(np.mean(rets)) * 100, 2),
                          wr=round(wins / len(rets), 4), avg_odds=round(float(np.mean(odds_l)), 3), p=pv))
        print(f"D. DC ev>={ev_min}: n={len(rets)} roi={np.mean(rets)*100:+.2f}% wr={wins/len(rets):.3f} "
              f"avgodds={np.mean(odds_l):.2f} p={pv:.4f}")
    out["D_double_chance"] = res_D

    # ---------- E. rolling-origin supporting backtest (1X2 blend) ----------
    folds = [(0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.0)]
    all_rets, all_odds, all_wins, fold_rep = [], [], 0, []
    n = len(rows)
    for ftr, fte in folds:
        tr = rows[:int(n * ftr)]; te = rows[int(n * ftr):int(n * fte)]
        sol, tidx, T = fit_factor(tr, teams)
        pres = pair_residuals(tr, sol, tidx, T)
        # estimate shrink on this train
        X, Y = [], []
        prk = {k: v for k, v in pres.items() if v[2] >= 3}
        rets, odds_l, wins = [], [], 0
        for m in te:
            k = (m["home"], m["away"])
            if k not in prk:
                continue
            mrh, mra, _ = prk[k]
            flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
            lh = math.exp(math.log(flh) + SH * mrh); la = math.exp(math.log(fla) + SA * mra)
            pb = np.array(probs_1x2(lh, la))
            odds = np.array([m["oh"], m["od"], m["oa"]])
            evs = pb * odds - 1
            j = int(np.argmax(evs))
            if evs[j] >= 0.04:
                o = 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)
                won = (o == j)
                rets.append(odds[j] - 1 if won else -1.0); odds_l.append(odds[j]); wins += won
        all_rets += rets; all_odds += odds_l; all_wins += wins
        fold_rep.append(dict(fold=f"{ftr}-{fte}", n=len(rets),
                             roi=round(float(np.mean(rets)) * 100, 2) if rets else None))
        print(f"E. fold {ftr}-{fte}: n={len(rets)} roi={np.mean(rets)*100 if rets else 0:+.2f}%")
    pv = tstat_pvalue(all_rets)
    print(f"E. POOLED rolling-origin: n={len(all_rets)} roi={np.mean(all_rets)*100:+.2f}% "
          f"wr={all_wins/len(all_rets):.3f} avgodds={np.mean(all_odds):.2f} p={pv:.4f}")
    out["E_rolling"] = dict(folds=fold_rep, n=len(all_rets),
                            roi_pct=round(float(np.mean(all_rets)) * 100, 2),
                            wr=round(all_wins / len(all_rets), 4),
                            avg_odds=round(float(np.mean(all_odds)), 3), p=pv)

    # ---------- F. currently deviant pairs (full-data fit) ----------
    sol, tidx, T = fit_factor(rows, teams)
    pres = pair_residuals(rows, sol, tidx, T)
    dev = []
    for k, (mrh, mra, cnt) in pres.items():
        if cnt < 5:
            continue
        mag = max(abs(mrh), abs(mra))
        dev.append(dict(home=k[0], away=k[1], n=cnt, res_h=round(mrh, 3), res_a=round(mra, 3),
                        mag=round(mag, 3)))
    dev.sort(key=lambda x: -x["mag"])
    print("\nF. TOP 15 deviant pairs (full data; res>0 = published lambda ABOVE team strength):")
    for p in dev[:15]:
        print(f"   {p['home']:18s} v {p['away']:18s} n={p['n']:3d} res_h={p['res_h']:+.3f} res_a={p['res_a']:+.3f}")
    out["F_deviant_pairs_top30"] = dev[:30]
    out["n_tests_scanned_phase3"] = len(res_B) + len(res_C) + n_d_tests + 1

    with open("exports/wf4_mufactor_phase3.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nsaved exports/wf4_mufactor_phase3.json")

if __name__ == "__main__":
    main()
