# -*- coding: utf-8 -*-
"""WF4 - mu factorization phase 6: final hardening of V5 (recalibrated blend).
 1. STRICT 70/30 walk-forward (primary per methodology): train 70%, bet test 30%.
 2. per-fold stability of V5 (rolling folds).
 3. corruption guard: exclude results where HT>FT on either side, re-run.
 4. audit 3 sample winning longshot bets directly against DB.
 5. current deviant pairs with V5 bet direction (full-data fit) = actionable list.
Outputs exports/wf4_mufactor_final.json
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

def mc_pvalue(bets, nsim=300000, seed=11):
    rng = np.random.default_rng(seed)
    o = np.array([b[0] for b in bets]); w = np.array([b[1] for b in bets])
    p0 = 1.0 / o
    obs_roi = float((w * o - 1).mean())
    hits, done = 0, 0
    chunk = max(1, int(2e7 / max(len(o), 1)))
    while done < nsim:
        k = min(chunk, nsim - done)
        sims = (rng.random((k, len(o))) < p0) * o[None, :]
        hits += int(((sims - 1).mean(axis=1) >= obs_roi).sum())
        done += k
    z = (w.sum() - p0.sum()) / math.sqrt((p0 * (1 - p0)).sum())
    return obs_roi, hits / nsim, float(z)

def v5_machine(tr, teams):
    """fit factor + pair residuals + GLM coefs on train; return scorer."""
    sol, tidx, T = fit_factor(tr, teams)
    pr = defaultdict(lambda: [0.0, 0.0, 0])
    rec = []
    for m in tr:
        flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
        rh = math.log(m["lh"]) - math.log(flh); ra = math.log(m["la"]) - math.log(fla)
        k = (m["home"], m["away"])
        pr[k][0] += rh; pr[k][1] += ra; pr[k][2] += 1
        rec.append((k, flh, fla, rh, ra, m["sa"], m["sb"]))
    pres = {k: (v[0] / v[2], v[1] / v[2], v[2]) for k, v in pr.items() if v[2] >= 3}
    Xh, Xa, Yh, Ya = [], [], [], []
    for k, flh, fla, rh, ra, sa, sb in rec:
        if k not in pres:
            continue
        mrh, mra, cnt = pres[k]
        looh = (mrh * cnt - rh) / (cnt - 1); looa = (mra * cnt - ra) / (cnt - 1)
        Xh.append([1.0, math.log(flh), looh]); Yh.append(sa)
        Xa.append([1.0, math.log(fla), looa]); Ya.append(sb)
    bh = poisson_glm(Xh, Yh); ba = poisson_glm(Xa, Ya)

    def score(m):
        k = (m["home"], m["away"])
        if k not in pres:
            return None
        mrh, mra, _ = pres[k]
        flh, fla = factor_lambdas(sol, tidx, T, m["home"], m["away"])
        lh = math.exp(bh[0] + bh[1] * math.log(flh) + bh[2] * mrh)
        la = math.exp(ba[0] + ba[1] * math.log(fla) + ba[2] * mra)
        return np.array(probs_1x2(lh, la))
    return score, (bh, ba), pres

def settle(m):
    return 0 if m["sa"] > m["sb"] else (1 if m["sa"] == m["sb"] else 2)

def main():
    out = {}
    d = pickle.load(open("exports/_wf4_mufactor_data.pkl", "rb"))
    rows, ncut, teams = d["rows"], d["ncut"], d["teams"]
    n = len(rows)
    EVMIN = 0.04

    # corruption guard data: HT scores
    eng = create_engine(load_settings().db_url)
    ht = {}
    with eng.connect() as c:
        for eid, ha, hb, sa, sb in c.execute(text(
                "SELECT r.event_id, r.ht_score_a, r.ht_score_b, r.score_a, r.score_b "
                "FROM results r JOIN events e ON e.id=r.event_id "
                "WHERE e.competition='InstantLeague-8035'")):
            ht[eid] = (ha, hb, sa, sb)
    bad = set()
    for m in rows:
        h = ht.get(m["id"])
        if h and h[0] is not None and h[1] is not None:
            if h[0] > h[2] or h[1] > h[3]:
                bad.add(m["id"])
    print(f"corruption guard: {len(bad)} events with HT>FT among {len(rows)} (excluded in guarded runs)")
    out["n_ht_gt_ft"] = len(bad)

    # ---------- 1. STRICT 70/30 V5 ----------
    tr, te = rows[:ncut], rows[ncut:]
    score, (bh, ba), pres = v5_machine(tr, teams)
    print(f"V5 train coefs home a={bh[0]:.3f} c={bh[1]:.3f} s={bh[2]:.3f} | "
          f"away a={ba[0]:.3f} c={ba[1]:.3f} s={ba[2]:.3f}")
    out["v5_coefs_7030"] = dict(home=[float(x) for x in bh], away=[float(x) for x in ba])
    for tag, guard in [("strict7030", False), ("strict7030_guard", True)]:
        bets, log = [], []
        for m in te:
            if guard and m["id"] in bad:
                continue
            pb = score(m)
            if pb is None:
                continue
            odds = np.array([m["oh"], m["od"], m["oa"]])
            evs = pb * odds - 1
            j = int(np.argmax(evs))
            if evs[j] >= EVMIN:
                bets.append((float(odds[j]), int(settle(m) == j)))
                log.append(dict(id=m["id"], pair=(m["home"], m["away"]), sel="HXA"[j],
                                odd=float(odds[j]), won=int(settle(m) == j),
                                start=str(m["start"])))
        roi, pmc, z = mc_pvalue(bets)
        wr = float(np.mean([b[1] for b in bets])); ao = float(np.mean([b[0] for b in bets]))
        npr = len(set((l["pair"], l["sel"]) for l in log))
        print(f"V5 {tag}: n={len(bets)} roi={roi*100:+.2f}% wr={wr:.3f} avgodds={ao:.2f} "
              f"z={z:+.2f} MCp={pmc:.5f} distinct(pair,sel)={npr}")
        out[f"v5_{tag}"] = dict(n=len(bets), roi_pct=round(roi * 100, 2), wr=round(wr, 4),
                                avg_odds=round(ao, 3), z=round(z, 2), p_mc=pmc,
                                distinct_pair_sel=npr)
        if tag == "strict7030":
            log7030 = log

    # ---------- 2+3. rolling folds V5, with guard ----------
    folds = [(0.4, 0.55), (0.55, 0.7), (0.7, 0.85), (0.85, 1.0)]
    pooled, pooled_g, fold_rep = [], [], []
    pooled_log = []
    for ftr, fte in folds:
        trf = rows[:int(n * ftr)]; tef = rows[int(n * ftr):int(n * fte)]
        sc, _, _ = v5_machine(trf, teams)
        fb = []
        for m in tef:
            pb = sc(m)
            if pb is None:
                continue
            odds = np.array([m["oh"], m["od"], m["oa"]])
            evs = pb * odds - 1
            j = int(np.argmax(evs))
            if evs[j] >= EVMIN:
                bet = (float(odds[j]), int(settle(m) == j))
                fb.append(bet); pooled.append(bet)
                pooled_log.append(dict(id=m["id"], pair=(m["home"], m["away"]), sel="HXA"[j],
                                       odd=float(odds[j]), won=int(settle(m) == j)))
                if m["id"] not in bad:
                    pooled_g.append(bet)
        r = float(np.mean([b[1] * b[0] - 1 for b in fb])) * 100 if fb else None
        fold_rep.append(dict(fold=f"{ftr}-{fte}", n=len(fb), roi_pct=round(r, 2) if r is not None else None))
        print(f"V5 fold {ftr}-{fte}: n={len(fb)} roi={r:+.2f}%")
    for tag, b in [("rolling", pooled), ("rolling_guard", pooled_g)]:
        roi, pmc, z = mc_pvalue(b)
        wr = float(np.mean([x[1] for x in b])); ao = float(np.mean([x[0] for x in b]))
        print(f"V5 {tag} pooled: n={len(b)} roi={roi*100:+.2f}% wr={wr:.3f} avgodds={ao:.2f} "
              f"z={z:+.2f} MCp={pmc:.5f}")
        out[f"v5_{tag}"] = dict(n=len(b), roi_pct=round(roi * 100, 2), wr=round(wr, 4),
                                avg_odds=round(ao, 3), z=round(z, 2), p_mc=pmc)
    out["v5_folds"] = fold_rep
    npr = len(set((l["pair"], l["sel"]) for l in pooled_log))
    out["v5_rolling_distinct_pair_sel"] = npr
    print(f"rolling distinct (pair,sel): {npr}")

    # ---------- 4. audit 3 winning longshot bets ----------
    winners = [l for l in pooled_log if l["won"] and l["odd"] >= 6][:3]
    audit = []
    with eng.connect() as c:
        for w in winners:
            r = c.execute(text(
                "SELECT e.team_a, e.team_b, r.score_a, r.score_b, "
                "(SELECT odds_home FROM odds_snapshots o WHERE o.event_id=e.id ORDER BY o.id LIMIT 1), "
                "(SELECT odds_draw FROM odds_snapshots o WHERE o.event_id=e.id ORDER BY o.id LIMIT 1), "
                "(SELECT odds_away FROM odds_snapshots o WHERE o.event_id=e.id ORDER BY o.id LIMIT 1) "
                "FROM events e JOIN results r ON r.event_id=e.id WHERE e.id=:i"), {"i": w["id"]}).fetchone()
            audit.append(dict(bet=w, db=dict(home=r[0], away=r[1], score=f"{r[2]}-{r[3]}",
                                             open_odds=[float(r[4]), float(r[5]), float(r[6])])))
            print(f"AUDIT id={w['id']} bet {w['sel']}@{w['odd']} on {r[0]} v {r[1]} -> {r[2]}-{r[3]} "
                  f"open odds {r[4]}/{r[5]}/{r[6]}")
    out["audit_winning_bets"] = audit

    # ---------- 5. current deviant pairs + V5 direction (full data fit) ----------
    sc_full, (bhf, baf), pres_full = v5_machine(rows, teams)
    dev = []
    seen_pairs = set()
    # use latest published odds per pair (most recent occurrence in rows)
    for m in reversed(rows):
        k = (m["home"], m["away"])
        if k in seen_pairs or k not in pres_full:
            continue
        seen_pairs.add(k)
        pb = sc_full(m)
        odds = np.array([m["oh"], m["od"], m["oa"]])
        evs = pb * odds - 1
        j = int(np.argmax(evs))
        mrh, mra, cnt = pres_full[k]
        if evs[j] >= EVMIN:
            dev.append(dict(home=k[0], away=k[1], n_hist=cnt, res_h=round(mrh, 3),
                            res_a=round(mra, 3), bet="HXA"[j], ev=round(float(evs[j]), 3),
                            last_odds=[round(float(o), 2) for o in odds]))
    dev.sort(key=lambda x: -x["ev"])
    print(f"\ncurrently deviant pairs with V5 EV>={EVMIN} at last seen odds: {len(dev)}")
    for p in dev[:20]:
        print(f"  {p['home']:18s} v {p['away']:18s} bet={p['bet']} ev={p['ev']:+.3f} "
              f"res=({p['res_h']:+.3f},{p['res_a']:+.3f}) odds={p['last_odds']}")
    out["deviant_pairs_actionable"] = dev

    with open("exports/wf4_mufactor_final.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=float)
    print("\nsaved exports/wf4_mufactor_final.json")

if __name__ == "__main__":
    main()
