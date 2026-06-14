# -*- coding: utf-8 -*-
"""
WF4 - sequences temporelles d'equipe INDEPENDANTES des cotes.
Test central: l'ecart entre resultat recent et mu (sur/sous-regime vs cotes)
predit-il le PROCHAIN match au-dela de la cote ?
Methode: regression logistique residuelle (LRT baseline-cotes vs baseline+residuals),
+ buckets de sur/sous-regime, + walk-forward 70/30 sur 8035.
Sortie brute: exports/wf4_seq.json
LECTURE SEULE sur la DB.
"""
import sys, json, math, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore", category=FutureWarning)
from datetime import datetime
import numpy as np
from scipy.special import gammaln, expit
from scipy import stats
from sqlalchemy import create_engine, text
from scraper.config import load_settings

RNG = np.random.default_rng(42)
K = 15  # grille Poisson 0..K

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW = LEAGUES[1:]

# ---------- data ----------

def load_data():
    eng = create_engine(load_settings().db_url)
    corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())
    q = text("""
        SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start, e.round_info,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
        WHERE e.competition IN :comps
        ORDER BY e.expected_start, e.id
    """).bindparams(**{})
    rows = []
    with eng.connect() as c:
        res = c.execute(text("""
            SELECT e.id, e.competition, e.team_a, e.team_b, e.expected_start, e.round_info,
                   r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
                   o.odds_home, o.odds_draw, o.odds_away
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.id = (
                SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
            ORDER BY e.expected_start, e.id
        """)).fetchall()
    n_guard = 0
    for r in res:
        (eid, comp, ta, tb, start, rinfo, sa, sb, hta, htb, oh, od, oa) = r
        if comp not in LEAGUES:
            continue
        if eid in corrupted:
            continue
        if sa is None or sb is None or oh is None or od is None or oa is None:
            continue
        if min(oh, od, oa) <= 1.0:
            continue
        # garde-fou maison corruption (les nouvelles ligues ne sont pas auditees)
        if hta is not None and htb is not None and (hta > sa or htb > sb):
            n_guard += 1
            continue
        rows.append(dict(id=eid, comp=comp, ta=ta, tb=tb, start=str(start),
                         rnd=int(rinfo) if rinfo is not None and str(rinfo).isdigit() else -1,
                         sa=int(sa), sb=int(sb),
                         oh=float(oh), od=float(od), oa=float(oa)))
    # DEDUP anti-leakage: events quasi-dupliques (meme comp/equipes, coup d'envoi
    # a moins de 30 min) -> on garde le MIN(id). 194/356 paires 8035 ont un score
    # IDENTIQUE (vs ~10% attendu par hasard) => vrais doublons du meme match;
    # sans dedup, la copie 2 a son propre resultat dans son historique (fuite directe).
    def ts(s):
        return datetime.fromisoformat(s).timestamp()
    bykey = {}
    for r in rows:
        bykey.setdefault((r["comp"], r["ta"], r["tb"]), []).append(r)
    drop = set()
    for key, lst in bykey.items():
        lst.sort(key=lambda r: (ts(r["start"]), r["id"]))
        for i in range(1, len(lst)):
            if ts(lst[i]["start"]) - ts(lst[i - 1]["start"]) < 1800 and lst[i - 1]["id"] not in drop:
                drop.add(lst[i]["id"])
    rows = [r for r in rows if r["id"] not in drop]
    print(f"loaded {len(rows)} clean matches (dedup near-dups: {len(drop)} dropped), guard-excluded HT>FT: {n_guard}")
    return rows

# ---------- lambda inversion (Newton vectorise) ----------

def grid_probs(lh, la):
    k = np.arange(K + 1)
    lp_h = -lh[:, None] + k[None, :] * np.log(lh[:, None]) - gammaln(k + 1)[None, :]
    lp_a = -la[:, None] + k[None, :] * np.log(la[:, None]) - gammaln(k + 1)[None, :]
    Ph, Pa = np.exp(lp_h), np.exp(lp_a)
    Jt = Ph[:, :, None] * Pa[:, None, :]
    i, j = np.meshgrid(k, k, indexing="ij")
    p_home = (Jt * (i > j)[None]).sum((1, 2))
    p_draw = (Jt * (i == j)[None]).sum((1, 2))
    return p_home, p_draw

def invert_lambdas(ph, pd):
    n = len(ph)
    x = np.full(n, math.log(1.6)); y = np.full(n, math.log(1.2))
    eps = 1e-5
    for _ in range(40):
        lh, la = np.exp(x), np.exp(y)
        f1, f2 = grid_probs(lh, la)
        F1, F2 = f1 - ph, f2 - pd
        if max(np.abs(F1).max(), np.abs(F2).max()) < 1e-9:
            break
        a1, a2 = grid_probs(np.exp(x + eps), la)
        b1, b2 = grid_probs(lh, np.exp(y + eps))
        J11, J21 = (a1 - f1) / eps, (a2 - f2) / eps
        J12, J22 = (b1 - f1) / eps, (b2 - f2) / eps
        det = J11 * J22 - J12 * J21
        det = np.where(np.abs(det) < 1e-12, 1e-12, det)
        dx = (F1 * J22 - F2 * J12) / det
        dy = (J11 * F2 - J21 * F1) / det
        step = np.clip(np.maximum(np.abs(dx), np.abs(dy)), 0, 0.5) / np.maximum(np.maximum(np.abs(dx), np.abs(dy)), 1e-12)
        x = np.clip(x - dx * step, math.log(0.05), math.log(6.0))
        y = np.clip(y - dy * step, math.log(0.05), math.log(6.0))
    return np.exp(x), np.exp(y)

# ---------- logistic + LRT ----------

def fit_logistic_ll(X, y):
    """retourne LL maximise (sans penalite)"""
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
    m.fit(X, y)
    p = m.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))), m

def lrt_logistic(X_base, X_extra, y):
    ll0, _ = fit_logistic_ll(X_base, y)
    ll1, m1 = fit_logistic_ll(np.hstack([X_base, X_extra]), y)
    df = X_extra.shape[1]
    chi2 = 2 * (ll1 - ll0)
    p = float(stats.chi2.sf(max(chi2, 0), df))
    return dict(ll0=ll0, ll1=ll1, chi2=float(chi2), df=df, p=p,
                coefs_extra=[float(c) for c in m1.coef_[0][-df:]])

def ftest_ols(X_base, X_extra, y):
    def rss(X):
        Xd = np.hstack([np.ones((len(y), 1)), X])
        beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
        r = y - Xd @ beta
        return float(r @ r), Xd.shape[1], beta
    rss0, p0, _ = rss(X_base)
    rss1, p1, beta1 = rss(np.hstack([X_base, X_extra]))
    q = X_extra.shape[1]
    n = len(y)
    F = ((rss0 - rss1) / q) / (rss1 / (n - p1))
    p = float(stats.f.sf(max(F, 0), q, n - p1))
    return dict(F=float(F), df=q, p=p, coefs_extra=[float(b) for b in beta1[-q:]])

# ---------- build features ----------

def build(rows):
    """rajoute lambdas + probas normalisees, puis residuals d'equipe pre-match."""
    oh = np.array([r["oh"] for r in rows]); od = np.array([r["od"] for r in rows]); oa = np.array([r["oa"] for r in rows])
    inv = 1 / oh + 1 / od + 1 / oa
    ph, pd, pa = (1 / oh) / inv, (1 / od) / inv, (1 / oa) / inv
    lh, la = invert_lambdas(ph, pd)
    # verif inversion
    chk_h, chk_d = grid_probs(lh, la)
    err = max(np.abs(chk_h - ph).max(), np.abs(chk_d - pd).max())
    print(f"lambda inversion max err = {err:.2e}; mean lh={lh.mean():.3f} la={la.mean():.3f}")
    for i, r in enumerate(rows):
        r.update(ph=float(ph[i]), pd=float(pd[i]), pa=float(pa[i]),
                 lh=float(lh[i]), la=float(la[i]), margin=float(inv[i] - 1))
    # historiques par equipe (cle = comp + team, collisions gerees)
    hist = {}  # key -> list of dicts residuals (ordre chrono car rows triees)
    for r in rows:
        feats = {}
        for side, team in (("h", (r["comp"], r["ta"])), ("a", (r["comp"], r["tb"]))):
            hlist = hist.get(team, [])
            for N in (3, 5):
                if len(hlist) >= N:
                    sub = hlist[-N:]
                    feats[f"{side}_rpts{N}"] = float(np.mean([x["rpts"] for x in sub]))
                    feats[f"{side}_rwin{N}"] = float(np.mean([x["rwin"] for x in sub]))
                    feats[f"{side}_rgd{N}"] = float(np.mean([x["rgd"] for x in sub]))
                    feats[f"{side}_rgf{N}"] = float(np.mean([x["rgf"] for x in sub]))
                else:
                    feats[f"{side}_rpts{N}"] = None
        r["feats"] = feats
        # mise a jour des historiques APRES la decision (pas de fuite)
        for side, team, pw, pdr, gf, ga, lf, lg in (
                ("h", (r["comp"], r["ta"]), r["ph"], r["pd"], r["sa"], r["sb"], r["lh"], r["la"]),
                ("a", (r["comp"], r["tb"]), r["pa"], r["pd"], r["sb"], r["sa"], r["la"], r["lh"])):
            win = 1.0 if gf > ga else 0.0
            pts = 3.0 if gf > ga else (1.0 if gf == ga else 0.0)
            hist.setdefault(team, []).append(dict(
                rwin=win - pw, rpts=pts - (3 * pw + pdr),
                rgf=gf - lf, rgd=(gf - ga) - (lf - lg)))
    return rows

def logit(p):
    return np.log(p / (1 - p))

def run_tests(rows):
    results = {"tests": [], "buckets": {}, "walkforward": {}}
    n_tests = 0

    def subset(rows, scope):
        if scope == "8035":
            return [r for r in rows if r["comp"] == "InstantLeague-8035"]
        if scope == "pooled-newleagues":
            return [r for r in rows if r["comp"] in NEW]
        return list(rows)

    for scope in ("8035", "pooled-newleagues", "pooled-9"):
        sub = subset(rows, scope)
        for N in (3, 5):
            ok = [r for r in sub if r["feats"].get(f"h_rpts{N}") is not None
                  and r["feats"].get(f"a_rpts{N}") is not None]
            if len(ok) < 200:
                continue
            y_hw = np.array([1.0 if r["sa"] > r["sb"] else 0.0 for r in ok])
            y_dr = np.array([1.0 if r["sa"] == r["sb"] else 0.0 for r in ok])
            y_tot = np.array([float(r["sa"] + r["sb"]) for r in ok])
            Xb_hw = logit(np.array([r["ph"] for r in ok]))[:, None]
            Xb_dr = logit(np.array([r["pd"] for r in ok]))[:, None]
            Xb_tot = np.array([r["lh"] + r["la"] for r in ok])[:, None]
            for fam in ("rpts", "rwin", "rgd"):
                Xe = np.array([[r["feats"][f"h_{fam}{N}"], r["feats"][f"a_{fam}{N}"]] for r in ok])
                t = lrt_logistic(Xb_hw, Xe, y_hw)
                t.update(scope=scope, N=N, family=fam, target="homewin", n=len(ok))
                results["tests"].append(t); n_tests += 1
            # draw
            Xe = np.array([[r["feats"][f"h_rpts{N}"], r["feats"][f"a_rpts{N}"]] for r in ok])
            t = lrt_logistic(Xb_dr, Xe, y_dr)
            t.update(scope=scope, N=N, family="rpts", target="draw", n=len(ok))
            results["tests"].append(t); n_tests += 1
            # total goals (OLS F-test, baseline mu_total)
            Xe = np.array([[r["feats"][f"h_rgf{N}"], r["feats"][f"a_rgf{N}"]] for r in ok])
            t = ftest_ols(Xb_tot, Xe, y_tot)
            t.update(scope=scope, N=N, family="rgf", target="total_goals", n=len(ok))
            results["tests"].append(t); n_tests += 1

    # ---- buckets sur/sous-regime (rpts5 diff home-away), pooled-9 ----
    N = 5
    ok = [r for r in rows if r["feats"].get(f"h_rpts{N}") is not None
          and r["feats"].get(f"a_rpts{N}") is not None]
    diff = np.array([r["feats"][f"h_rpts{N}"] - r["feats"][f"a_rpts{N}"] for r in ok])
    y_hw = np.array([1.0 if r["sa"] > r["sb"] else 0.0 for r in ok])
    ph = np.array([r["ph"] for r in ok])
    res = y_hw - ph
    qs = np.quantile(diff, np.linspace(0, 1, 11))
    buckets = []
    for b in range(10):
        m = (diff >= qs[b]) & (diff <= qs[b + 1] if b == 9 else diff < qs[b + 1])
        if m.sum() == 0:
            continue
        gap = float(res[m].mean())
        se = float(res[m].std(ddof=1) / math.sqrt(m.sum()))
        buckets.append(dict(decile=b + 1, n=int(m.sum()),
                            diff_lo=float(qs[b]), diff_hi=float(qs[b + 1]),
                            mean_calib_gap=gap, se=se, z=gap / se if se > 0 else 0.0))
        n_tests += 1
    results["buckets"]["rpts5_diff_pooled9"] = buckets
    # correlation spearman residual-vs-residual (test global non parametrique)
    rho, p_rho = stats.spearmanr(diff, res)
    results["buckets"]["spearman_diff_vs_calibgap"] = dict(rho=float(rho), p=float(p_rho), n=len(ok))
    n_tests += 1

    # ---- walk-forward 8035: la regle "back l'equipe en sur-regime" rapporte-t-elle ? ----
    sub8 = [r for r in rows if r["comp"] == "InstantLeague-8035"
            and r["feats"].get("h_rpts5") is not None and r["feats"].get("a_rpts5") is not None]
    sub8.sort(key=lambda r: (r["start"], r["id"]))
    cut = int(len(sub8) * 0.7)
    train, test = sub8[:cut], sub8[cut:]
    newl = [r for r in rows if r["comp"] in NEW
            and r["feats"].get("h_rpts5") is not None and r["feats"].get("a_rpts5") is not None]

    def eval_rule(pop, sel, thr):
        pnls, odds_used, wins = [], [], 0
        for r in pop:
            d = r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"]
            side = None
            if sel * d >= thr:
                side = "h"
            elif sel * (-d) >= thr:
                side = "a"
            if side is None:
                continue
            o = r["oh"] if side == "h" else r["oa"]
            won = (r["sa"] > r["sb"]) if side == "h" else (r["sb"] > r["sa"])
            odds_used.append(o)
            pnls.append((o - 1) if won else -1.0)
            wins += int(won)
        n = len(pnls)
        if n == 0:
            return dict(n=0)
        pnls = np.array(pnls)
        roi = float(pnls.mean())
        se = float(pnls.std(ddof=1) / math.sqrt(n))
        # p-value bootstrap (H0: ROI<=0) + z normal
        boot = RNG.choice(pnls, size=(4000, n), replace=True).mean(axis=1)
        p_boot = float((boot <= 0).mean())
        return dict(n=n, wr=wins / n, roi_pct=100 * roi, se_roi_pct=100 * se,
                    z=roi / se if se > 0 else 0.0, p_boot_roi_le_0=p_boot,
                    avg_odds=float(np.mean(odds_used)))

    wf = {}
    dtr = np.array([r["feats"]["h_rpts5"] - r["feats"]["a_rpts5"] for r in train])
    thr = float(np.quantile(np.abs(dtr), 0.8))
    for name, sel in (("back_overperf", 1), ("back_underperf", -1)):
        wf[name] = dict(threshold=thr,
                        train70=eval_rule(train, sel, thr),
                        test30=eval_rule(test, sel, thr),
                        pooled_newleagues=eval_rule(newl, sel, thr))
        n_tests += 3
    results["walkforward"]["8035_rpts5_rule"] = wf

    # ---- confirmation predictive OOS du meilleur candidat LRT (rgd5 / rpts5 homewin 8035):
    # fit train70, delta log-loss sur test30 vs baseline cotes seules
    y_tr = np.array([1.0 if r["sa"] > r["sb"] else 0.0 for r in train])
    y_te = np.array([1.0 if r["sa"] > r["sb"] else 0.0 for r in test])
    Xb_tr = logit(np.array([r["ph"] for r in train]))[:, None]
    Xb_te = logit(np.array([r["ph"] for r in test]))[:, None]
    oos = {}
    for fam in ("rgd", "rpts"):
        Xe_tr = np.array([[r["feats"][f"h_{fam}5"], r["feats"][f"a_{fam}5"]] for r in train])
        Xe_te = np.array([[r["feats"][f"h_{fam}5"], r["feats"][f"a_{fam}5"]] for r in test])
        _, m0 = fit_logistic_ll(Xb_tr, y_tr)
        _, m1 = fit_logistic_ll(np.hstack([Xb_tr, Xe_tr]), y_tr)
        def ll(m, X, y):
            p = np.clip(m.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
            return float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        ll0 = ll(m0, Xb_te, y_te)
        ll1 = ll(m1, np.hstack([Xb_te, Xe_te]), y_te)
        oos[fam + "5"] = dict(test_logloss_base=-ll0, test_logloss_resid=-ll1,
                              delta=-(ll1 - ll0), improves=bool(ll1 > ll0))
        n_tests += 1
    results["walkforward"]["8035_oos_logloss_homewin"] = oos
    results["n_tests_scanned"] = n_tests
    return results

def main():
    rows = load_data()
    rows = build(rows)
    out = run_tests(rows)
    out["n_rows_clean"] = len(rows)
    out["per_league"] = {}
    for lg in LEAGUES:
        out["per_league"][lg] = sum(1 for r in rows if r["comp"] == lg)
    with open("exports/wf4_seq.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    # resume console
    print("\n=== LRT / F tests (p<=0.05 d'abord) ===")
    for t in sorted(out["tests"], key=lambda x: x["p"]):
        print(f"{t['scope']:>18} N={t['N']} {t['family']:>5} -> {t['target']:<11} n={t['n']:>6} "
              f"p={t['p']:.4g} coefs={[round(c,4) for c in t['coefs_extra']]}")
    print("\n=== buckets rpts5 diff (pooled-9) ===")
    for b in out["buckets"]["rpts5_diff_pooled9"]:
        print(f" D{b['decile']:>2} n={b['n']:>5} gap={b['mean_calib_gap']:+.4f} z={b['z']:+.2f}")
    print("spearman:", out["buckets"]["spearman_diff_vs_calibgap"])
    print("\n=== walk-forward 8035 (train70 / test30 / pooled-newleagues) ===")
    print(json.dumps(out["walkforward"], indent=1))
    print("\nn_tests_scanned =", out["n_tests_scanned"])

if __name__ == "__main__":
    main()
