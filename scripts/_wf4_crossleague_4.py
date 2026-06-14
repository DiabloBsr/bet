# -*- coding: utf-8 -*-
"""WF4 cross-league part 4.

A. Cross-market identity: market devig p_over3.5 vs Poisson P(>=4 | mu inverted from 1X2).
B. High-mu drill-down: finer buckets, per-league, mean mu within bucket.
C. O/U 3.5 fine buckets at high p_over (cups) + ROI betting Under at offered odds.
D. Extreme-favorite calibration by era (E2 check old/recent/new).
E. FTTS favori (E1) transposition: bet FTTS '1' when odds_home <= 1.5, settle via goals_json.
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson, binomtest, ttest_1samp, norm
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
CHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
REF = "InstantLeague-8035"
NEWWIN = "2026-06-12 00:00:00"

with open("exports/_wf4_cl_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)
with open("exports/_wf4_cl_extra.json", "r", encoding="utf-8") as f:
    extra = json.load(f)
for e in events:
    e["total"] = e["sa"] + e["sb"]
    e["mu"] = (e["lh"] or 0) + (e["la"] or 0)
    x = extra.get(str(e["id"]))
    if x:
        try:
            d = json.loads(x) if isinstance(x, str) else x
            m = d.get("+/-") or {}
            u, o = m.get("< 3.5"), m.get("> 3.5")
            if u and o and u > 1 and o > 1:
                s = 1 / u + 1 / o
                e["p_over35"] = (1 / o) / s
                e["odds_under"] = u
                e["odds_over"] = o
            f2 = d.get("FTTS") or {}
            if f2.get("1") and f2["1"] > 1 and f2["1"] < 99.5:
                e["ftts1_odds"] = f2["1"]
        except Exception:
            pass

n_tests = 0
out = {}

# ---------- A. cross-market identity ----------
print("A. CROSS-MARKET IDENTITY: market p_over3.5 vs Poisson P(>=4 | mu_1X2)")
resA = {}
for g, sel in [("8035", lambda e: e["league"] == REF),
               ("CHAMP-NEW", lambda e: e["league"] in CHAMP),
               ("CUP", lambda e: e["league"] in CUP)]:
    evs = [e for e in events if sel(e) and e["lh"] and "p_over35" in e]
    pm = np.array([e["p_over35"] for e in evs])
    pg = np.array([1 - poisson.cdf(3, e["mu"]) for e in evs])  # WRONG: needs grid sum
    # exact: P(total>=4) with total = X+Y, X~Pois(lh), Y~Pois(la) indep => total~Pois(mu). OK.
    d = pm - pg
    resA[g] = {"n": len(evs), "mean_diff": round(float(d.mean()), 4),
               "mad": round(float(np.abs(d).mean()), 4),
               "corr": round(float(np.corrcoef(pm, pg)[0, 1]), 4)}
    print(f"  {g:10s} n={len(evs):5d} mean(pm-pg)={d.mean():+.4f} MAD={np.abs(d).mean():.4f} corr={np.corrcoef(pm,pg)[0,1]:.4f}")
    # by mu bucket
    for (a, b) in [(0, 2.3), (2.3, 3.3), (3.3, 4.0), (4.0, 9)]:
        m = [(e["p_over35"] - (1 - poisson.cdf(3, e["mu"]))) for e in evs if a <= e["mu"] < b]
        if len(m) > 50:
            print(f"      mu[{a}-{b}): n={len(m):5d} mean_diff={np.mean(m):+.4f}")
out["cross_market"] = resA

# ---------- B. high-mu drill-down ----------
print("\nB. GOALS BIAS, FINE HIGH-MU BUCKETS (obs total - mu)")
resB = {}
FB = [(3.3, 3.8), (3.8, 4.5), (4.5, 9)]
for g, sel in [("8035", lambda e: e["league"] == REF),
               ("8056-CL", lambda e: e["league"] == "InstantLeague-8056"),
               ("8065-CdM", lambda e: e["league"] == "InstantLeague-8065"),
               ("8060-CAN", lambda e: e["league"] == "InstantLeague-8060"),
               ("8043-De", lambda e: e["league"] == "InstantLeague-8043"),
               ("CHAMP-NEW", lambda e: e["league"] in CHAMP)]:
    rows = []
    for (a, b) in FB:
        evs = [e for e in events if sel(e) and e["lh"] and a <= e["mu"] < b]
        if len(evs) < 40:
            rows.append({"mu_b": f"[{a}-{b})", "n": len(evs)})
            continue
        d = np.array([e["total"] - e["mu"] for e in evs])
        mm = float(np.mean([e["mu"] for e in evs]))
        t, p = ttest_1samp(d, 0)
        n_tests += 1
        rows.append({"mu_b": f"[{a}-{b})", "n": len(evs), "mean_mu_in": round(mm, 2),
                     "bias": round(float(d.mean()), 3),
                     "se": round(float(d.std() / math.sqrt(len(d))), 3), "p": float(p)})
    resB[g] = rows
    print(f"  {g:10s} " + "  ".join(
        f"{r['mu_b']}:{r.get('bias', '—')}(n={r['n']},mu={r.get('mean_mu_in','-')})" for r in rows))
out["highmu_bias"] = resB

# ---------- C. O/U fine buckets high p_over + Under ROI ----------
print("\nC. O/U 3.5 FINE HIGH BUCKETS + ROI UNDER AT OFFERED ODDS")
resC = []
for g, sel in [("8035", lambda e: e["league"] == REF),
               ("CHAMP-NEW", lambda e: e["league"] in CHAMP),
               ("CUP", lambda e: e["league"] in CUP),
               ("8056-CL", lambda e: e["league"] == "InstantLeague-8056"),
               ("8065-CdM", lambda e: e["league"] == "InstantLeague-8065")]:
    for (a, b) in [(0.45, 0.52), (0.52, 0.60), (0.60, 1.01)]:
        evs = [e for e in events if sel(e) and "p_over35" in e and a <= e["p_over35"] < b]
        if len(evs) < 60:
            continue
        n = len(evs)
        obs = sum(1 for e in evs if e["total"] > 3.5)
        expp = float(np.mean([e["p_over35"] for e in evs]))
        pv = binomtest(obs, n, expp).pvalue
        # ROI under
        profits = np.array([e["odds_under"] * (e["total"] < 3.5) - 1 for e in evs])
        roi = float(profits.mean())
        se = profits.std() / math.sqrt(n)
        pv_roi = 2 * norm.sf(abs(roi / se)) if se > 0 else 1.0
        n_tests += 2
        resC.append({"group": g, "b": f"pov[{a}-{b})", "n": n,
                     "obs_over": round(obs / n, 4), "exp_over": round(expp, 4),
                     "p_calib": float(pv), "roi_under": round(roi, 4),
                     "p_roi": float(pv_roi),
                     "avg_under_odds": round(float(np.mean([e['odds_under'] for e in evs])), 3)})
        r = resC[-1]
        print(f"  {g:10s} {r['b']:14s} n={n:5d} over_obs={r['obs_over']:.3f} exp={r['exp_over']:.3f} "
              f"p={r['p_calib']:.3g} | ROI under={r['roi_under']:+.4f} (odds {r['avg_under_odds']}) p={r['p_roi']:.3g}")
out["ou_fine"] = resC

# ---------- D. extreme favorite by era ----------
print("\nD. EXTREME FAVORITE [1.10-1.20] CALIBRATION/ROI BY ERA")
resD = []
def fav_stats(g, evs):
    global n_tests
    bets = []
    for e in evs:
        if e["oh"] <= e["oa"]:
            o, w, pdev = e["oh"], e["out"] == "H", e["ph"]
        else:
            o, w, pdev = e["oa"], e["out"] == "A", e["pa"]
        if 1.10 <= o <= 1.20:
            bets.append((o, w, pdev))
    if len(bets) < 60:
        return None
    n = len(bets)
    wr = sum(w for _, w, _ in bets) / n
    exp = sum(p for _, _, p in bets) / n
    roi = (sum(o * w for o, w, _ in bets) - n) / n
    pv = binomtest(sum(w for _, w, _ in bets), n, exp).pvalue
    n_tests += 1
    r = {"group": g, "n": n, "wr": round(wr, 4), "exp_devig": round(exp, 4),
         "roi": round(roi, 4), "p_vs_devig": float(pv)}
    resD.append(r)
    print(f"  {g:18s} n={n:5d} wr={wr:.4f} exp_devig={exp:.4f} roi={roi:+.4f} p={r['p_vs_devig']:.4g}")
    return r
fav_stats("8035-old", [e for e in events if e["league"] == REF and e["ts"] < NEWWIN])
fav_stats("8035-recent", [e for e in events if e["league"] == REF and e["ts"] >= NEWWIN])
fav_stats("POOLED-ALL-NEW", [e for e in events if e["league"] != REF])
fav_stats("POOLED-NEW+8035rec", [e for e in events if e["league"] != REF or e["ts"] >= NEWWIN])
out["e2_era"] = resD

# ---------- E. FTTS favori (E1) transposition ----------
print("\nE. FTTS '1' WHEN odds_home <= 1.5, settle via goals_json (first goal team)")
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    gj = {r[0]: r[1] for r in c.execute(text(
        "SELECT event_id, goals_json FROM results"))}
resE = []
for g, sel in [("8035-old", lambda e: e["league"] == REF and e["ts"] < NEWWIN),
               ("8035-recent", lambda e: e["league"] == REF and e["ts"] >= NEWWIN),
               ("POOLED-ALL-NEW", lambda e: e["league"] != REF),
               ("POOLED-CHAMP-NEW", lambda e: e["league"] in CHAMP),
               ("POOLED-CUP", lambda e: e["league"] in CUP)]:
    bets = []
    n_nogj = 0
    for e in events:
        if not sel(e) or e["oh"] > 1.5 or "ftts1_odds" not in e:
            continue
        raw = gj.get(e["id"])
        if e["total"] > 0:
            if not raw:
                n_nogj += 1
                continue
            try:
                gl = json.loads(raw)
                if not isinstance(gl, list) or len(gl) == 0:
                    n_nogj += 1
                    continue
            except Exception:
                n_nogj += 1
                continue
            first = sorted(gl, key=lambda x: (x["minute"], x["homeScore"] + x["awayScore"]))[0]
            win = first["team"] == "Home"
        else:
            win = False  # no goal -> '1' loses
        bets.append((e["ftts1_odds"], win))
    if len(bets) < 60:
        continue
    n = len(bets)
    wr = sum(w for _, w in bets) / n
    profits = np.array([o * w - 1 for o, w in bets])
    roi = float(profits.mean())
    se = profits.std() / math.sqrt(n)
    pv = 2 * norm.sf(abs(roi / se)) if se > 0 else 1.0
    n_tests += 1
    r = {"group": g, "n": n, "skipped_no_goalsjson": n_nogj, "wr": round(wr, 4),
         "roi": round(roi, 4), "avg_odds": round(float(np.mean([o for o, _ in bets])), 3),
         "p": float(pv)}
    resE.append(r)
    print(f"  {g:18s} n={n:5d} (skip {n_nogj}) wr={wr:.4f} roi={roi:+.4f} "
          f"odds={r['avg_odds']:.3f} p={pv:.4g}")
out["ftts_e1"] = resE

out["n_tests_part4"] = n_tests
with open("exports/wf4_crossleague_part4.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_crossleague_part4.json; n_tests part4 =", n_tests)
