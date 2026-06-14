# -*- coding: utf-8 -*-
"""WF4 cross-league part 3.

1. Goals bias as a function of mu (is cross-league bias heterogeneity just mu composition?)
2. 8035 old vs recent formal comparison + J0 vs regular rounds.
3. ROI scan at offered opening odds: 'Score exact' (28 cells) and 'Total de buts'
   per league group, walk-forward split for 8035.
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from collections import defaultdict
from scipy.stats import binomtest, ttest_1samp, norm
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

# round_info for 8035 J0 split
eng = create_engine(load_settings().db_url)
with eng.connect() as c:
    rmap = {r[0]: r[1] for r in c.execute(text(
        "SELECT id, round_info FROM events WHERE competition='InstantLeague-8035'"))}
for e in events:
    if e["league"] == REF:
        e["round"] = rmap.get(e["id"], None)

n_tests = 0
out = {}

# ---------- 1. bias vs mu buckets ----------
print("1. GOALS BIAS BY MU BUCKET (obs total - mu)")
MB = [(0, 1.8), (1.8, 2.3), (2.3, 2.8), (2.8, 3.3), (3.3, 9)]
res1 = {}
for g, sel in [("8035", lambda e: e["league"] == REF),
               ("8035-old", lambda e: e["league"] == REF and e["ts"] < NEWWIN),
               ("8035-recent", lambda e: e["league"] == REF and e["ts"] >= NEWWIN),
               ("ALL-NEW", lambda e: e["league"] != REF),
               ("CHAMP-NEW", lambda e: e["league"] in CHAMP),
               ("CUP", lambda e: e["league"] in CUP)]:
    rows = []
    for (a, b) in MB:
        evs = [e for e in events if sel(e) and e["lh"] and a <= e["mu"] < b]
        if len(evs) < 80:
            rows.append(None)
            continue
        d = np.array([e["total"] - e["mu"] for e in evs])
        t, p = ttest_1samp(d, 0)
        n_tests += 1
        rows.append({"mu_b": f"[{a}-{b})", "n": len(evs),
                     "bias": round(float(d.mean()), 3),
                     "se": round(float(d.std() / math.sqrt(len(d))), 3),
                     "p": float(p)})
    res1[g] = rows
    s = "  ".join(f"{r['mu_b']}:{r['bias']:+.3f}(n={r['n']})" if r else "-" for r in rows)
    print(f"  {g:12s} {s}")
out["bias_by_mu"] = res1

# ---------- 2. 8035 old vs recent + J0 ----------
print("\n2. 8035 SUBGROUP DEVIATIONS (ratio obs/exp Poisson)")
from scipy.stats import poisson
KEY = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)]
def grid_ratios(evs, label):
    global n_tests
    n = len(evs)
    if n < 100:
        print(f"  {label} SKIPPED (n={n})")
        return None
    lh = np.array([e["lh"] for e in evs])
    la = np.array([e["la"] for e in evs])
    rows = {}
    for (i, j) in KEY:
        exp = float(np.sum(poisson.pmf(i, lh) * poisson.pmf(j, la)))
        obs = sum(1 for e in evs if e["sa"] == i and e["sb"] == j)
        pv = binomtest(obs, n, exp / n).pvalue
        n_tests += 1
        rows[f"{i}-{j}"] = {"obs": obs, "exp": round(exp, 1),
                            "ratio": round(obs / exp, 3), "p": float(pv)}
    exp_d = float(np.sum(sum(poisson.pmf(k, lh) * poisson.pmf(k, la) for k in range(11))))
    obs_d = sum(1 for e in evs if e["sa"] == e["sb"])
    n_tests += 1
    rows["DRAWS"] = {"obs": obs_d, "exp": round(exp_d, 1),
                     "ratio": round(obs_d / exp_d, 3),
                     "p": float(binomtest(obs_d, n, exp_d / n).pvalue)}
    print(f"  {label} (n={n}): " + "  ".join(
        f"{k}:{v['ratio']:.2f}" for k, v in rows.items()))
    return {"n": n, "cells": rows}

evs35 = [e for e in events if e["league"] == REF and e["lh"]]
nj0 = sum(1 for e in evs35 if e.get("round") == "0")
print(f"  8035 J0 events with odds+result: {nj0} "
      f"(old: {sum(1 for e in evs35 if e.get('round') == '0' and e['ts'] < NEWWIN)})")
res2 = {}
res2["old-J0"] = grid_ratios([e for e in evs35 if e["ts"] < NEWWIN and e.get("round") == "0"], "8035 old J0      ")
res2["old-J1-38"] = grid_ratios([e for e in evs35 if e["ts"] < NEWWIN and e.get("round") != "0"], "8035 old J1-38   ")
res2["recent-J0"] = grid_ratios([e for e in evs35 if e["ts"] >= NEWWIN and e.get("round") == "0"], "8035 recent J0   ")
res2["recent-all"] = grid_ratios([e for e in evs35 if e["ts"] >= NEWWIN], "8035 recent all  ")
# two-proportion z old vs recent for 0-0 and DRAWS
def two_prop(evsA, evsB, pred):
    a = sum(1 for e in evsA if pred(e)); b = sum(1 for e in evsB if pred(e))
    na, nb = len(evsA), len(evsB)
    p1, p2 = a / na, b / nb
    pp = (a + b) / (na + nb)
    se = math.sqrt(pp * (1 - pp) * (1 / na + 1 / nb))
    z = (p1 - p2) / se
    return p1, p2, z, 2 * norm.sf(abs(z))
oldE = [e for e in evs35 if e["ts"] < NEWWIN]
recE = [e for e in evs35 if e["ts"] >= NEWWIN]
for name, pred in [("0-0", lambda e: e["sa"] == 0 and e["sb"] == 0),
                   ("draw", lambda e: e["sa"] == e["sb"]),
                   ("2-1", lambda e: e["sa"] == 2 and e["sb"] == 1)]:
    p1, p2, z, pv = two_prop(oldE, recE, pred)
    n_tests += 1
    print(f"  old-vs-recent {name}: old={p1:.4f} recent={p2:.4f} z={z:+.2f} p={pv:.4g}")
    res2[f"oldrec_{name}"] = {"old": p1, "recent": p2, "z": z, "p": pv}
out["8035_subgroups"] = res2

# ---------- 3. ROI scan: Score exact + Total de buts at offered odds ----------
print("\n3. ROI SCAN AT OFFERED OPENING ODDS")
def get_market(e, mname):
    x = extra.get(str(e["id"]))
    if not x:
        return None
    try:
        d = json.loads(x) if isinstance(x, str) else x
        return d.get(mname)
    except Exception:
        return None

evs35s = sorted(evs35, key=lambda e: e["ts"])
cut = int(len(evs35s) * 0.70)
SCAN_GROUPS = [
    ("8035-wf-train", evs35s[:cut]),
    ("8035-wf-test", evs35s[cut:]),
    ("8035-recent", recE),
    ("POOLED-ALL-NEW", [e for e in events if e["league"] != REF]),
    ("POOLED-CHAMP-NEW", [e for e in events if e["league"] in CHAMP]),
    ("POOLED-CUP", [e for e in events if e["league"] in CUP]),
]
scan = []
def roi_scan(group, evs, mname, selname, settle):
    """settle(e) -> True/False win. Bet 1u at offered odds when odds < 100."""
    global n_tests
    bets = []
    for e in evs:
        m = get_market(e, mname)
        if not m:
            continue
        o = m.get(selname)
        if not o or o <= 1 or o >= 99.5:
            continue
        bets.append((o, settle(e)))
    n = len(bets)
    if n < 100:
        return None
    wins = sum(1 for _, w in bets if w)
    ret = sum(o for o, w in bets if w)
    roi = (ret - n) / n
    avg = float(np.mean([o for o, _ in bets]))
    # p-value: bootstrap-free normal approx on profit per bet
    profits = np.array([o * w - 1 for o, w in bets], float)
    se = profits.std() / math.sqrt(n)
    z = profits.mean() / se if se > 0 else 0
    pv = 2 * norm.sf(abs(z))
    n_tests += 1
    return {"group": group, "market": mname, "sel": selname, "n": n,
            "wr": round(wins / n, 4), "roi": round(roi, 4),
            "avg_odds": round(avg, 3), "p": float(pv)}

SCORES = [f"{i}-{j}" for i in range(7) for j in range(7) if i + j <= 6]
for gname, evs in SCAN_GROUPS:
    for s in SCORES:
        i, j = map(int, s.split("-"))
        r = roi_scan(gname, evs, "Score exact", s,
                     lambda e, i=i, j=j: e["sa"] == i and e["sb"] == j)
        if r:
            scan.append(r)
    for t in ["0", "1", "2", "3", "4", "5", "6"]:
        ti = int(t)
        if ti < 6:
            r = roi_scan(gname, evs, "Total de buts", t,
                         lambda e, ti=ti: e["total"] == ti)
        else:
            r = roi_scan(gname, evs, "Total de buts", t,
                         lambda e: e["total"] >= 6)   # assume 6 = 6+
            if r:
                r["sel"] = "6(=6+)"
        if r:
            scan.append(r)
    # also exact-6 settlement for "6"
    r = roi_scan(gname, evs, "Total de buts", "6", lambda e: e["total"] == 6)
    if r:
        r["sel"] = "6(=exactly6)"
        scan.append(r)

print(f"  scanned {len(scan)} (group,selection) cells")
print("  Top by ROI with n>=150 and p<0.05:")
for r in sorted([r for r in scan if r["n"] >= 150 and r["p"] < 0.05],
                key=lambda x: -x["roi"])[:25]:
    print(f"   {r['group']:18s} {r['market']:14s} {r['sel']:12s} n={r['n']:5d} "
          f"wr={r['wr']:.4f} roi={r['roi']:+.4f} odds={r['avg_odds']:7.3f} p={r['p']:.4g}")
print("  Worst by ROI (info):")
for r in sorted([r for r in scan if r["n"] >= 150 and r["p"] < 0.05],
                key=lambda x: x["roi"])[:10]:
    print(f"   {r['group']:18s} {r['market']:14s} {r['sel']:12s} n={r['n']:5d} "
          f"wr={r['wr']:.4f} roi={r['roi']:+.4f} odds={r['avg_odds']:7.3f} p={r['p']:.4g}")
out["roi_scan"] = scan
out["n_tests_part3"] = n_tests
with open("exports/wf4_crossleague_part3.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_crossleague_part3.json; n_tests part3 =", n_tests)
