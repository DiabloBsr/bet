# -*- coding: utf-8 -*-
"""WF4 cross-league part 6 — final numbers.

1. Placeholder discriminator: len(goals_json) - FT_sum for GOALS_JSON_VS_FT corrupted.
2. FTTS '1' (home<=1.5) pooled NEW-ERA CHAMPIONSHIPS (8035-recent + 8036/37/42/43/44).
3. Universal new-era score-grid ratios (pooled 8035-recent + all new).
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import norm, poisson, binomtest
from sqlalchemy import create_engine, text
from scraper.config import load_settings

CHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"}
REF = "InstantLeague-8035"
NEWWIN = "2026-06-12 00:00:00"

with open("exports/_wf4_cl_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)
with open("exports/_wf4_cl_extra.json", "r", encoding="utf-8") as f:
    extra = json.load(f)
for e in events:
    e["total"] = e["sa"] + e["sb"]

eng = create_engine(load_settings().db_url)
out = {}
n_tests = 0

# 1. placeholder discriminator
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    cj = json.load(f)
ids_ft = [int(k) for k, v in cj["events"].items() if "GOALS_JSON_VS_FT" in v]
with eng.connect() as c:
    rows = c.execute(text(
        "SELECT r.score_a, r.score_b, r.goals_json FROM results r WHERE r.event_id IN (" +
        ",".join(map(str, ids_ft)) + ")")).fetchall()
diffs = []
for sa, sb, raw in rows:
    if not raw:
        continue
    try:
        gl = json.loads(raw)
    except Exception:
        continue
    if isinstance(gl, list) and len(gl) > 0:
        diffs.append(len(gl) - (sa + sb))
pos = sum(1 for d in diffs if d > 0)
neg = sum(1 for d in diffs if d < 0)
print(f"1. GOALS_JSON_VS_FT: n={len(diffs)}, len(gj)>FT: {pos} ({100*pos/len(diffs):.0f}%), "
      f"len(gj)<FT: {neg} -> {'FT-placeholder (gj=truth)' if pos/len(diffs)>0.8 else 'mixed/alien'}")
out["placeholder_check"] = {"n": len(diffs), "gj_gt_ft": pos, "gj_lt_ft": neg}

# 2. FTTS pooled new-era championships
with eng.connect() as c:
    gjmap = {r[0]: r[1] for r in c.execute(text("SELECT event_id, goals_json FROM results"))}
def first_team(e):
    if e["total"] == 0:
        return "None"
    raw = gjmap.get(e["id"])
    if not raw:
        return None
    try:
        gl = json.loads(raw)
        if not isinstance(gl, list) or len(gl) == 0:
            return None
        return sorted(gl, key=lambda x: (x["minute"], x["homeScore"] + x["awayScore"]))[0]["team"]
    except Exception:
        return None
def ftts_roi(label, evs):
    global n_tests
    bets = []
    for e in evs:
        if e["oh"] > 1.5:
            continue
        x = extra.get(str(e["id"]))
        if not x:
            continue
        try:
            f2 = (json.loads(x) if isinstance(x, str) else x).get("FTTS") or {}
        except Exception:
            continue
        o = f2.get("1")
        if not o or o <= 1 or o >= 99.5:
            continue
        ft = first_team(e)
        if ft is None:
            continue
        bets.append((o, ft == "Home"))
    n = len(bets)
    wr = sum(w for _, w in bets) / n
    profits = np.array([o * w - 1 for o, w in bets])
    roi = float(profits.mean())
    se = profits.std() / math.sqrt(n)
    pv = 2 * norm.sf(abs(roi / se))
    n_tests += 1
    r = {"label": label, "n": n, "wr": round(wr, 4), "roi": round(roi, 4),
         "avg_odds": round(float(np.mean([o for o, _ in bets])), 3), "p": float(pv)}
    print(f"2. {label}: n={n} wr={wr:.4f} roi={roi:+.4f} odds={r['avg_odds']} p={pv:.4g}")
    return r
out["ftts_newera_champ"] = ftts_roi(
    "FTTS1 new-era champs (8035rec+5 ligues)",
    [e for e in events if e["league"] in CHAMP or (e["league"] == REF and e["ts"] >= NEWWIN)])
out["ftts_allchamp_full"] = ftts_roi(
    "FTTS1 all champs incl 8035-old",
    [e for e in events if e["league"] in CHAMP or e["league"] == REF])

# 3. universal new-era grid ratios
evs = [e for e in events if e["lh"] and (e["league"] != REF or e["ts"] >= NEWWIN)]
lh = np.array([e["lh"] for e in evs]); la = np.array([e["la"] for e in evs])
n = len(evs)
print(f"3. NEW-ERA pooled (n={n}) obs/exp grid ratios:")
ratios = {}
for (i, j) in [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (2, 0), (0, 2), (3, 3)]:
    exp = float(np.sum(poisson.pmf(i, lh) * poisson.pmf(j, la)))
    obs = sum(1 for e in evs if e["sa"] == i and e["sb"] == j)
    pv = binomtest(obs, n, exp / n).pvalue
    n_tests += 1
    ratios[f"{i}-{j}"] = {"obs": obs, "exp": round(exp, 1), "ratio": round(obs / exp, 3),
                          "p": float(pv)}
    print(f"   {i}-{j}: obs={obs} exp={exp:.1f} ratio={obs/exp:.3f} p={pv:.3g}")
exp_d = float(np.sum(sum(poisson.pmf(k, lh) * poisson.pmf(k, la) for k in range(11))))
obs_d = sum(1 for e in evs if e["sa"] == e["sb"])
n_tests += 1
ratios["DRAWS"] = {"obs": obs_d, "exp": round(exp_d, 1), "ratio": round(obs_d / exp_d, 3),
                   "p": float(binomtest(obs_d, n, exp_d / n).pvalue)}
print(f"   DRAWS: ratio={obs_d/exp_d:.3f} p={ratios['DRAWS']['p']:.3g}")
out["newera_grid_ratios"] = {"n": n, "cells": ratios}
out["n_tests_part6"] = n_tests
with open("exports/wf4_univrefute_part6_rerun.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved exports/wf4_univrefute_part6_rerun.json; n_tests part6 =", n_tests)
