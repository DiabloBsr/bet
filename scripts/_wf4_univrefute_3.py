# -*- coding: utf-8 -*-
"""Adversarial refutation of the 'universal new-era grid ratios' finding.

A. Per-group obs/exp grid ratios (8035-old, 8035-recent, each champ, each cup,
   pooled-champ, pooled-cup) + homogeneity chi2 across new-era groups.
B. Cup settlement forensics: goal minutes > 90, HT>FT, draws in cup rounds.
C. Guard-exclusion census per era/league (was new-era filtered by the same
   corruption-shaped guard?).
D. Old-RAW (corrupted restored) grid ratios on 8035: do they land on new-era?
E. Intra-new-era stability: time quartiles + margin halves.
READ-ONLY on DB. Outputs exports/wf4_univrefute.json
"""
import sys, json, math
sys.path.insert(0, ".")
import numpy as np
from scipy.stats import poisson, binomtest, chi2 as chi2dist, skellam
from scipy.optimize import brentq
from sqlalchemy import create_engine, text
from collections import Counter, defaultdict

CHAMP = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"]
CUP = ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
REF = "InstantLeague-8035"
NEWWIN = "2026-06-12 00:00:00"
CELLS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2), (2, 0), (0, 2), (3, 3)]

with open("exports/_wf4_cl_events.json", "r", encoding="utf-8") as f:
    events = json.load(f)
for e in events:
    e["total"] = e["sa"] + e["sb"]
evs_all = [e for e in events if e["lh"]]

eng = create_engine(load_settings().db_url) if False else None
from scraper.config import load_settings
eng = create_engine(load_settings().db_url)

out = {}

def grid_ratios(evs):
    lh = np.array([e["lh"] for e in evs]); la = np.array([e["la"] for e in evs])
    n = len(evs)
    res = {}
    for (i, j) in CELLS:
        exp = float(np.sum(poisson.pmf(i, lh) * poisson.pmf(j, la)))
        obs = sum(1 for e in evs if e["sa"] == i and e["sb"] == j)
        res[f"{i}-{j}"] = (obs, exp)
    exp_d = float(np.sum(sum(poisson.pmf(k, lh) * poisson.pmf(k, la) for k in range(11))))
    obs_d = sum(1 for e in evs if e["sa"] == e["sb"])
    res["DRAWS"] = (obs_d, exp_d)
    return n, res

groups = {}
groups["8035-old"] = [e for e in evs_all if e["league"] == REF and e["ts"] < NEWWIN]
groups["8035-recent"] = [e for e in evs_all if e["league"] == REF and e["ts"] >= NEWWIN]
for l in CHAMP + CUP:
    groups[l.replace("InstantLeague-", "")] = [e for e in evs_all if e["league"] == l]
groups["pooled-champ"] = [e for e in evs_all if e["league"] in CHAMP]
groups["pooled-cup"] = [e for e in evs_all if e["league"] in CUP]

print("A. PER-GROUP GRID RATIOS (obs/exp, binom p)")
keycells = ["0-0", "1-0", "0-1", "1-1", "2-1", "1-2", "2-2", "3-3", "DRAWS"]
tableA = {}
for g, evs in groups.items():
    n, res = grid_ratios(evs)
    row = {}
    for c in keycells:
        obs, exp = res[c]
        pv = binomtest(obs, n, min(exp / n, 1.0)).pvalue if exp > 0 else 1.0
        row[c] = {"obs": obs, "exp": round(exp, 1),
                  "ratio": round(obs / exp, 3) if exp > 0 else None, "p": float(pv)}
    tableA[g] = {"n": n, "cells": row}
    print(f"  {g:14s} n={n:6d} " + " ".join(
        f"{c}:{row[c]['ratio']:.2f}" for c in keycells))
out["per_group_ratios"] = tableA

# homogeneity of ratio across NEW-ERA groups (Poisson approx: var(r)=obs/exp^2)
print("\n  homogeneity across new-era groups (8035-recent + 8 leagues):")
newg = ["8035-recent"] + [l.replace("InstantLeague-", "") for l in CHAMP + CUP]
hom = {}
for c in keycells:
    rs, ws = [], []
    for g in newg:
        obs, exp = tableA[g]["cells"][c]["obs"], tableA[g]["cells"][c]["exp"]
        if exp < 10:
            continue
        r = obs / exp
        var = max(obs, 1) / exp ** 2
        rs.append(r); ws.append(1 / var)
    rs, ws = np.array(rs), np.array(ws)
    rbar = float(np.sum(rs * ws) / np.sum(ws))
    chi2 = float(np.sum(ws * (rs - rbar) ** 2))
    dof = len(rs) - 1
    p = float(chi2dist.sf(chi2, dof))
    hom[c] = {"pooled_r": round(rbar, 3), "chi2": round(chi2, 2), "dof": dof, "p": p,
              "per_group": [round(x, 3) for x in rs]}
    print(f"   {c:6s} rbar={rbar:.3f} chi2={chi2:.1f} dof={dof} p={p:.4g} {'***' if p<0.01 else ''}")
out["homogeneity_newera"] = hom

# champ-new-era only pooled (8035-recent + 5 champs) vs cup pooled, headline cells
champ_new = groups["8035-recent"] + groups["pooled-champ"]
n_cn, res_cn = grid_ratios(champ_new)
n_cu, res_cu = grid_ratios(groups["pooled-cup"])
print("\n  CHAMP-NEW-ERA pooled vs CUP pooled:")
cmp_rows = {}
for c in keycells:
    o1, e1 = res_cn[c]; o2, e2 = res_cu[c]
    r1, r2 = o1 / e1, o2 / e2
    # 2-sample ratio z-test (Poisson)
    se = math.sqrt(o1 / e1 ** 2 + o2 / e2 ** 2) if o1 > 0 and o2 > 0 else None
    z = (r1 - r2) / se if se else None
    from scipy.stats import norm
    pv = 2 * norm.sf(abs(z)) if z is not None else 1.0
    cmp_rows[c] = {"champ_r": round(r1, 3), "cup_r": round(r2, 3), "p": float(pv)}
    print(f"   {c:6s} champ={r1:.3f} (n={n_cn}) cup={r2:.3f} (n={n_cu}) p={pv:.4g} {'***' if pv<0.01 else ''}")
out["champ_vs_cup"] = {"n_champ": n_cn, "n_cup": n_cu, "cells": cmp_rows}

# B. cup settlement forensics
print("\nB. CUP SETTLEMENT FORENSICS")
ids_by_l = defaultdict(list)
for e in events:
    ids_by_l[e["league"]].append(e["id"])
fore = {}
with eng.connect() as c:
    for l in [REF] + CHAMP + CUP:
        ids = ids_by_l[l]
        rows = c.execute(text(
            "SELECT r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json, e.round_info "
            "FROM results r JOIN events e ON e.id=r.event_id WHERE r.event_id IN (" +
            ",".join(map(str, ids)) + ")")).fetchall()
        n_gt90 = 0; n_goals = 0; n_htgtft = 0; n_draw = 0; n = 0
        minute_max = 0
        for sa, sb, ha, hb, gj, ri in rows:
            n += 1
            if sa == sb:
                n_draw += 1
            if ha is not None and (ha > sa or hb > sb):
                n_htgtft += 1
            if gj:
                try:
                    gl = json.loads(gj)
                    for g in gl:
                        n_goals += 1
                        if g["minute"] > 90:
                            n_gt90 += 1
                        minute_max = max(minute_max, g["minute"])
                except Exception:
                    pass
        fore[l] = {"n": n, "draw_rate": round(n_draw / n, 4), "goals": n_goals,
                   "goals_gt90": n_gt90, "minute_max": minute_max, "ht_gt_ft": n_htgtft}
        print(f"  {l}: n={n} draws={n_draw/n:.3f} goals>90min={n_gt90}/{n_goals} max_min={minute_max} ht>ft={n_htgtft}")
out["cup_settlement"] = fore

# C. guard-exclusion census (rerun part-1 query, classify)
print("\nC. GUARD-EXCLUSION CENSUS PER LEAGUE/ERA")
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corrupted = set(int(k) for k in json.load(f)["events"].keys())
SQL = """
SELECT e.id, e.competition, e.expected_start, r.score_a, r.score_b,
       r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = e.id
JOIN odds_snapshots o ON o.id = m.mid
WHERE e.competition IN (%s)
""" % ",".join("'" + l + "'" for l in [REF] + CHAMP + CUP)
census = defaultdict(Counter)
excl_scores = defaultdict(Counter)
with eng.connect() as c:
    for r in c.execute(text(SQL)):
        l = r.competition
        era = ("old" if str(r.expected_start) < NEWWIN else "recent") if l == REF else "new"
        key = f"{l}|{era}"
        if r.id in corrupted:
            census[key]["corrupted_excl"] += 1
            continue
        sa, sb = r.score_a, r.score_b
        if sa is None:
            census[key]["null_score"] += 1
            continue
        if r.ht_score_a is not None and (r.ht_score_a > sa or r.ht_score_b > sb):
            census[key]["ht_gt_ft"] += 1
            excl_scores[key][f"{sa}-{sb}"] += 1
            continue
        if r.goals_json:
            try:
                gl = json.loads(r.goals_json)
                if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb:
                    census[key]["gj_mismatch"] += 1
                    excl_scores[key][f"{sa}-{sb}"] += 1
                    continue
            except Exception:
                pass
        if not r.odds_home or r.odds_home <= 1 or not r.odds_draw or r.odds_draw <= 1 \
           or not r.odds_away or r.odds_away <= 1:
            census[key]["bad_odds"] += 1
            continue
        census[key]["kept"] += 1
for k in sorted(census):
    print(f"  {k}: {dict(census[k])}  excl_scores_top={excl_scores[k].most_common(5)}")
out["guard_census"] = {k: dict(v) for k, v in census.items()}
out["guard_excl_scores"] = {k: dict(v) for k, v in excl_scores.items()}

# D. old-RAW ratios (corrupted restored at recorded FT) on 8035
print("\nD. 8035 OLD-RAW GRID RATIOS (corrupted restored, recorded FT)")
def invert_lambdas(ph, pd, pa):
    def f_delta(delta, mu):
        lh = (mu + delta) / 2.0; la = (mu - delta) / 2.0
        if lh <= 0.01 or la <= 0.01:
            return 1e9
        return skellam.sf(0, lh, la) - ph
    def f_mu(mu):
        lo, hi = -mu + 0.021, mu - 0.021
        try:
            d = brentq(f_delta, lo, hi, args=(mu,), xtol=1e-8)
        except ValueError:
            return None
        lh, la = (mu + d) / 2.0, (mu - d) / 2.0
        return skellam.pmf(0, lh, la) - pd, d
    def g(mu):
        r = f_mu(mu)
        return 1e9 if r is None else r[0]
    try:
        mu = brentq(g, 0.3, 9.0, xtol=1e-7)
    except ValueError:
        return None
    _, d = f_mu(mu)
    return (mu + d) / 2.0, (mu - d) / 2.0

with eng.connect() as c:
    rows = c.execute(text(
        "SELECT e.id, e.expected_start, r.score_a, r.score_b, o.odds_home, o.odds_draw, o.odds_away "
        "FROM events e JOIN results r ON r.event_id=e.id "
        "JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id=e.id "
        "JOIN odds_snapshots o ON o.id=m.mid "
        "WHERE e.competition='" + REF + "' AND e.id IN (" +
        ",".join(map(str, corrupted)) + ")")).fetchall()
cache = {}
restored = []
for eid, ts, sa, sb, oh, od, oa in rows:
    if not oh or oh <= 1 or not od or od <= 1 or not oa or oa <= 1 or sa is None:
        continue
    if str(ts) >= NEWWIN:
        continue
    s = 1 / oh + 1 / od + 1 / oa
    ph, pd, pa = (1 / oh) / s, (1 / od) / s, (1 / oa) / s
    key = (round(ph, 4), round(pd, 4))
    if key not in cache:
        cache[key] = invert_lambdas(ph, pd, pa)
    v = cache[key]
    if v is None:
        continue
    restored.append({"sa": sa, "sb": sb, "lh": v[0], "la": v[1]})
print(f"  corrupted 8035-old restored with odds: {len(restored)}")
oldraw = groups["8035-old"] + restored
for label, evs in [("8035-old (audited)", groups["8035-old"]),
                   ("8035-old-RAW (+corrupted)", oldraw),
                   ("8035-recent", groups["8035-recent"])]:
    n, res = grid_ratios(evs)
    msg = []
    rows_out = {}
    for c in keycells:
        obs, exp = res[c]
        rows_out[c] = round(obs / exp, 3)
        msg.append(f"{c}:{obs/exp:.3f}")
    out[f"ratios_{label}"] = {"n": n, **rows_out}
    print(f"  {label:26s} n={n:5d} " + " ".join(msg))

# E. intra-new-era stability: time quartiles + margin halves
print("\nE. INTRA-NEW-ERA STABILITY")
newera = [e for e in evs_all if e["league"] != REF or e["ts"] >= NEWWIN]
newera_sorted = sorted(newera, key=lambda e: e["ts"])
qs = np.array_split(newera_sorted, 4)
stab = {}
for qi, q in enumerate(qs):
    q = list(q)
    n, res = grid_ratios(q)
    row = {c: round(res[c][0] / res[c][1], 3) for c in keycells}
    stab[f"Q{qi+1}"] = {"n": n, "ts_range": [q[0]["ts"], q[-1]["ts"]], **row}
    print(f"  Q{qi+1} n={n} [{q[0]['ts']} .. {q[-1]['ts']}] " +
          " ".join(f"{c}:{row[c]:.2f}" for c in keycells))
margins = [(1/e["oh"] + 1/e["od"] + 1/e["oa"] - 1) for e in newera]
med = float(np.median(margins))
lo = [e for e, m in zip(newera, margins) if m < med]
hi = [e for e, m in zip(newera, margins) if m >= med]
for label, evs in [("margin<med", lo), ("margin>=med", hi)]:
    n, res = grid_ratios(evs)
    row = {c: round(res[c][0] / res[c][1], 3) for c in keycells}
    stab[label] = {"n": n, **row}
    print(f"  {label} n={n} " + " ".join(f"{c}:{row[c]:.2f}" for c in keycells))
out["stability"] = stab
out["margin_median_newera"] = med

with open("exports/wf4_univrefute.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_univrefute.json")
