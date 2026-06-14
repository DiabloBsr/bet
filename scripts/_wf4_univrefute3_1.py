# -*- coding: utf-8 -*-
"""WF4 adversarial refutation #3 of the cross-league poolability verdict.

Attacks:
A. LOOK-AHEAD: opening snapshots captured AFTER expected_start (50-85% per league,
   typically +1 min; instant leagues resolve in ~2 min). Tests:
   A1. within-event drift: events with BOTH pre-start and post-start snapshots ->
       does implied prob of the ACTUAL winner increase post-start? (leak detector)
   A2. sharpness: log-loss of devig 1X2 probs, early-open vs late-open, per league.
A3. CONSERVATIVE RE-VERDICT on strictly pre-start openings only:
   margins, global 1X2 calibration, O/U 3.5 calibration, bucket cells p<0.01.
B. SUB-PERIOD: new leagues split into two halves by expected_start; calibration each half.
C. Bootstrap CI on pooled-new O/U 3.5 calibration gap.
Read-only DB. Output: exports/wf4_univrefute3.json
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy.stats import binomtest, chi2 as chi2dist, ttest_ind

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
CHAMP = {"InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
         "InstantLeague-8043", "InstantLeague-8044"}
CUP = {"InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"}
REF = "InstantLeague-8035"

eng = create_engine(load_settings().db_url)
with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corrupted = set(int(k) for k in json.load(f)["events"].keys())

inlist = "(" + ",".join("'" + l + "'" for l in LEAGUES) + ")"

SQL = """
SELECT e.id, e.competition, e.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, o.captured_at
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = e.id
JOIN odds_snapshots o ON o.id = m.mid
WHERE e.competition IN %s
""" % inlist

rows = []
with eng.connect() as c:
    for r in c.execute(text(SQL)):
        rows.append(dict(r._mapping))

events = []
for r in rows:
    if r["id"] in corrupted:
        continue
    sa, sb = r["score_a"], r["score_b"]
    ha, hb = r["ht_score_a"], r["ht_score_b"]
    if sa is None or sb is None:
        continue
    if ha is not None and hb is not None and (ha > sa or hb > sb):
        continue
    gj = r["goals_json"]
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb:
                continue
        except Exception:
            pass
    oh, od, oa = r["odds_home"], r["odds_draw"], r["odds_away"]
    if not oh or not od or not oa or oh <= 1 or od <= 1 or oa <= 1:
        continue
    s = 1.0 / oh + 1.0 / od + 1.0 / oa
    out = "H" if sa > sb else ("A" if sa < sb else "D")
    x = r["extra_markets"]
    p_over35 = odds_under = None
    if x:
        try:
            d = json.loads(x) if isinstance(x, str) else x
            m = d.get("+/-") or {}
            u, o = m.get("< 3.5"), m.get("> 3.5")
            if u and o and u > 1 and o > 1:
                s2 = 1 / u + 1 / o
                p_over35 = (1 / o) / s2
                odds_under = u
        except Exception:
            pass
    events.append({
        "id": r["id"], "league": r["competition"], "ts": str(r["expected_start"]),
        "cap": str(r["captured_at"]), "late": str(r["captured_at"]) >= str(r["expected_start"]),
        "sa": sa, "sb": sb, "total": sa + sb,
        "oh": oh, "od": od, "oa": oa,
        "ph": (1/oh)/s, "pd": (1/od)/s, "pa": (1/oa)/s,
        "margin": s - 1.0, "out": out, "p_over35": p_over35, "odds_under": odds_under,
    })

print("clean events:", len(events))
out_json = {"n_events": len(events)}

# ============ A1. within-event pre/post-start drift toward actual winner ============
print("\n=== A1. WITHIN-EVENT PRE->POST-START DRIFT TOWARD WINNER ===")
ids = [e["id"] for e in events]
ev_by_id = {e["id"]: e for e in events}
SQL2 = """
SELECT o.event_id, o.captured_at, o.odds_home, o.odds_draw, o.odds_away, e.expected_start
FROM odds_snapshots o JOIN events e ON e.id = o.event_id
WHERE e.competition IN %s
ORDER BY o.event_id, o.id
""" % inlist
from collections import defaultdict
snaps = defaultdict(list)
with eng.connect() as c:
    for eid, cap, oh, od, oa, es in c.execute(text(SQL2)):
        if eid in ev_by_id and oh and od and oa and oh > 1 and od > 1 and oa > 1:
            snaps[eid].append((str(cap), oh, od, oa, str(es)))

drift = {"all": [], "by_league": defaultdict(list)}
n_span = 0
for eid, ss in snaps.items():
    e = ev_by_id[eid]
    pre = [s for s in ss if s[0] < s[4]]
    post = [s for s in ss if s[0] >= s[4]]
    if not pre or not post:
        continue
    n_span += 1
    def pwin(snap):
        oh, od, oa = snap[1], snap[2], snap[3]
        s = 1/oh + 1/od + 1/oa
        p = {"H": (1/oh)/s, "D": (1/od)/s, "A": (1/oa)/s}
        return p[e["out"]]
    d = pwin(post[-1]) - pwin(pre[-1])   # last post vs last pre
    drift["all"].append(d)
    drift["by_league"][e["league"]].append(d)
res_a1 = {"n_events_spanning": n_span}
if n_span > 30:
    a = np.array(drift["all"])
    se = a.std() / math.sqrt(len(a))
    res_a1.update({"mean_drift_pwinner": round(float(a.mean()), 5),
                   "se": round(float(se), 5),
                   "z": round(float(a.mean() / se), 2) if se > 0 else None,
                   "n": len(a)})
    print(f"  events with pre+post snapshots: {n_span}")
    print(f"  mean delta p(actual winner) post-start vs pre-start: {a.mean():+.5f} (se {se:.5f})")
    for l, v in sorted(drift["by_league"].items()):
        if len(v) >= 30:
            v = np.array(v)
            print(f"    {l}: n={len(v)} mean={v.mean():+.5f} (se {v.std()/math.sqrt(len(v)):.5f})")
else:
    print("  too few spanning events:", n_span)
out_json["A1_drift"] = res_a1

# ============ A2. sharpness early vs late ============
print("\n=== A2. LOG-LOSS / CALIBRATION EARLY-OPEN vs LATE-OPEN ===")
res_a2 = []
for l in LEAGUES + ["ALL-NEW"]:
    evs = [e for e in events if (e["league"] != REF if l == "ALL-NEW" else e["league"] == l)]
    early = [e for e in evs if not e["late"]]
    late = [e for e in evs if e["late"]]
    if len(early) < 100 or len(late) < 100:
        continue
    def ll(es):
        return np.array([-math.log(max(e["p" + e["out"].lower()], 1e-9)) for e in es])
    le, ll_ = ll(early), ll(late)
    t, p = ttest_ind(le, ll_, equal_var=False)
    # calibration gap obs-exp on home prob
    def gap(es):
        obs = sum(1 for e in es if e["out"] == "H") / len(es)
        exp = sum(e["ph"] for e in es) / len(es)
        return obs - exp
    res_a2.append({"league": l, "n_early": len(early), "n_late": len(late),
                   "logloss_early": round(float(le.mean()), 4),
                   "logloss_late": round(float(ll_.mean()), 4),
                   "p_diff": float(p),
                   "homegap_early": round(gap(early), 4),
                   "homegap_late": round(gap(late), 4)})
    r = res_a2[-1]
    print(f"  {l:20s} early n={r['n_early']:5d} LL={r['logloss_early']:.4f} | late n={r['n_late']:5d} "
          f"LL={r['logloss_late']:.4f} p={r['p_diff']:.3g} | homegap e={r['homegap_early']:+.4f} l={r['homegap_late']:+.4f}")
out_json["A2_sharpness"] = res_a2

# ============ A3. conservative re-verdict: strictly pre-start openings ============
print("\n=== A3. RE-VERDICT ON STRICTLY PRE-START OPENINGS ONLY ===")
early_ev = [e for e in events if not e["late"]]
print("  early-only events:", len(early_ev), "by league:",
      {l: sum(1 for e in early_ev if e["league"] == l) for l in LEAGUES})
res_a3 = {"n_early": len(early_ev)}
# margins
res_a3["margins"] = {l: round(float(np.mean([e["margin"] for e in early_ev if e["league"] == l])), 4)
                     for l in LEAGUES}
print("  margins:", res_a3["margins"])
# global calibration per league + pooled
glb = []
for l in LEAGUES + ["POOLED-ALL-NEW"]:
    evs = [e for e in early_ev if (e["league"] != REF if l == "POOLED-ALL-NEW" else e["league"] == l)]
    n = len(evs)
    if n < 100:
        continue
    obs = np.array([sum(1 for e in evs if e["out"] == o) for o in "HDA"], float)
    exp = np.array([sum(e["p" + o] for e in evs) for o in "hda"])
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    p = float(chi2dist.sf(chi2, 2))
    glb.append({"league": l, "n": n, "obs": [round(x/n, 4) for x in obs],
                "exp": [round(x/n, 4) for x in exp], "p": p})
    print(f"  GLOBAL {l:22s} n={n:5d} obs={glb[-1]['obs']} exp={glb[-1]['exp']} p={p:.4f}")
res_a3["global_calib"] = glb
# O/U 3.5 pooled-new early-only
evs = [e for e in early_ev if e["league"] != REF and e["p_over35"]]
obs = sum(1 for e in evs if e["total"] > 3.5)
exp = sum(e["p_over35"] for e in evs)
pv = binomtest(obs, len(evs), exp / len(evs)).pvalue
res_a3["ou35_poolednew"] = {"n": len(evs), "obs": round(obs/len(evs), 4),
                            "exp": round(exp/len(evs), 4), "p": float(pv)}
print(f"  O/U3.5 pooled-new early-only: n={len(evs)} obs={obs/len(evs):.4f} exp={exp/len(evs):.4f} p={pv:.4g}")
# bucket calib cells (own odds), count p<0.01
BUCKETS = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.72), (0.72, 1.01)]
cells = []
for l in LEAGUES:
    evs = [e for e in early_ev if e["league"] == l]
    for (a, b) in BUCKETS:
        ev2 = [e for e in evs if a <= e["ph"] < b]
        if len(ev2) < 80:
            continue
        n = len(ev2)
        obs = np.array([sum(1 for e in ev2 if e["out"] == o) for o in "HDA"], float)
        exp = np.array([sum(e["p" + o] for e in ev2) for o in "hda"])
        chi2 = float(np.sum((obs - exp) ** 2 / exp))
        p = float(chi2dist.sf(chi2, 2))
        cells.append({"league": l, "b": f"[{a}-{b})", "n": n, "p": p})
sig = [c for c in cells if c["p"] < 0.01]
res_a3["bucket_cells"] = {"n_cells": len(cells), "n_p_lt_01": len(sig), "sig": sig}
print(f"  bucket calib cells: {len(cells)} tested, p<0.01: {len(sig)} -> {sig}")
out_json["A3_early_only"] = res_a3

# ============ B. sub-period halves for new leagues ============
print("\n=== B. NEW LEAGUES SPLIT IN 2 HALVES (by expected_start) ===")
news = sorted([e for e in events if e["league"] != REF], key=lambda e: e["ts"])
half = len(news) // 2
res_b = []
for name, evs in [("H1", news[:half]), ("H2", news[half:])]:
    n = len(evs)
    obs = np.array([sum(1 for e in evs if e["out"] == o) for o in "HDA"], float)
    exp = np.array([sum(e["p" + o] for e in evs) for o in "hda"])
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    p = float(chi2dist.sf(chi2, 2))
    ev2 = [e for e in evs if e["p_over35"]]
    obs2 = sum(1 for e in ev2 if e["total"] > 3.5)
    exp2 = sum(e["p_over35"] for e in ev2)
    pv2 = binomtest(obs2, len(ev2), exp2 / len(ev2)).pvalue
    res_b.append({"half": name, "n": n, "span": (evs[0]["ts"], evs[-1]["ts"]),
                  "calib1x2_p": p, "obs": [round(x/n, 4) for x in obs],
                  "exp": [round(x/n, 4) for x in exp],
                  "ou35": {"n": len(ev2), "obs": round(obs2/len(ev2), 4),
                           "exp": round(exp2/len(ev2), 4), "p": float(pv2)}})
    r = res_b[-1]
    print(f"  {name} n={n} [{r['span'][0][:16]} -> {r['span'][1][:16]}] 1x2 obs={r['obs']} exp={r['exp']} p={p:.4f} | "
          f"OU35 obs={r['ou35']['obs']} exp={r['ou35']['exp']} p={r['ou35']['p']:.4f}")
out_json["B_halves"] = res_b

# ============ C. bootstrap CI pooled-new OU gap ============
print("\n=== C. BOOTSTRAP CI ON POOLED-NEW O/U 3.5 GAP (obs-exp) ===")
evs = [e for e in events if e["league"] != REF and e["p_over35"]]
y = np.array([1.0 if e["total"] > 3.5 else 0.0 for e in evs])
px = np.array([e["p_over35"] for e in evs])
rng = np.random.default_rng(42)
gaps = []
for _ in range(2000):
    idx = rng.integers(0, len(evs), len(evs))
    gaps.append(float(y[idx].mean() - px[idx].mean()))
gaps = np.array(gaps)
res_c = {"n": len(evs), "gap": round(float(y.mean() - px.mean()), 5),
         "ci95": [round(float(np.percentile(gaps, 2.5)), 5),
                  round(float(np.percentile(gaps, 97.5)), 5)]}
print(f"  n={len(evs)} gap={res_c['gap']:+.5f} CI95={res_c['ci95']}")
out_json["C_bootstrap_ou"] = res_c

with open("exports/wf4_univrefute3.json", "w", encoding="utf-8") as f:
    json.dump(out_json, f, ensure_ascii=False, indent=1)
print("\nsaved exports/wf4_univrefute3.json")
