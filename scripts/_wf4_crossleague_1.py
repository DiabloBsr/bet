# -*- coding: utf-8 -*-
"""WF4 cross-league universality — data load + 1X2 calibration by bucket + chi2 vs 8035.

Read-only on DB. Outputs exports/wf4_crossleague.json (partial, part 1).
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy.stats import chi2_contingency, skellam
from scipy.optimize import brentq

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

SQL = """
SELECT e.id, e.competition, e.expected_start,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
FROM events e
JOIN results r ON r.event_id = e.id
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m
     ON m.event_id = e.id
JOIN odds_snapshots o ON o.id = m.mid
WHERE e.competition IN :leagues
"""

rows = []
with eng.connect() as c:
    res = c.execute(text(SQL).bindparams(leagues=tuple(LEAGUES))
                    if False else text(SQL.replace(":leagues",
                        "(" + ",".join("'" + l + "'" for l in LEAGUES) + ")")))
    for r in res:
        rows.append(dict(r._mapping))

print("raw rows:", len(rows))

events = []
n_corrupt_excl = 0
n_guard_excl = 0
for r in rows:
    if r["id"] in corrupted:
        n_corrupt_excl += 1
        continue
    sa, sb = r["score_a"], r["score_b"]
    ha, hb = r["ht_score_a"], r["ht_score_b"]
    if sa is None or sb is None:
        n_guard_excl += 1
        continue
    # uniform integrity guard (new leagues not audited for corruption)
    if ha is not None and hb is not None and (ha > sa or hb > sb):
        n_guard_excl += 1
        continue
    gj = r["goals_json"]
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb:
                n_guard_excl += 1
                continue
        except Exception:
            pass
    oh, od, oa = r["odds_home"], r["odds_draw"], r["odds_away"]
    if not oh or not od or not oa or oh <= 1 or od <= 1 or oa <= 1:
        n_guard_excl += 1
        continue
    s = 1.0 / oh + 1.0 / od + 1.0 / oa
    ph, pd, pa = (1.0 / oh) / s, (1.0 / od) / s, (1.0 / oa) / s
    out = "H" if sa > sb else ("A" if sa < sb else "D")
    events.append({
        "id": r["id"], "league": r["competition"], "ts": r["expected_start"],
        "sa": sa, "sb": sb, "ha": ha, "hb": hb,
        "oh": oh, "od": od, "oa": oa, "ph": ph, "pd": pd, "pa": pa,
        "margin": s - 1.0, "out": out, "total": sa + sb,
        "extra": r["extra_markets"],
    })

print("clean events:", len(events), "| corrupt excl:", n_corrupt_excl,
      "| guard excl:", n_guard_excl)
from collections import Counter, defaultdict
cnt = Counter(e["league"] for e in events)
for l in LEAGUES:
    print(f"  {l}: {cnt[l]}")

# ---------- invert (lh, la) from devigged 1X2 via skellam ----------
def invert_lambdas(ph, pd, pa):
    """Solve P_draw = skellam.pmf(0,lh,la), P_home = skellam.sf(0,lh,la)."""
    # parametrize: for fixed mu=lh+la, find delta. 2D Newton via nested brentq.
    def pdraw_given(mu, delta):
        lh = (mu + delta) / 2.0
        la = (mu - delta) / 2.0
        if lh <= 0.01 or la <= 0.01:
            return None
        return lh, la
    def f_delta(delta, mu):
        v = pdraw_given(mu, delta)
        if v is None:
            return 1e9
        lh, la = v
        return skellam.sf(0, lh, la) - ph
    def f_mu(mu):
        # for this mu find delta matching home prob, return draw prob error
        lo, hi = -mu + 0.021, mu - 0.021
        try:
            d = brentq(f_delta, lo, hi, args=(mu,), xtol=1e-8)
        except ValueError:
            return None
        lh, la = (mu + d) / 2.0, (mu - d) / 2.0
        return skellam.pmf(0, lh, la) - pd, d
    lo_mu, hi_mu = 0.3, 9.0
    def g(mu):
        r = f_mu(mu)
        return 1e9 if r is None else r[0]
    try:
        mu = brentq(g, lo_mu, hi_mu, xtol=1e-7)
    except ValueError:
        return None
    _, d = f_mu(mu)
    return (mu + d) / 2.0, (mu - d) / 2.0

# invert for everyone (cache by rounded probs to speed up)
cache = {}
n_fail = 0
for e in events:
    key = (round(e["ph"], 4), round(e["pd"], 4))
    if key in cache:
        v = cache[key]
    else:
        v = invert_lambdas(e["ph"], e["pd"], e["pa"])
        cache[key] = v
    if v is None:
        n_fail += 1
        e["lh"] = e["la"] = e["mu"] = None
    else:
        e["lh"], e["la"] = v
        e["mu"] = v[0] + v[1]
print("lambda inversion failures:", n_fail)

# sanity: distribution of margins per league
print("\nmargin 1X2 per league (mean):")
for l in LEAGUES:
    ms = [e["margin"] for e in events if e["league"] == l]
    print(f"  {l}: {np.mean(ms):.4f} (n={len(ms)})")

# ---------- A. 1X2 outcome distribution by p_home bucket, chi2 vs 8035 ----------
BUCKETS = [(0.0, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60),
           (0.60, 0.72), (0.72, 1.01)]
def bidx(p):
    for i, (a, b) in enumerate(BUCKETS):
        if a <= p < b:
            return i
    return len(BUCKETS) - 1

tab = defaultdict(lambda: np.zeros(3))   # (league,bucket) -> [H,D,A]
for e in events:
    i = bidx(e["ph"])
    j = {"H": 0, "D": 1, "A": 2}[e["out"]]
    tab[(e["league"], i)][j] += 1

results_1x2 = []
n_tests = 0
groups = {l: [l] for l in LEAGUES if l != REF}
groups["pooled-champ-new"] = sorted(CHAMP)
groups["pooled-cup"] = sorted(CUP)
groups["pooled-all-new"] = sorted(CHAMP | CUP)

for gname, gleagues in groups.items():
    for i, (a, b) in enumerate(BUCKETS):
        ref = tab[(REF, i)]
        oth = np.sum([tab[(l, i)] for l in gleagues], axis=0)
        if ref.sum() < 30 or oth.sum() < 30:
            continue
        ct = np.vstack([ref, oth])
        # drop zero columns
        ct = ct[:, ct.sum(axis=0) > 0]
        chi2, p, dof, _ = chi2_contingency(ct)
        n_tests += 1
        results_1x2.append({
            "group": gname, "bucket": f"ph[{a:.2f}-{b:.2f})",
            "n_ref": int(ref.sum()), "n_grp": int(oth.sum()),
            "ref_HDA": [round(x / ref.sum(), 4) for x in ref],
            "grp_HDA": [round(x / oth.sum(), 4) for x in oth],
            "chi2": round(chi2, 3), "p": p,
        })

print("\n1X2 bucket chi2 vs 8035 — significant at p<0.01:")
for r in sorted(results_1x2, key=lambda x: x["p"])[:15]:
    flag = " ***" if r["p"] < 0.01 else ""
    print(f"  {r['group']:18s} {r['bucket']:16s} nref={r['n_ref']:5d} ngrp={r['n_grp']:5d} "
          f"ref={r['ref_HDA']} grp={r['grp_HDA']} p={r['p']:.4f}{flag}")

# ---------- B. global calibration per league: observed vs devig-expected ----------
calib = []
for l in LEAGUES + ["POOLED-CHAMP-NEW", "POOLED-CUP", "POOLED-ALL-NEW"]:
    if l == "POOLED-CHAMP-NEW":
        evs = [e for e in events if e["league"] in CHAMP]
    elif l == "POOLED-CUP":
        evs = [e for e in events if e["league"] in CUP]
    elif l == "POOLED-ALL-NEW":
        evs = [e for e in events if e["league"] != REF]
    else:
        evs = [e for e in events if e["league"] == l]
    n = len(evs)
    if n == 0:
        continue
    obs = np.array([sum(1 for e in evs if e["out"] == o) for o in "HDA"], float)
    exp = np.array([sum(e["p" + o.lower()] for e in evs) for o in "hda"])
    chi2 = float(np.sum((obs - exp) ** 2 / exp))
    from scipy.stats import chi2 as chi2dist
    p = float(chi2dist.sf(chi2, 2))
    n_tests += 1
    calib.append({"league": l, "n": n,
                  "obs_HDA": [round(x / n, 4) for x in obs],
                  "exp_HDA": [round(x / n, 4) for x in exp],
                  "chi2": round(chi2, 3), "p": p})
print("\nGlobal 1X2 calibration (obs vs devig expected):")
for r in calib:
    flag = " ***" if r["p"] < 0.01 else ""
    print(f"  {r['league']:22s} n={r['n']:5d} obs={r['obs_HDA']} exp={r['exp_HDA']} p={r['p']:.4f}{flag}")

# save intermediate
out = {"n_events_clean": len(events),
       "per_league_n": {l: cnt[l] for l in LEAGUES},
       "lambda_inversion_failures": n_fail,
       "margins": {l: round(float(np.mean([e['margin'] for e in events if e['league']==l])), 4) for l in LEAGUES},
       "chi2_1x2_buckets": results_1x2,
       "global_calibration": calib,
       "n_tests_part1": n_tests}
with open("exports/wf4_crossleague.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

# dump per-event compact for next scripts (JSON; ts kept as string)
with open("exports/_wf4_cl_events.json", "w", encoding="utf-8") as f:
    json.dump([{k: (str(e[k]) if k == "ts" else e[k])
                for k in ("id", "league", "ts", "sa", "sb", "ha", "hb",
                          "oh", "od", "oa", "ph", "pd", "pa", "lh", "la", "out")}
               for e in events], f)
# extra markets kept separately (id -> json string) for totals script
with open("exports/_wf4_cl_extra.json", "w", encoding="utf-8") as f:
    json.dump({str(e["id"]): e["extra"] for e in events}, f)
print("\nsaved exports/wf4_crossleague.json + caches; n_tests so far =", n_tests)
