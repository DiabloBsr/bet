# -*- coding: utf-8 -*-
# WF4 ROUND-STRUCTURE - extraction: per-event opening odds + lambdas + results
# READ-ONLY DB. Cache -> scripts/_wf4_roundstruct_data.pkl
import sys, json, pickle, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

e = create_engine(load_settings().db_url)

with open("exports/corrupted_events.json", "r", encoding="utf-8") as f:
    corr = json.load(f)
CORRUPT = set(int(k) for k in corr["events"].keys())

LEAGUES = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
           "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
           "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]

SQL = """
SELECT ev.id, ev.competition, ev.expected_start, ev.round_info, ev.team_a, ev.team_b,
       r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json,
       o.odds_home, o.odds_draw, o.odds_away
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots o ON o.event_id = ev.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
  AND ev.competition IN ({comps})
"""

GMAX = 13
def grid_probs(lh, la):
    ph = poisson.pmf(np.arange(GMAX + 1), lh)
    pa = poisson.pmf(np.arange(GMAX + 1), la)
    return np.outer(ph, pa)

def invert_lambdas(oh, od, oa):
    imp = np.array([1 / oh, 1 / od, 1 / oa])
    fair = imp / imp.sum()
    def resid(x):
        lh, la = np.exp(x)
        g = grid_probs(lh, la)
        return [np.tril(g, -1).sum() - fair[0], np.trace(g) - fair[1]]
    diff0 = math.log(max(fair[0], 1e-6) / max(fair[2], 1e-6)) * 0.55
    x0 = [math.log(max(0.2, 1.4 + diff0 / 2)), math.log(max(0.2, 1.4 - diff0 / 2))]
    sol = least_squares(resid, x0, xtol=1e-12, ftol=1e-12)
    lh, la = np.exp(sol.x)
    err = max(abs(v) for v in resid(sol.x))
    return float(lh), float(la), float(err)

def main():
    comps = ",".join("'" + c + "'" for c in LEAGUES)
    with e.connect() as conn:
        rows = conn.execute(text(SQL.format(comps=comps))).fetchall()
    print("raw rows:", len(rows))

    stats = {"corrupt_excl": 0, "guard_ht": 0, "guard_goals": 0, "dupe_excl": 0,
             "bad_odds": 0, "inv_fail": 0}
    seen_fixture = {}   # (comp, expected_start, team_a, team_b) -> id kept
    recs = []
    for (eid, comp, est, rnd, ta, tb, sa, sb, hta, htb, gj, oh, od, oa) in rows:
        if eid in CORRUPT:
            stats["corrupt_excl"] += 1; continue
        if oh is None or od is None or oa is None or oh <= 1 or od <= 1 or oa <= 1:
            stats["bad_odds"] += 1; continue
        # corruption guards (new leagues not audited)
        if hta is not None and htb is not None and (hta > sa or htb > sb):
            stats["guard_ht"] += 1; continue
        if gj:
            try:
                gl = json.loads(gj)
                if isinstance(gl, list) and len(gl) > 0 and len(gl) != sa + sb:
                    stats["guard_goals"] += 1; continue
            except Exception:
                pass
        key = (comp, est, ta, tb)
        if key in seen_fixture:
            stats["dupe_excl"] += 1; continue
        seen_fixture[key] = eid
        recs.append(dict(id=eid, comp=comp, est=est, rnd=rnd, ta=ta, tb=tb,
                         sa=sa, sb=sb, hta=hta, htb=htb,
                         oh=float(oh), od=float(od), oa=float(oa)))
    print("kept:", len(recs), stats)

    # invert lambdas
    nfail = 0
    for r in recs:
        lh, la, err = invert_lambdas(r["oh"], r["od"], r["oa"])
        if err > 1e-6:
            nfail += 1
        r["lh"], r["la"], r["inv_err"] = lh, la, err
    print("inversion err>1e-6:", nfail)

    with open("scripts/_wf4_roundstruct_data.pkl", "wb") as f:
        pickle.dump(recs, f)
    # round-instance summary
    from collections import Counter, defaultdict
    g = defaultdict(int)
    for r in recs:
        g[(r["comp"], r["est"])] += 1
    sizes = defaultdict(Counter)
    for (comp, est), n in g.items():
        sizes[comp][n] += 1
    for comp in sorted(sizes):
        print(comp, "rounds:", sum(sizes[comp].values()), "size-dist:", dict(sorted(sizes[comp].items())))

if __name__ == "__main__":
    main()
