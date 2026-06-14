# -*- coding: utf-8 -*-
"""WF4 - FTTS E1 nouvelles ligues: verifications + splits domestic/cups + buckets cote FTTS.

Verifs:
 1. taux goals_json NULL par ligue (events finis AVEC cotes) - reconcilier avec inventaire
 2. fetched_at du snapshot d'ouverture vs expected_start (pas d'info post-coup-d-envoi)
 3. spot-check settlement (10 events affiches)
Tests additionnels (comptes):
 - pooled domestic (8036/37/42/43/44) vs pooled cups (8056/60/65): home fav <=1.4 / <=1.5, FTTS '1'
 - away fav idem
 - buckets de cote FTTS f1 (1.0-1.2/1.2-1.4/1.4-1.6/1.6-2.0) avec home fav <=1.5 et sans filtre
 - split temporel 70/30 par expected_start de la strategie domestic home<=1.5
Sortie: exports/wf4_ftts_checks.json
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

DOMESTIC = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
            "InstantLeague-8043", "InstantLeague-8044"]
CUPS = ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW_LEAGUES = DOMESTIC + CUPS

e = create_engine(load_settings().db_url)
in_clause = "(" + ",".join("'%s'" % l for l in NEW_LEAGUES) + ")"

with e.connect() as c:
    # 1. goals_json null rate on finished events WITH odds
    q1 = text("""
    SELECT e.competition,
           SUM(CASE WHEN r.goals_json IS NULL OR r.goals_json='' OR r.goals_json='[]' THEN 1 ELSE 0 END),
           SUM(CASE WHEN r.score_a + r.score_b = 0 THEN 1 ELSE 0 END),
           COUNT(*)
    FROM events e JOIN results r ON r.event_id=e.id
    WHERE EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id)
      AND e.competition IN """ + in_clause + " GROUP BY e.competition")
    print("goals_json NULL/empty rate (events finis avec cotes):")
    null_rates = {}
    for comp, nnull, n00, ntot in c.execute(q1).fetchall():
        null_rates[comp] = dict(null_or_empty=int(nnull), zero_zero=int(n00), total=int(ntot))
        print("  %s: null/empty=%d, 0-0=%d, total=%d" % (comp, nnull, n00, ntot))

    # also on finished events WITHOUT odds (to explain inventory discrepancy)
    q1b = text("""
    SELECT e.competition,
           SUM(CASE WHEN r.goals_json IS NULL OR r.goals_json='' OR r.goals_json='[]' THEN 1 ELSE 0 END),
           COUNT(*)
    FROM events e JOIN results r ON r.event_id=e.id
    WHERE NOT EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=e.id)
      AND e.competition IN """ + in_clause + " GROUP BY e.competition")
    print("goals_json NULL/empty (events finis SANS cotes):")
    for comp, nnull, ntot in c.execute(q1b).fetchall():
        print("  %s: null/empty=%d / %d" % (comp, nnull, ntot))

    # 2. opening snapshot timing (colonne reelle: captured_at)
    q2 = text("""
    SELECT SUM(CASE WHEN o.captured_at <= e.expected_start THEN 1 ELSE 0 END), COUNT(*),
           MAX(julianday(o.captured_at) - julianday(e.expected_start)) * 24 * 60
    FROM events e
    JOIN results r ON r.event_id=e.id
    JOIN odds_snapshots o ON o.event_id=e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)
      AND e.competition IN """ + in_clause)
    pre, tot, worst = c.execute(q2).fetchone()
    print("opening snapshot fetched_at <= expected_start: %d/%d (worst lateness %.1f min)" % (pre, tot, worst or 0))

    # main dataset
    q = text("""
    SELECT e.id, e.competition, e.expected_start, o.odds_home, o.odds_draw, o.odds_away,
           o.extra_markets, r.score_a, r.score_b, r.goals_json
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o ON o.event_id = e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
      AND e.competition IN """ + in_clause)
    rows = c.execute(q).fetchall()


def settle_ftts(score_a, score_b, goals_json):
    total = int(score_a) + int(score_b)
    if total == 0:
        return "Pas de but"
    if not goals_json:
        return None
    try:
        g = json.loads(goals_json)
    except Exception:
        return None
    if not isinstance(g, list) or len(g) != total:
        return None
    firsts = [x for x in g if (x.get("homeScore", 0) + x.get("awayScore", 0)) == 1]
    if len(firsts) != 1:
        return None
    return {"Home": "1", "Away": "2"}.get(firsts[0].get("team"))


data = {}
unsettleable = 0
for r in rows:
    eid, comp, start, oh, od, oa, em_raw, sa, sb, gj = r
    if eid in data:
        continue
    try:
        em = json.loads(em_raw) if em_raw else {}
    except Exception:
        em = {}
    ftts = em.get("FTTS")
    if not ftts or "1" not in ftts or "2" not in ftts:
        continue
    outcome = settle_ftts(sa, sb, gj)
    if outcome is None:
        unsettleable += 1
        continue
    data[eid] = dict(comp=comp, start=str(start), oh=float(oh), oa=float(oa),
                     f1=float(ftts["1"]), f2=float(ftts["2"]), outcome=outcome,
                     sa=int(sa), sb=int(sb), gj=gj)

print("clean:", len(data), "unsettleable:", unsettleable)

# 3. spot-check settlement
print("\nSPOT-CHECK (10 events):")
for i, (eid, d) in enumerate(sorted(data.items())):
    if i >= 10:
        break
    g = json.loads(d["gj"]) if d["gj"] else []
    first = g[0] if g else None
    print("  ev%d %s %d-%d outcome=%s first_goal=%s f1=%.2f f2=%.2f" % (
        eid, d["comp"][-4:], d["sa"], d["sb"], d["outcome"],
        ("%s@%d" % (first["team"], first["minute"])) if first else "none", d["f1"], d["f2"]))


def evaluate(bets):
    n = len(bets)
    if n == 0:
        return dict(n=0)
    wins = sum(1 for o, w in bets if w)
    ret = sum((o - 1) if w else -1 for o, w in bets)
    avg_odds = sum(o for o, w in bets) / n
    var = sum(o - 1 for o, w in bets)
    z = ret / math.sqrt(var) if var > 0 else 0.0
    p_one = 0.5 * math.erfc(z / math.sqrt(2))
    implied = sum(1.0 / o for o, w in bets) / n
    return dict(n=n, wins=wins, wr=round(wins / n, 4), roi_pct=round(100 * ret / n, 2),
                avg_odds=round(avg_odds, 3), z=round(z, 3), p_one_sided=round(p_one, 6),
                implied_wr=round(implied, 4), calib_ratio=round(wins / n / implied, 3))


D = list(data.values())
results = {}
tests = 0


def run(name, sel, subset=None):
    global tests
    pool = subset if subset is not None else D
    bets = [b for b in (sel(d) for d in pool) if b is not None]
    results[name] = evaluate(bets)
    tests += 1


DOM = [d for d in D if d["comp"] in DOMESTIC]
CUP = [d for d in D if d["comp"] in CUPS]

for T in [1.4, 1.5]:
    run("dom_home_le%s_ftts1" % T, lambda d, T=T: (d["f1"], d["outcome"] == "1") if d["oh"] <= T else None, DOM)
    run("cup_home_le%s_ftts1" % T, lambda d, T=T: (d["f1"], d["outcome"] == "1") if d["oh"] <= T else None, CUP)
    run("dom_away_le%s_ftts2" % T, lambda d, T=T: (d["f2"], d["outcome"] == "2") if d["oa"] <= T else None, DOM)
    run("cup_away_le%s_ftts2" % T, lambda d, T=T: (d["f2"], d["outcome"] == "2") if d["oa"] <= T else None, CUP)
    run("dom_anyfav_le%s_fttsfav" % T, lambda d, T=T: (d["f1"], d["outcome"] == "1") if d["oh"] <= T
        else ((d["f2"], d["outcome"] == "2") if d["oa"] <= T else None), DOM)

# FTTS odds buckets (home side), with and without fav filter
FB = [(1.0, 1.2), (1.2, 1.4), (1.4, 1.6), (1.6, 2.0)]
for lo, hi in FB:
    run("fttsbucket_home_%s-%s_all" % (lo, hi),
        lambda d, lo=lo, hi=hi: (d["f1"], d["outcome"] == "1") if lo <= d["f1"] < hi else None)
    run("fttsbucket_home_%s-%s_fav150" % (lo, hi),
        lambda d, lo=lo, hi=hi: (d["f1"], d["outcome"] == "1") if lo <= d["f1"] < hi and d["oh"] <= 1.5 else None)
    run("fttsbucket_dom_home_%s-%s_fav150" % (lo, hi),
        lambda d, lo=lo, hi=hi: (d["f1"], d["outcome"] == "1") if lo <= d["f1"] < hi and d["oh"] <= 1.5 else None, DOM)

# temporal split 70/30 of domestic home<=1.5
DOMs = sorted(DOM, key=lambda d: d["start"])
cut = int(0.7 * len(DOMs))
run("dom_home_le1.5_ftts1_TRAIN70", lambda d: (d["f1"], d["outcome"] == "1") if d["oh"] <= 1.5 else None, DOMs[:cut])
run("dom_home_le1.5_ftts1_TEST30", lambda d: (d["f1"], d["outcome"] == "1") if d["oh"] <= 1.5 else None, DOMs[cut:])
# same for pooled all-new-leagues
ALLs = sorted(D, key=lambda d: d["start"])
cut2 = int(0.7 * len(ALLs))
run("all_home_le1.5_ftts1_TRAIN70", lambda d: (d["f1"], d["outcome"] == "1") if d["oh"] <= 1.5 else None, ALLs[:cut2])
run("all_home_le1.5_ftts1_TEST30", lambda d: (d["f1"], d["outcome"] == "1") if d["oh"] <= 1.5 else None, ALLs[cut2:])

out = dict(null_rates=null_rates, opening_pre_kickoff="%d/%d" % (pre, tot),
           n_clean=len(data), n_tests_scanned=tests, results=results)
with open("exports/wf4_ftts_checks.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print("\nn_tests_scanned (script 3):", tests)
print("%-42s %5s %6s %7s %7s %7s %9s %6s" % ("strategy", "n", "WR", "ROI%", "avgOdd", "z", "p(1s)", "calib"))
for k, v in results.items():
    if v.get("n", 0) == 0:
        print("%-42s %5d" % (k, 0))
        continue
    print("%-42s %5d %6.3f %7.2f %7.3f %7.2f %9.5f %6s" % (
        k, v["n"], v["wr"], v["roi_pct"], v["avg_odds"], v["z"], v["p_one_sided"], v["calib_ratio"]))
