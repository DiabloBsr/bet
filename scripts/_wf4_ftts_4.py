# -*- coding: utf-8 -*-
"""WF4 - Audit timing du snapshot d'ouverture + robustesse pre-kickoff de l'edge FTTS domestic.

1. Distribution de (captured_at - expected_start) en minutes, nouvelles ligues ET 8035.
2. Valeurs de odds_snapshots.status.
3. captured_at vs results.finished_at (le snapshot precede-t-il la fin du match ?).
4. Re-run de dom_home_le1.5_ftts1 / le1.4 sur le SOUS-ENSEMBLE captured_at <= expected_start
   et sur le sous-ensemble captured_at <= expected_start + 2 min.
Sortie: exports/wf4_ftts_timing.json
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

DOMESTIC = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
            "InstantLeague-8043", "InstantLeague-8044"]
CUPS = ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
NEW_LEAGUES = DOMESTIC + CUPS
in_new = "(" + ",".join("'%s'" % l for l in NEW_LEAGUES) + ")"

e = create_engine(load_settings().db_url)
with e.connect() as c:
    # status values
    print("status values:", c.execute(text(
        "SELECT status, COUNT(*) FROM odds_snapshots GROUP BY status")).fetchall())

    # lateness distribution helper
    def lateness(where):
        q = text("""
        SELECT CAST(ROUND((julianday(o.captured_at) - julianday(e.expected_start)) * 1440) AS INT) AS lag_min,
               COUNT(*)
        FROM events e
        JOIN results r ON r.event_id=e.id
        JOIN odds_snapshots o ON o.event_id=e.id
        WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)
          AND """ + where + " GROUP BY lag_min ORDER BY lag_min")
        return c.execute(q).fetchall()

    print("\nlag (min) opening snapshot vs expected_start - NOUVELLES LIGUES:")
    lag_new = lateness("e.competition IN " + in_new)
    for lag, n in lag_new:
        print("  %+4d min: %d" % (lag, n))

    print("\nlag (min) - 8035:")
    lag_8035 = lateness("e.competition = 'InstantLeague-8035'")
    # compress: bucket
    buckets = {}
    for lag, n in lag_8035:
        b = max(min(lag, 20), -60)
        buckets[b] = buckets.get(b, 0) + n
    for b in sorted(buckets):
        print("  %+4d min: %d" % (b, buckets[b]))

    # snapshot vs finished_at
    q3 = text("""
    SELECT SUM(CASE WHEN o.captured_at < r.finished_at THEN 1 ELSE 0 END), COUNT(*)
    FROM events e
    JOIN results r ON r.event_id=e.id
    JOIN odds_snapshots o ON o.event_id=e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)
      AND e.competition IN """ + in_new)
    pre_fin, tot = c.execute(q3).fetchone()
    print("\nopening snapshot captured BEFORE finished_at: %d/%d" % (pre_fin, tot))

    # do odds of late snapshots differ? (would indicate live odds) -> compare 1X2 overround
    q4 = text("""
    SELECT AVG(1.0/o.odds_home + 1.0/o.odds_draw + 1.0/o.odds_away),
           AVG(CASE WHEN o.captured_at <= e.expected_start THEN 1.0 ELSE 0.0 END)
    FROM events e
    JOIN results r ON r.event_id=e.id
    JOIN odds_snapshots o ON o.event_id=e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id=e.id)
      AND e.competition IN """ + in_new)
    print("avg 1X2 overround (all openings):", c.execute(q4).fetchone())

    # main dataset with timing
    q = text("""
    SELECT e.id, e.competition, e.expected_start, o.captured_at, r.finished_at,
           o.odds_home, o.odds_away, o.extra_markets, r.score_a, r.score_b, r.goals_json
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o ON o.event_id = e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
      AND e.competition IN """ + in_new)
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


from datetime import datetime

def parse_dt(s):
    s = str(s)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(s)


data = []
for r in rows:
    eid, comp, start, cap, fin, oh, oa, em_raw, sa, sb, gj = r
    try:
        em = json.loads(em_raw) if em_raw else {}
    except Exception:
        em = {}
    ftts = em.get("FTTS")
    if not ftts or "1" not in ftts or "2" not in ftts:
        continue
    outcome = settle_ftts(sa, sb, gj)
    if outcome is None:
        continue
    lag_min = (parse_dt(cap) - parse_dt(start)).total_seconds() / 60.0
    data.append(dict(comp=comp, oh=float(oh), oa=float(oa), f1=float(ftts["1"]),
                     f2=float(ftts["2"]), outcome=outcome, lag=lag_min))


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
                calib_ratio=round(wins / n / implied, 3))


results = {}
tests = 0


def run(name, pool, T, side="home"):
    global tests
    if side == "home":
        bets = [(d["f1"], d["outcome"] == "1") for d in pool if d["oh"] <= T]
    else:
        bets = [(d["f2"], d["outcome"] == "2") for d in pool if d["oa"] <= T]
    results[name] = evaluate(bets)
    tests += 1


DOM = [d for d in data if d["comp"] in DOMESTIC]
for label, pool in [("strictpre", [d for d in DOM if d["lag"] <= 0]),
                    ("pre2min", [d for d in DOM if d["lag"] <= 2]),
                    ("late", [d for d in DOM if d["lag"] > 2])]:
    for T in (1.4, 1.5):
        run("dom_home_le%s_ftts1_%s" % (T, label), pool, T)

# same check pooled all-new
ALL = data
for label, pool in [("strictpre", [d for d in ALL if d["lag"] <= 0]),
                    ("late", [d for d in ALL if d["lag"] > 0])]:
    run("all_home_le1.5_ftts1_%s" % label, pool, 1.5)

out = dict(lag_new=[[int(l), int(n)] for l, n in lag_new],
           pre_finished="%d/%d" % (pre_fin, tot), n_tests_scanned=tests, results=results)
with open("exports/wf4_ftts_timing.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print("\nn_tests_scanned (script 4):", tests)
print("%-42s %5s %6s %7s %7s %7s %9s %6s" % ("strategy", "n", "WR", "ROI%", "avgOdd", "z", "p(1s)", "calib"))
for k, v in results.items():
    if v.get("n", 0) == 0:
        print("%-42s %5d" % (k, 0))
        continue
    print("%-42s %5d %6.3f %7.2f %7.3f %7.2f %9.5f %6s" % (
        k, v["n"], v["wr"], v["roi_pct"], v["avg_odds"], v["z"], v["p_one_sided"], v["calib_ratio"]))
