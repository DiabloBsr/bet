# -*- coding: utf-8 -*-
"""WF4 - Edge E1 (FTTS favori) sur les 8 nouvelles ligues.

- Cotes d'OUVERTURE (snapshot MIN(id) par event).
- Settlement FTTS depuis goals_json (premier but = entree cumulative h+a==1),
  garde-fou: skip si goals_json invalide ou len != score_a+score_b (sauf 0-0 -> 'Pas de but').
- Strategies testees (comptees dans n_tests_scanned):
  * E1 replique: home fav <=1.50 -> FTTS '1' (pooled + 8 ligues)
  * miroir: away fav <=1.50 -> FTTS '2' (pooled + 8 ligues)
  * bins de cote fav [1.0-1.3/1.3-1.5/1.5-1.7/1.7-2.0] home & away (pooled)
  * contrarian: fav <=1.50 -> FTTS outsider (2 tests)
  * sweep seuil cote 1X2 fav <=T pour T in 1.3..2.0 home & away (pooled)
Sortie: exports/wf4_ftts.json
"""
import sys, json, math
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

NEW_LEAGUES = ["InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
               "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
               "InstantLeague-8060", "InstantLeague-8065"]

corrupted = set()
try:
    with open("exports/corrupted_events.json", encoding="utf-8") as f:
        d = json.load(f)
    corrupted = set(int(k) for k in d["events"].keys())
except Exception as ex:
    print("WARN corrupted_events:", ex)

e = create_engine(load_settings().db_url)
with e.connect() as c:
    in_clause = "(" + ",".join("'%s'" % l for l in NEW_LEAGUES) + ")"
    q = text("""
    SELECT e.id, e.competition, e.expected_start, o.odds_home, o.odds_draw, o.odds_away,
           o.extra_markets, r.score_a, r.score_b, r.goals_json
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o ON o.event_id = e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
      AND e.competition IN """ + in_clause)
    rows = c.execute(q).fetchall()

print("raw rows:", len(rows))


def settle_ftts(score_a, score_b, goals_json):
    """Returns '1', '2', 'Pas de but', or None (unsettleable)."""
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
        return None  # garde-fou corruption
    firsts = [x for x in g if (x.get("homeScore", 0) + x.get("awayScore", 0)) == 1]
    if len(firsts) != 1:
        return None
    t = firsts[0].get("team")
    if t == "Home":
        return "1"
    if t == "Away":
        return "2"
    return None


# build clean dataset, dedupe by event_id
data = {}
skip_corrupt, skip_noftts, skip_settle, dup = 0, 0, 0, 0
settle_fail_by_league = {}
total_by_league = {}
for r in rows:
    eid, comp, start, oh, od, oa, em_raw, sa, sb, gj = r
    if eid in data:
        dup += 1
        continue
    if eid in corrupted:
        skip_corrupt += 1
        continue
    total_by_league[comp] = total_by_league.get(comp, 0) + 1
    try:
        em = json.loads(em_raw) if em_raw else {}
    except Exception:
        em = {}
    ftts = em.get("FTTS")
    if not ftts or "1" not in ftts or "2" not in ftts:
        skip_noftts += 1
        continue
    outcome = settle_ftts(sa, sb, gj)
    if outcome is None:
        skip_settle += 1
        settle_fail_by_league[comp] = settle_fail_by_league.get(comp, 0) + 1
        continue
    data[eid] = dict(comp=comp, start=start, oh=float(oh), od=float(od), oa=float(oa),
                     f1=float(ftts["1"]), f2=float(ftts["2"]),
                     fnb=float(ftts.get("Pas de but", 0) or 0), outcome=outcome,
                     sa=int(sa), sb=int(sb))

print("clean events:", len(data), "| dup:", dup, "| corrupted:", skip_corrupt,
      "| no FTTS:", skip_noftts, "| unsettleable:", skip_settle)
print("unsettleable by league:", settle_fail_by_league)
print("total by league:", total_by_league)


def evaluate(bets):
    """bets = list of (odds, won_bool). Returns metrics dict."""
    n = len(bets)
    if n == 0:
        return dict(n=0)
    wins = sum(1 for o, w in bets if w)
    ret = sum((o - 1) if w else -1 for o, w in bets)
    roi = ret / n
    avg_odds = sum(o for o, w in bets) / n
    # null: true prob = 1/o per bet (zero-edge at offered odds)
    # var of return (o*X - 1), X~Bern(1/o): p(1-p)*o^2 = o - 1
    var = sum(o - 1 for o, w in bets)
    z = ret / math.sqrt(var) if var > 0 else 0.0
    p_one = 0.5 * math.erfc(z / math.sqrt(2))  # one-sided H1: ROI > 0
    wr = wins / n
    implied = sum(1.0 / o for o, w in bets) / n
    return dict(n=n, wins=wins, wr=round(wr, 4), roi_pct=round(100 * roi, 2),
                avg_odds=round(avg_odds, 3), z=round(z, 3), p_one_sided=round(p_one, 6),
                implied_wr=round(implied, 4), calib_ratio=round(wr / implied, 3) if implied else None)


results = {}
tests = 0
D = list(data.values())


def run(name, sel):
    global tests
    bets = [b for b in (sel(d) for d in D) if b is not None]
    results[name] = evaluate(bets)
    tests += 1
    return results[name]


def run_league(name, sel):
    global tests
    for lg in NEW_LEAGUES:
        bets = [b for b in (sel(d) for d in D if d["comp"] == lg) if b is not None]
        results[name + "::" + lg] = evaluate(bets)
        tests += 1


# --- E1 core: home fav <=1.50 -> FTTS '1'
e1 = lambda d: (d["f1"], d["outcome"] == "1") if d["oh"] <= 1.50 else None
run("E1_home_fav150_ftts1", e1)
run_league("E1_home_fav150_ftts1", e1)

# --- mirror: away fav <=1.50 -> FTTS '2'
e1a = lambda d: (d["f2"], d["outcome"] == "2") if d["oa"] <= 1.50 else None
run("E1_away_fav150_ftts2", e1a)
run_league("E1_away_fav150_ftts2", e1a)

# --- combined both sides
run("E1_anyfav150_ftts_fav", lambda d: (d["f1"], d["outcome"] == "1") if d["oh"] <= 1.50
    else ((d["f2"], d["outcome"] == "2") if d["oa"] <= 1.50 else None))

# --- bins on fav 1X2 odds
BINS = [(1.0, 1.3), (1.3, 1.5), (1.5, 1.7), (1.7, 2.0)]
for lo, hi in BINS:
    run("bin_home_%s-%s_ftts1" % (lo, hi),
        lambda d, lo=lo, hi=hi: (d["f1"], d["outcome"] == "1") if lo <= d["oh"] < hi else None)
    run("bin_away_%s-%s_ftts2" % (lo, hi),
        lambda d, lo=lo, hi=hi: (d["f2"], d["outcome"] == "2") if lo <= d["oa"] < hi else None)

# --- contrarian: fav <=1.50 -> bet outsider FTTS
run("contra_homefav150_ftts2", lambda d: (d["f2"], d["outcome"] == "2") if d["oh"] <= 1.50 else None)
run("contra_awayfav150_ftts1", lambda d: (d["f1"], d["outcome"] == "1") if d["oa"] <= 1.50 else None)

# --- threshold sweep fav <= T
for T in [1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]:
    run("sweep_home_le%s_ftts1" % T,
        lambda d, T=T: (d["f1"], d["outcome"] == "1") if d["oh"] <= T else None)
    run("sweep_away_le%s_ftts2" % T,
        lambda d, T=T: (d["f2"], d["outcome"] == "2") if d["oa"] <= T else None)

out = dict(n_clean_events=len(data), n_tests_scanned=tests,
           skip=dict(dup=dup, corrupted=skip_corrupt, no_ftts=skip_noftts, unsettleable=skip_settle,
                     unsettleable_by_league=settle_fail_by_league),
           events_by_league=total_by_league, results=results)
with open("exports/wf4_ftts.json", "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1, ensure_ascii=False)

print("")
print("n_tests_scanned:", tests)
print("")
print("%-40s %5s %6s %7s %7s %7s %9s %6s" % ("strategy", "n", "WR", "ROI%", "avgOdd", "z", "p(1s)", "calib"))
for k, v in results.items():
    if v.get("n", 0) == 0:
        print("%-40s %5d" % (k, 0))
        continue
    print("%-40s %5d %6.3f %7.2f %7.3f %7.2f %9.5f %6s" % (
        k, v["n"], v["wr"], v["roi_pct"], v["avg_odds"], v["z"], v["p_one_sided"], v["calib_ratio"]))
