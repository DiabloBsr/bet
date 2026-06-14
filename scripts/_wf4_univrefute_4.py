# -*- coding: utf-8 -*-
"""Refutation part 2: data integrity of the new-era sample.
1. goals_json raw structure for 0-0 events (old vs new) — what is 'nonempty'?
2. HT coherence: goals_json-derived HT vs recorded HT, per league (new era).
3. Survivorship: events without results row, per league/era.
READ-ONLY.
"""
import sys, json
sys.path.insert(0, ".")
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from collections import Counter, defaultdict

REF = "InstantLeague-8035"
LEAGUES = [REF, "InstantLeague-8036", "InstantLeague-8037", "InstantLeague-8042",
           "InstantLeague-8043", "InstantLeague-8044", "InstantLeague-8056",
           "InstantLeague-8060", "InstantLeague-8065"]
NEWWIN = "2026-06-12 00:00:00"
eng = create_engine(load_settings().db_url)

print("1. RAW goals_json samples for 0-0 events")
with eng.connect() as c:
    rows = c.execute(text(
        "SELECT e.competition, e.expected_start, r.goals_json FROM results r "
        "JOIN events e ON e.id=r.event_id WHERE r.score_a=0 AND r.score_b=0 "
        "AND e.competition IN ('InstantLeague-8035','InstantLeague-8056') "
        "ORDER BY e.id DESC LIMIT 6")).fetchall()
    for comp, ts, gj in rows:
        print(f"  {comp} {ts}: {repr(gj)[:120]}")
    rows = c.execute(text(
        "SELECT e.competition, e.expected_start, r.goals_json FROM results r "
        "JOIN events e ON e.id=r.event_id WHERE r.score_a=0 AND r.score_b=0 "
        "AND e.competition='InstantLeague-8035' AND e.expected_start < '" + NEWWIN + "' "
        "ORDER BY e.id DESC LIMIT 3")).fetchall()
    for comp, ts, gj in rows:
        print(f"  OLD {comp} {ts}: {repr(gj)[:120]}")

print("\n2. HT COHERENCE (goals_json minute<=45 vs recorded HT), per league")
with eng.connect() as c:
    rows = c.execute(text(
        "SELECT e.competition, e.expected_start, r.score_a, r.score_b, "
        "r.ht_score_a, r.ht_score_b, r.goals_json FROM results r "
        "JOIN events e ON e.id=r.event_id WHERE e.competition IN (" +
        ",".join("'" + l + "'" for l in LEAGUES) + ")")).fetchall()
coh = defaultdict(Counter)
for comp, ts, sa, sb, ha, hb, gj in rows:
    era = ("old" if str(ts) < NEWWIN else "new") if comp == REF else "new"
    key = f"{comp}|{era}"
    if not gj:
        coh[key]["gj_null"] += 1
        continue
    try:
        gl = json.loads(gj)
    except Exception:
        coh[key]["unparseable"] += 1
        continue
    if not isinstance(gl, list):
        coh[key]["not_list"] += 1
        continue
    if len(gl) != (sa or 0) + (sb or 0):
        coh[key]["len_mismatch"] += 1
        continue
    if ha is None:
        coh[key]["ht_null"] += 1
        continue
    gh = sum(1 for g in gl if g.get("minute", 99) <= 45 and g.get("team") == "Home")
    ga = sum(1 for g in gl if g.get("minute", 99) <= 45 and g.get("team") == "Away")
    coh[key]["ht_ok" if (gh, ga) == (ha, hb) else "ht_mismatch"] += 1
for k in sorted(coh):
    tot = sum(coh[k].values())
    print(f"  {k}: n={tot} {dict(coh[k])}")

print("\n3. SURVIVORSHIP: finished-window events without results, per league/era")
with eng.connect() as c:
    maxts = c.execute(text(
        "SELECT MAX(e.expected_start) FROM events e JOIN results r ON r.event_id=e.id"
    )).scalar()
    print("  max finished expected_start:", maxts)
    rows = c.execute(text(
        "SELECT e.competition, e.expected_start, "
        "CASE WHEN r.id IS NULL THEN 0 ELSE 1 END AS has_res, "
        "CASE WHEN o.event_id IS NULL THEN 0 ELSE 1 END AS has_odds "
        "FROM events e LEFT JOIN results r ON r.event_id=e.id "
        "LEFT JOIN (SELECT DISTINCT event_id FROM odds_snapshots) o ON o.event_id=e.id "
        "WHERE e.competition IN (" + ",".join("'" + l + "'" for l in LEAGUES) + ") "
        "AND e.expected_start <= '" + str(maxts)[:19] + "'")).fetchall()
surv = defaultdict(Counter)
for comp, ts, has_res, has_odds in rows:
    era = ("old" if str(ts) < NEWWIN else "new") if comp == REF else "new"
    key = f"{comp}|{era}"
    surv[key]["total"] += 1
    if has_odds and not has_res:
        surv[key]["odds_no_result"] += 1
    if not has_res:
        surv[key]["no_result"] += 1
for k in sorted(surv):
    d = surv[k]
    print(f"  {k}: total={d['total']} no_result={d['no_result']} "
          f"({100*d['no_result']/d['total']:.1f}%) odds_no_result={d['odds_no_result']} "
          f"({100*d['odds_no_result']/d['total']:.1f}%)")
