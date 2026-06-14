# -*- coding: utf-8 -*-
# ADVERSARIAL CHECK 1: verify _wf4_roundstruct_data.pkl against the live DB (READ-ONLY)
#  - sample 400 events: opening odds (MIN snapshot id), scores, competition, expected_start
#  - confirm corrupted ids excluded, odds are the true first snapshot
import sys, json, pickle, random
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))
random.seed(99)
sample = random.sample(recs, 400)
ids = [r["id"] for r in sample]
by_id = {r["id"]: r for r in sample}

SQL = """
SELECT ev.id, ev.competition, ev.expected_start, r.score_a, r.score_b,
       o.odds_home, o.odds_draw, o.odds_away, o.id AS snap_id, o.captured_at
FROM events ev
JOIN results r ON r.event_id = ev.id
JOIN odds_snapshots o ON o.event_id = ev.id
WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
  AND ev.id IN ({ids})
"""
mism = 0
fetched_after_start = 0
with e.connect() as conn:
    rows = conn.execute(text(SQL.format(ids=",".join(map(str, ids))))).fetchall()
    print("db rows for sample:", len(rows))
    for (eid, comp, est, sa, sb, oh, od, oa, snap_id, fat) in rows:
        r = by_id[eid]
        ok = (r["comp"] == comp and r["est"] == est and r["sa"] == sa and r["sb"] == sb
              and abs(r["oh"] - oh) < 1e-9 and abs(r["od"] - od) < 1e-9 and abs(r["oa"] - oa) < 1e-9)
        if not ok:
            mism += 1
            print("MISMATCH", eid, r, (comp, est, sa, sb, oh, od, oa))
        # look-ahead check: opening snapshot fetched BEFORE expected_start?
        if fat is not None and est is not None and str(fat) > str(est):
            fetched_after_start += 1
print("mismatches:", mism, "/ 400")
print("opening snapshot fetched AFTER expected_start:", fetched_after_start, "/ 400")
