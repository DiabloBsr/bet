# -*- coding: utf-8 -*-
import sys, pickle
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from collections import Counter

e = create_engine(load_settings().db_url)
with e.connect() as c:
    n_ev = c.execute(text("SELECT COUNT(DISTINCT event_id) FROM results")).scalar()
    n_rows = c.execute(text("SELECT COUNT(*) FROM results")).scalar()
    print("results rows:", n_rows, "distinct event_id:", n_ev)
    # per-league distinct finished with odds now
    rows = c.execute(text("""
        SELECT ev.competition, COUNT(DISTINCT ev.id) FROM events ev
        JOIN results r ON r.event_id=ev.id
        WHERE EXISTS (SELECT 1 FROM odds_snapshots o WHERE o.event_id=ev.id)
        GROUP BY ev.competition
    """)).fetchall()
    for r in rows: print(r)

recs = pickle.load(open("scripts/_wf4_roundstruct_data.pkl", "rb"))
ids = Counter(r["id"] for r in recs)
dup = {k: v for k, v in ids.items() if v > 1}
print("kept recs:", len(recs), "distinct ids:", len(ids), "dup ids:", len(dup))
