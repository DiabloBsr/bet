# -*- coding: utf-8 -*-
"""WF4 round-structure exploration: how do events group into round instances?"""
import sys, json
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
with e.connect() as c:
    # Sample: for 8035, group by (competition, round_info, expected_start) - how many matches share exact start?
    rows = c.execute(text("""
        SELECT competition, round_info, expected_start, COUNT(*) as n
        FROM events
        WHERE competition LIKE 'InstantLeague-%'
        GROUP BY competition, round_info, expected_start
        ORDER BY competition, expected_start
    """)).fetchall()

from collections import Counter, defaultdict
per_comp = defaultdict(Counter)
for comp, rnd, st, n in rows:
    per_comp[comp][n] += 1

for comp in sorted(per_comp):
    print(comp, dict(sorted(per_comp[comp].items())))

# Check a specific example: 8035 round 5, list a few expected_starts
with e.connect() as c:
    rows = c.execute(text("""
        SELECT round_info, expected_start, COUNT(*)
        FROM events WHERE competition='InstantLeague-8035'
        GROUP BY round_info, expected_start
        ORDER BY expected_start LIMIT 30
    """)).fetchall()
for r in rows:
    print(r)
