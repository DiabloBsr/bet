# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
with e.connect() as c:
    # which round_info do the 20-groups have?
    rows = c.execute(text("""
        SELECT round_info, COUNT(*) FROM (
            SELECT round_info, expected_start, COUNT(*) as n
            FROM events WHERE competition='InstantLeague-8035'
            GROUP BY round_info, expected_start HAVING n=20
        ) GROUP BY round_info ORDER BY CAST(round_info AS INT)
    """)).fetchall()
    print("20-groups by round_info:", rows[:10], "...", len(rows))
    # one example 20-group: show teams
    ex = c.execute(text("""
        SELECT expected_start FROM events WHERE competition='InstantLeague-8035'
        GROUP BY round_info, expected_start HAVING COUNT(*)=20 LIMIT 1
    """)).fetchone()
    rows = c.execute(text("""
        SELECT id, round_info, team_a, team_b FROM events
        WHERE competition='InstantLeague-8035' AND expected_start=:st ORDER BY team_a
    """), {"st": ex[0]}).fetchall()
    for r in rows: print(r)
