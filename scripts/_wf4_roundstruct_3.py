# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
with e.connect() as c:
    rows = c.execute(text("""
        SELECT e.expected_start, MIN(r.finished_at), MAX(r.finished_at), COUNT(*)
        FROM events e JOIN results r ON r.event_id=e.id
        WHERE e.competition='InstantLeague-8036'
        GROUP BY e.expected_start ORDER BY e.expected_start LIMIT 8
    """)).fetchall()
    for r in rows: print(r)
