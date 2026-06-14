import sys, json
sys.path.insert(0,'.')
from sqlalchemy import create_engine, text
from scraper.config import load_settings
eng = create_engine(load_settings().db_url)
q = """
SELECT r.score_a, r.score_b, r.goals_json
FROM events e JOIN results r ON r.event_id=e.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
with eng.connect() as c:
    rows = c.execute(text(q)).fetchall()
n00 = sum(1 for a,b,g in rows if a+b==0)
n00_gj = [g for a,b,g in rows if a+b==0][:5]
from collections import Counter
tots = Counter(a+b for a,b,_ in rows)
print("0-0 count:", n00, "sample gj:", n00_gj)
print("total dist:", sorted(tots.items()))
print("tot>=7:", sum(v for k,v in tots.items() if k>=7), "/", len(rows))
print("tot==6:", tots[6])
