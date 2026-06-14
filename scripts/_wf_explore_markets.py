import sys, json
sys.path.insert(0, '.')
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, e.round_info, os.extra_markets
FROM events e
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f ON f.event_id=e.id
JOIN odds_snapshots os ON os.id=f.mid
JOIN results r ON r.event_id=e.id
WHERE e.round_info != '0' AND os.extra_markets IS NOT NULL
LIMIT 5
"""
with eng.connect() as c:
    rows = c.execute(text(q)).fetchall()
for rid, ri, em in rows[:2]:
    if isinstance(em, str):
        em = json.loads(em)
    print("EVENT", rid, "round", ri)
    print("type:", type(em))
    if isinstance(em, dict):
        for k, v in em.items():
            print(" MARKET:", repr(k))
            print("   ", json.dumps(v, ensure_ascii=False)[:400])
    print("="*80)
