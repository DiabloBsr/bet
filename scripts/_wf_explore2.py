import sys, json
sys.path.insert(0, '.')
from collections import Counter
from sqlalchemy import create_engine, text
from scraper.config import load_settings

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, os.extra_markets, r.goals_json, r.score_a, r.score_b
FROM events e
JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f ON f.event_id=e.id
JOIN odds_snapshots os ON os.id=f.mid
JOIN results r ON r.event_id=e.id
WHERE e.round_info != '0'
"""
with eng.connect() as c:
    rows = c.execute(text(q)).fetchall()
print("total rows:", len(rows))

ou_keys = Counter(); th_keys = Counter(); ta_keys = Counter(); mb_keys = Counter()
have = Counter()
for rid, em, gj, sa, sb in rows:
    if em is None: continue
    if isinstance(em, str):
        try: em = json.loads(em)
        except: continue
    if not isinstance(em, dict): continue
    for mk in em: have[mk]+=1
    if '+/-' in em: ou_keys.update(em['+/-'].keys())
    if 'Total equipe domicile' in em: th_keys.update(em['Total equipe domicile'].keys())
    if 'Total equipe extérieur' in em: ta_keys.update(em['Total equipe extérieur'].keys())
    if 'Multi-Buts' in em: mb_keys.update(em['Multi-Buts'].keys())
print("\nmarket presence:")
for k,v in have.most_common(): print(f"  {k}: {v}")
print("\n+/- keys:", dict(ou_keys))
print("\nTotal dom keys:", dict(th_keys))
print("\nTotal ext keys:", dict(ta_keys))
print("\nMulti-Buts keys:", dict(mb_keys))

# goals_json format
n_gj = 0
for rid, em, gj, sa, sb in rows:
    if gj:
        n_gj += 1
        if n_gj <= 3:
            print("\ngoals_json sample event", rid, "score", sa, sb, ":", gj[:500])
print("\nrows with goals_json:", n_gj)
