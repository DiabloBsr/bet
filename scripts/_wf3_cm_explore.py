import sys, json, collections
sys.path.insert(0,'.')
from scraper.config import load_settings
from sqlalchemy import create_engine, text
import pandas as pd

eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
       os.id AS snap_id, os.odds_home, os.odds_draw, os.odds_away, os.extra_markets,
       r.score_a, r.score_b
FROM events e
JOIN (SELECT event_id, MIN(id) AS mid FROM odds_snapshots GROUP BY event_id) m ON m.event_id = e.id
JOIN odds_snapshots os ON os.id = m.mid
JOIN results r ON r.event_id = e.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
df = pd.read_sql(q, eng)
print("rows joined:", len(df))
df = df.sort_values('snap_id').drop_duplicates(['team_a','team_b','expected_start'], keep='first')
print("after dedup:", len(df))

keys_per_market = collections.defaultdict(collections.Counter)
cap_count = collections.Counter()
overround = []
n_em = 0
for em_raw in df['extra_markets']:
    if em_raw is None: continue
    em = json.loads(em_raw) if isinstance(em_raw, str) else em_raw
    n_em += 1
    for mk, sels in em.items():
        for k, o in sels.items():
            keys_per_market[mk][k] += 1
            if o == 100.0: cap_count[mk] += 1
    cs = em.get('Score exact')
    if cs:
        s = sum(1.0/o for o in cs.values())
        overround.append(s)

print("with extra_markets:", n_em)
print("\n--- markets and keys (count of distinct keys, sample) ---")
for mk, cnt in keys_per_market.items():
    ks = list(cnt.keys())
    print(f"{mk!r}: {len(ks)} keys, caps@100={cap_count[mk]}")
    if len(ks) <= 25:
        print("   ", ks)
    else:
        print("   sample:", ks[:30])

ov = pd.Series(overround)
print("\nCS grid overround sum(1/o): ", ov.describe())

# totals distribution empirical
tot = (df['score_a'] + df['score_b'])
print("\nempirical total goals distribution:")
print(tot.value_counts().sort_index())
