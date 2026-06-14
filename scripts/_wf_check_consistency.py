import sys, json
sys.path.insert(0,'.')
from sqlalchemy import create_engine, text
from scraper.config import load_settings
eng = create_engine(load_settings().db_url)
q = """
SELECT e.id, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
FROM events e JOIN results r ON r.event_id=e.id
WHERE e.round_info != '0' AND r.score_a IS NOT NULL
"""
with eng.connect() as c:
    rows = c.execute(text(q)).fetchall()

stats = dict(total=0, gj_null=0, gj_empty=0, gj_match=0, gj_mismatch=0)
mm00 = 0; mm_other = 0
ht_consistent_with_gj = 0; ht_consistent_with_score = 0; mm_ht_checked = 0
for eid, sa, sb, hta, htb, gj in rows:
    stats['total'] += 1
    if not gj:
        stats['gj_null'] += 1; continue
    try:
        g = json.loads(gj)
    except Exception:
        stats['gj_null'] += 1; continue
    if not g:
        stats['gj_empty'] += 1
        if sa+sb != 0: stats['gj_mismatch'] += 1
        else: stats['gj_match'] += 1
        continue
    last = g[-1]
    if int(last['homeScore']) == sa and int(last['awayScore']) == sb and len(g) == sa+sb:
        stats['gj_match'] += 1
    else:
        stats['gj_mismatch'] += 1
        if sa+sb == 0: mm00 += 1
        else: mm_other += 1
        # arbitrate with HT score
        if hta is not None:
            mm_ht_checked += 1
            gh = sum(1 for x in g if x['minute'] <= 45 and x['team']=='Home')
            ga = sum(1 for x in g if x['minute'] <= 45 and x['team']=='Away')
            if (gh, ga) == (hta, htb): ht_consistent_with_gj += 1
            if (hta, htb) == (0,0) and sa+sb==0: ht_consistent_with_score += 1
print(stats)
print("mismatch where score=0-0:", mm00, " other mismatch:", mm_other)
print("among mismatches: HT agrees with goals_json:", ht_consistent_with_gj, "/", mm_ht_checked)

# For 0-0 score rows specifically
z = [(eid,hta,htb,gj) for eid,sa,sb,hta,htb,gj in rows if sa+sb==0]
z_gj_goals = [x for x in z if x[3] and json.loads(x[3])]
z_ht00 = sum(1 for e,hta,htb,g in z if (hta,htb)==(0,0))
print(f"\n0-0 rows: {len(z)} | with non-empty goals_json: {len(z_gj_goals)} | HT=0-0: {z_ht00}")
# of the 0-0 rows with goals in gj, what does HT say?
bad=0
for e,hta,htb,gj in z_gj_goals:
    g=json.loads(gj)
    gh = sum(1 for x in g if x['minute']<=45 and x['team']=='Home')
    ga = sum(1 for x in g if x['minute']<=45 and x['team']=='Away')
    if (hta,htb)!=(0,0) or (gh,ga)!=(0,0):
        bad+=1
print("0-0-score rows where HT or gj contradicts 0-0:", bad)
