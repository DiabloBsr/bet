# WF4 - BTTS pooled 9 ligues - etape 1: extraction des donnees
# Pour chaque match fini avec cote d'ouverture (9 ligues):
#   cotes 1X2 d'ouverture + dict "G/NG" du MEME snapshot d'ouverture (MIN(o.id))
#   + resultat FT + garde-fou corruption (HT>FT, goals_json incoherent)
# Sortie: exports/wf4_btts_data.json
import sys, json
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

LEAGUES = [
    "InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
    "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044",
    "InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065",
]

corrupted = set()
with open("exports/corrupted_events.json", encoding="utf-8") as f:
    d = json.load(f)
    corrupted = set(int(k) for k in d["events"].keys())

e = create_engine(load_settings().db_url)
out = []
stats = {"raw": 0, "corrupted_skip": 0, "guard_htft": 0, "guard_goals": 0,
         "no_gng": 0, "kept": 0}
in_list = ",".join("'%s'" % l for l in LEAGUES)
with e.connect() as c:
    rows = c.execute(text("""
        SELECT e.id, e.competition, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
        FROM events e
        JOIN results r ON r.event_id = e.id
        JOIN odds_snapshots o ON o.event_id = e.id
        WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
          AND e.competition IN (%s)
    """ % in_list)).fetchall()

for (eid, comp, est, oh, od, oa, em, sa, sb, hta, htb, gj) in rows:
    stats["raw"] += 1
    if eid in corrupted:
        stats["corrupted_skip"] += 1
        continue
    if sa is None or sb is None or oh is None or od is None or oa is None:
        continue
    # garde-fou corruption (les nouvelles ligues n'ont pas ete auditees)
    if hta is not None and htb is not None and (hta > sa or htb > sb):
        stats["guard_htft"] += 1
        continue
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and len(gl) > 0 and len(gl) != int(sa) + int(sb):
                stats["guard_goals"] += 1
                continue
        except Exception:
            pass
    gng = None
    if em:
        try:
            gng = json.loads(em).get("G/NG")
        except Exception:
            gng = None
    if not gng or "Oui" not in gng or "Non" not in gng:
        stats["no_gng"] += 1
        continue
    stats["kept"] += 1
    out.append({
        "id": eid, "comp": comp, "start": str(est),
        "oh": oh, "od": od, "oa": oa,
        "o_yes": float(gng["Oui"]), "o_no": float(gng["Non"]),
        "sa": int(sa), "sb": int(sb),
    })

with open("exports/wf4_btts_data.json", "w", encoding="utf-8") as f:
    json.dump({"stats": stats, "rows": out}, f)
print(stats)
from collections import Counter
print(Counter(r["comp"] for r in out))
