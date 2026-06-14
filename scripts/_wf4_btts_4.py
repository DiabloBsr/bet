# WF4 - BTTS - etape 4: marches freres de la famille BTTS (meme snapshot d'ouverture)
#   "G/NG equipe domicile", "G/NG equipe extérieur",
#   "Les deux équipes marquent / 1ère mi temps", "1X2 & G/NG" (6 selections)
# Settlement: FT (sa,sb) ; HT (hta,htb) pour BTTS 1ere MT.
# Sortie: exports/wf4_btts_family_data.json
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
    corrupted = set(int(k) for k in json.load(f)["events"].keys())

MK_HOME = "G/NG equipe domicile"
MK_AWAY = "G/NG equipe extérieur"
MK_HT = "Les deux équipes marquent / 1ère mi temps"
MK_COMBO = "1X2 & G/NG"

e = create_engine(load_settings().db_url)
out = []
stats = {"raw": 0, "skip": 0, "kept": 0}
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
    if eid in corrupted or sa is None or sb is None or not em \
       or oh is None or od is None or oa is None:
        stats["skip"] += 1
        continue
    if hta is not None and htb is not None and (hta > sa or htb > sb):
        stats["skip"] += 1
        continue
    if gj:
        try:
            gl = json.loads(gj)
            if isinstance(gl, list) and len(gl) > 0 and len(gl) != int(sa) + int(sb):
                stats["skip"] += 1
                continue
        except Exception:
            pass
    try:
        m = json.loads(em)
    except Exception:
        stats["skip"] += 1
        continue
    rec = {"id": eid, "comp": comp, "start": str(est),
           "oh": oh, "od": od, "oa": oa,
           "sa": int(sa), "sb": int(sb),
           "hta": (int(hta) if hta is not None else None),
           "htb": (int(htb) if htb is not None else None),
           "home": m.get(MK_HOME), "away": m.get(MK_AWAY),
           "ht": m.get(MK_HT), "combo": m.get(MK_COMBO)}
    out.append(rec)
    stats["kept"] += 1

with open("exports/wf4_btts_family_data.json", "w", encoding="utf-8") as f:
    json.dump({"stats": stats, "rows": out}, f)
print(stats)
have = {k: sum(1 for r in out if r[k]) for k in ("home", "away", "ht", "combo")}
print("dispo marches:", have)
# echantillon de cles combo
for r in out:
    if r["combo"]:
        print("cles combo:", sorted(r["combo"].keys()))
        break
