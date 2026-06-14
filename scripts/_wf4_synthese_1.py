# -*- coding: utf-8 -*-
"""WF4 SYNTHESE - estimation des volumes de paris/jour pour le rapport final.
Lecture seule (SELECT uniquement). Sortie: exports/wf4_synthese_volumes.json
"""
import sys, json, datetime
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

CHAMPS = ["InstantLeague-8035", "InstantLeague-8036", "InstantLeague-8037",
          "InstantLeague-8042", "InstantLeague-8043", "InstantLeague-8044"]
CUPS = ["InstantLeague-8056", "InstantLeague-8060", "InstantLeague-8065"]
ALL9 = CHAMPS + CUPS

corr = set()
try:
    d = json.load(open("exports/corrupted_events.json", encoding="utf-8"))
    corr = set(int(k) for k in d.get("events", {}).keys())
except Exception as e:
    print("warn corrupted:", e)

e = create_engine(load_settings().db_url)
out = {}

placeholders = ",".join("'" + x + "'" for x in ALL9)
with e.connect() as c:
    # cote d'ouverture par event (MIN id) + ligue + date du expected_start
    rows = c.execute(text(f"""
        SELECT ev.id, ev.competition, ev.expected_start,
               o.odds_home, o.odds_draw, o.odds_away
        FROM events ev
        JOIN odds_snapshots o ON o.id = (
            SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = ev.id)
        WHERE ev.competition IN ({placeholders})
    """)).fetchall()

per_day_league = {}
ftts_rule_per_day = {}   # champ leagues, home open <= 1.50
for rid, comp, est, oh, od, oa in rows:
    if rid in corr or not est:
        continue
    day = str(est)[:10]
    per_day_league.setdefault(day, {}).setdefault(comp, 0)
    per_day_league[day][comp] += 1
    if comp in CHAMPS and oh and oh <= 1.50:
        ftts_rule_per_day.setdefault(day, 0)
        ftts_rule_per_day[day] += 1

# on garde uniquement les jours "pleins" (toutes ligues actives) = depuis 2026-06-12
full_days = [d for d, leagues in sorted(per_day_league.items())
             if len(leagues) == 9]
out["days_all9_active"] = full_days
out["per_day_league"] = {d: per_day_league[d] for d in sorted(per_day_league)[-5:]}
out["events_per_day_all9"] = {d: sum(per_day_league[d].values()) for d in full_days}
out["ftts_rule_bets_per_day"] = {d: ftts_rule_per_day.get(d, 0) for d in full_days}

# volumes E2 (favori [1.10,1.20)) par jour, toutes ligues
e2_per_day = {}
for rid, comp, est, oh, od, oa in rows:
    if rid in corr or not est or not oh or not oa:
        continue
    fav = min(oh, oa)
    if 1.10 <= fav < 1.20:
        day = str(est)[:10]
        e2_per_day.setdefault(day, 0)
        e2_per_day[day] += 1
out["e2_fav110_120_per_day"] = {d: e2_per_day.get(d, 0) for d in full_days}

# part des events champ avec home <= 1.50 (taux de declenchement)
tot_champ = sum(1 for rid, comp, est, oh, od, oa in rows
                if comp in CHAMPS and rid not in corr and oh)
hit_champ = sum(1 for rid, comp, est, oh, od, oa in rows
                if comp in CHAMPS and rid not in corr and oh and oh <= 1.50)
out["champ_events_with_odds"] = tot_champ
out["champ_home_le_150"] = hit_champ
out["trigger_rate_pct"] = round(100.0 * hit_champ / max(tot_champ, 1), 2)

json.dump(out, open("exports/wf4_synthese_volumes.json", "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False)[:3000])
