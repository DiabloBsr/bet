# -*- coding: utf-8 -*-
"""WF5 counter-verify — la ligne 2.5 du marche Plus/Moins est-elle cotee, et quand ?

1. Sur un echantillon d'events OOS InstantLeague-8035 (paires OVER_GOLD + aleatoire),
   dump des cles du marche totals dans le snapshot d'OUVERTURE (MIN(id)).
2. Recherche de la sous-chaine "2.5"/"2,5" n'importe ou dans extra_markets,
   sur TOUS les snapshots de ces events (pas seulement l'ouverture).
Sortie: exports/wf5_over25_market_check.json
"""
import sys, json
from collections import Counter, defaultdict

sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text
from scraper.team_gold_data import OVER_GOLD

LEAGUE = "InstantLeague-8035"
engine = create_engine(load_settings().db_url)
corrupted = set(int(k) for k in json.load(open("exports/corrupted_events.json"))["events"].keys())

# events OOS des paires OVER_GOLD + 300 events recents quelconques de la ligue
with engine.connect() as c:
    rows = c.execute(text(
        "SELECT e.id, e.team_a, e.team_b FROM events e "
        "JOIN results r ON r.event_id=e.id "
        "WHERE e.competition=:lg AND r.finished_at>='2026-06-06' "
        "ORDER BY r.finished_at"), {"lg": LEAGUE}).fetchall()
gold_pairs = set(OVER_GOLD)
ev_gold = [r[0] for r in rows if (r[1], r[2]) in gold_pairs and r[0] not in corrupted]
ev_any = [r[0] for r in rows if r[0] not in corrupted][-300:]
ids = sorted(set(ev_gold) | set(ev_any))
print(f"events analyses: {len(ids)} (dont {len(ev_gold)} paires OVER_GOLD)")

market_names = Counter()          # noms de marches top-level a l'ouverture
totals_keys_open = Counter()      # cles du marche +/- a l'ouverture
has25_open = 0                    # "2.5"/"2,5" present qq part a l'ouverture
has25_any_snap = 0                # ... dans n'importe quel snapshot
n_open = 0
snap_counts = []
sample_open = None

CH = 200
for i in range(0, len(ids), CH):
    chunk = ids[i:i + CH]
    inlist = ",".join(str(x) for x in chunk)
    with engine.connect() as c:
        # ouverture
        rs = c.execute(text(
            "SELECT f.event_id, os.extra_markets FROM (SELECT event_id, MIN(id) sid "
            f"FROM odds_snapshots WHERE event_id IN ({inlist}) GROUP BY event_id) f "
            "JOIN odds_snapshots os ON os.id=f.sid")).fetchall()
        for ev, raw in rs:
            n_open += 1
            if not raw:
                continue
            try:
                em = json.loads(raw)
            except Exception:
                continue
            market_names.update(em.keys())
            tot = em.get("+/-") or {}
            totals_keys_open.update(tot.keys())
            if "2.5" in raw or "2,5" in raw:
                has25_open += 1
            if sample_open is None and tot:
                sample_open = {"event": ev, "+/-": tot}
        # tous les snapshots: la ligne 2.5 apparait-elle un jour ?
        rs2 = c.execute(text(
            "SELECT event_id, COUNT(*), "
            "SUM(CASE WHEN extra_markets LIKE '%2.5%' OR extra_markets LIKE '%2,5%' "
            "THEN 1 ELSE 0 END) "
            f"FROM odds_snapshots WHERE event_id IN ({inlist}) GROUP BY event_id")).fetchall()
        for ev, nsn, n25 in rs2:
            snap_counts.append(nsn)
            if n25 and n25 > 0:
                has25_any_snap += 1

out = dict(
    n_events=len(ids), n_open_snapshots=n_open,
    market_names_open=dict(market_names.most_common(20)),
    totals_keys_open=dict(totals_keys_open.most_common(20)),
    events_with_25_anywhere_in_opening=has25_open,
    events_with_25_in_any_snapshot=has25_any_snap,
    snapshots_per_event_minmax=[min(snap_counts), max(snap_counts)] if snap_counts else None,
    sample_opening_totals=sample_open,
)
with open("exports/wf5_over25_market_check.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
