# -*- coding: utf-8 -*-
"""WF4 E1/FTTS sanity check on new leagues. READ-ONLY."""
import sys, json
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
NEW = ["InstantLeague-8036","InstantLeague-8037","InstantLeague-8042","InstantLeague-8043",
       "InstantLeague-8044","InstantLeague-8056","InstantLeague-8060","InstantLeague-8065"]

with e.connect() as c:
    for lg in NEW:
        rows = c.execute(text("""
            SELECT e.id, r.score_a, r.score_b, r.goals_json, o.extra_markets,
                   o.odds_home, o.odds_draw, o.odds_away
            FROM events e
            JOIN results r ON r.event_id = e.id
            JOIN odds_snapshots o ON o.event_id = e.id
            WHERE e.competition = :lg
              AND o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
            LIMIT 8
        """), {"lg": lg}).fetchall()
        n_ftts = 0; n_gj = 0
        for (eid, sa, sb, gj, em, oh, od, oa) in rows:
            try:
                m = json.loads(em) if em else {}
            except Exception:
                m = {}
            if "FTTS" in m: n_ftts += 1
            g = None
            try:
                g = json.loads(gj) if gj else None
            except Exception:
                pass
            if g is not None: n_gj += 1
        print(lg, "sample", len(rows), "FTTS in open snap:", n_ftts, "goals_json parseable:", n_gj)
        # show one FTTS dict + one goals_json
        for (eid, sa, sb, gj, em, oh, od, oa) in rows[:1]:
            m = json.loads(em) if em else {}
            print("  ex FTTS:", m.get("FTTS"), "| 1x2:", oh, od, oa, "| score", sa, sb)
            print("  ex goals_json:", (gj or "")[:200])
