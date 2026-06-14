import sys, json
sys.path.insert(0, ".")
from scraper.config import load_settings
from sqlalchemy import create_engine, text

e = create_engine(load_settings().db_url)
with e.connect() as c:
    rows = c.execute(text("SELECT competition, COUNT(*) FROM events GROUP BY competition")).fetchall()
    print("COMPETITIONS:")
    for r in rows:
        print("  ", r[0], r[1])

    # FTTS presence in opening snapshots for new leagues
    q = text("""
    SELECT e.competition, COUNT(*) as n
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o ON o.event_id = e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
      AND e.competition != 'InstantLeague-8035'
    GROUP BY e.competition
    """)
    print("\nFINISHED+OPENING ODDS per new league:")
    for r in c.execute(q).fetchall():
        print("  ", r[0], r[1])

    # sample opening snapshot extra_markets FTTS for one new league
    q2 = text("""
    SELECT e.id, e.competition, o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, r.score_a, r.score_b, r.goals_json
    FROM events e
    JOIN results r ON r.event_id = e.id
    JOIN odds_snapshots o ON o.event_id = e.id
    WHERE o.id = (SELECT MIN(o2.id) FROM odds_snapshots o2 WHERE o2.event_id = e.id)
      AND e.competition = 'InstantLeague-8036'
    LIMIT 5
    """)
    print("\nSAMPLE 8036:")
    for r in c.execute(q2).fetchall():
        em = json.loads(r[5]) if r[5] else {}
        gj = r[8]
        print("  event", r[0], "1X2:", r[2], r[3], r[4], "FTTS:", em.get("FTTS"), "score", r[6], r[7])
        if gj:
            g = json.loads(gj)
            print("    goals_json first:", g[0] if g else None, "len", len(g) if g else 0)
        else:
            print("    goals_json: NULL")
