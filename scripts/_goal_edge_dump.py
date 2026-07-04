"""Dump par-event des marchés buts + cotes offertes + équipes, pour chasse d'edge
par agents : data/goal_edge_long.csv. Lecture seule.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

_SQL = """
SELECT e.competition lg, e.expected_start ts, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
       r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition LIKE 'InstantLeague-%'
  AND o.extra_markets IS NOT NULL AND o.extra_markets NOT IN ('','{}','null')
ORDER BY e.expected_start
"""


def gm(xm, name):
    if name in xm:
        return xm[name]
    for k, v in xm.items():
        if k.startswith(name):
            return v
    return None


def main():
    df = pd.read_sql(text(_SQL), create_engine(load_settings().db_url))
    rows = []
    for r in df.itertuples():
        try:
            xm = json.loads(r.xm) if isinstance(r.xm, str) else r.xm
        except Exception:
            continue
        if not isinstance(xm, dict):
            continue
        sa, sb = int(r.sa), int(r.sb); tot = sa + sb
        rec = {"lg": r.lg, "ts": r.ts, "home": r.home, "away": r.away,
               "oh": r.oh, "od": r.od, "oa": r.oa,
               "sa": sa, "sb": sb, "tot": tot,
               "over35": int(tot > 3.5), "under35": int(tot < 3.5),
               "btts": int(sa > 0 and sb > 0)}
        pm = gm(xm, "+/-")
        if isinstance(pm, dict):
            rec["o_over35"] = pm.get("> 3.5"); rec["o_under35"] = pm.get("< 3.5")
        gn = gm(xm, "G/NG")
        if isinstance(gn, dict):
            rec["o_btts_oui"] = gn.get("Oui"); rec["o_btts_non"] = gn.get("Non")
        tb = gm(xm, "Total de buts")
        if isinstance(tb, dict):
            for k, v in tb.items():
                if k.isdigit():
                    rec[f"o_tot{k}"] = v
        se = gm(xm, "Score exact")
        if isinstance(se, dict):
            key = f"{min(sa,6)}-{min(sb,6)}"
            for kk, vv in se.items():
                if kk.replace(" ", "").replace(":", "-") == key:
                    rec["o_exact_real"] = vv
                    break
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv("data/goal_edge_long.csv", index=False)
    print(f"écrit data/goal_edge_long.csv : {len(out)} events, {out.lg.nunique()} ligues, "
          f"{out.shape[1]} colonnes")
    print("colonnes:", list(out.columns))


if __name__ == "__main__":
    main()
