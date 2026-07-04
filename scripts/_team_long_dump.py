"""Dump du format long (1 ligne par équipe-match) avec résidus, pour analyse
par agents : data/team_long.csv. Lecture seule.
Colonnes : ts, team, opp, is_home, oh, od, oa, p_win, p_draw, won, drew, gf, ga, tot, resid
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
_SQL = """
SELECT e.expected_start ts, e.team_a home, e.team_b away,
       o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
JOIN results r ON r.event_id=e.id
WHERE r.score_a IS NOT NULL AND e.competition=:lg
  AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
ORDER BY e.expected_start
"""
df = pd.read_sql(text(_SQL), create_engine(load_settings().db_url), params={"lg": LG})
ih, idr, ia = 1/df.oh, 1/df.od, 1/df.oa; s = ih+idr+ia
df["p1"]=ih/s; df["pX"]=idr/s; df["p2"]=ia/s
H = pd.DataFrame({"ts":df.ts,"team":df.home,"opp":df.away,"is_home":1,"oh":df.oh,"od":df.od,"oa":df.oa,
                  "p_win":df.p1,"p_draw":df.pX,"won":(df.sa>df.sb).astype(int),"drew":(df.sa==df.sb).astype(int),
                  "gf":df.sa,"ga":df.sb,"tot":df.sa+df.sb})
A = pd.DataFrame({"ts":df.ts,"team":df.away,"opp":df.home,"is_home":0,"oh":df.oa,"od":df.od,"oa":df.oh,
                  "p_win":df.p2,"p_draw":df.pX,"won":(df.sb>df.sa).astype(int),"drew":(df.sa==df.sb).astype(int),
                  "gf":df.sb,"ga":df.sa,"tot":df.sa+df.sb})
L = pd.concat([H,A], ignore_index=True).sort_values("ts").reset_index(drop=True)
L["resid"] = L.won - L.p_win
L.to_csv("data/team_long.csv", index=False)
print(f"écrit data/team_long.csv : {len(L)} lignes, {L.team.nunique()} équipes")
