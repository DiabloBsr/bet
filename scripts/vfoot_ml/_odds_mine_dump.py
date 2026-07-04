"""Dump du dataset de minage cote-par-cote : toutes les colonnes exploitables
(cotes, implicites, favori, lambda, issues) -> data/vfoot_ml/odds_mine.csv.
Pour la chasse multi-agents (1 agent / combinaison de features)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
from scraper.market_inversion import exact_invert_1x2

LG = "InstantLeague-8035"
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, o.odds_home oh, o.odds_draw od, o.odds_away oa, r.score_a sa, r.score_b sb
    FROM events e JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}' AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), create_engine(load_settings().db_url))

inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_h"] = (1/df.oh)/inv; df["imp_d"] = (1/df.od)/inv; df["imp_a"] = (1/df.oa)/inv
df["fav"] = df[["imp_h", "imp_a"]].max(axis=1)
df["odds_ratio"] = df.oh/df.oa
lam = [exact_invert_1x2(h, d, a) for h, d, a in zip(df.oh, df.od, df.oa)]
df["lam_h"] = [x[0] for x in lam]; df["lam_a"] = [x[1] for x in lam]
df["lam_tot"] = df.lam_h + df.lam_a; df["lam_diff"] = df.lam_h - df.lam_a
df["sa"] = df.sa.clip(0, 6); df["sb"] = df.sb.clip(0, 6); df["total"] = df.sa + df.sb
df["home_win"] = (df.sa > df.sb).astype(int)
df["draw"] = (df.sa == df.sb).astype(int)
df["away_win"] = (df.sb > df.sa).astype(int)
df["over25"] = (df.total > 2.5).astype(int)
df["btts"] = ((df.sa > 0) & (df.sb > 0)).astype(int)
df["exact"] = df.sa.astype(str) + "-" + df.sb.astype(str)

cols = ["ts", "oh", "od", "oa", "imp_h", "imp_d", "imp_a", "fav", "odds_ratio",
        "lam_h", "lam_a", "lam_tot", "lam_diff", "sa", "sb", "total",
        "home_win", "draw", "away_win", "over25", "btts", "exact"]
df[cols].to_csv("data/vfoot_ml/odds_mine.csv", index=False)
print(f"écrit data/vfoot_ml/odds_mine.csv : {len(df)} matchs, {len(cols)} colonnes")
print("colonnes:", cols)
