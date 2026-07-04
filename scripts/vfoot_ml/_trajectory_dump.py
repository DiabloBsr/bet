"""Dataset TRAJECTOIRE par équipe : passé (2 matchs), présent, FUTUR (cotes du
match suivant capturées AVANT le match courant -> test de fuite).

1 ligne = 1 (équipe, match). Colonnes :
  présent : ts, team, opp, venue(H/A), odds(cote victoire équipe), imp(dévig),
            imp_draw, result(W/D/L), gf, ga, resid(=win-imp)
  passé   : p1_odds, p1_imp, p1_venue, p1_result, p1_margin, d_odds(=odds-p1_odds),
            p2_odds, p2_result, d_odds2(=p1_odds-p2_odds), gap_min(minutes depuis p1)
  futur   : fut_odds, fut_imp, fut_seen_before (1 si la 1re capture des cotes du
            match SUIVANT est antérieure au coup d'envoi du match courant)

-> data/vfoot_ml/trajectory.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"

eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, e.team_a, e.team_b,
           o.odds_home oh, o.odds_draw od, o.odds_away oa, o.captured_at cap,
           r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}'
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng)

inv = 1/df.oh + 1/df.od + 1/df.oa
imp_h, imp_d, imp_a = (1/df.oh)/inv, (1/df.od)/inv, (1/df.oa)/inv

rows = []
for side in ("H", "A"):
    if side == "H":
        t = pd.DataFrame({"ts": df.ts, "cap": df.cap, "team": df.team_a, "opp": df.team_b,
                          "venue": "H", "odds": df.oh, "imp": imp_h, "imp_draw": imp_d,
                          "gf": df.sa, "ga": df.sb})
    else:
        t = pd.DataFrame({"ts": df.ts, "cap": df.cap, "team": df.team_b, "opp": df.team_a,
                          "venue": "A", "odds": df.oa, "imp": imp_a, "imp_draw": imp_d,
                          "gf": df.sb, "ga": df.sa})
    rows.append(t)
L = pd.concat(rows, ignore_index=True).sort_values(["team", "ts"]).reset_index(drop=True)
L["win"] = (L.gf > L.ga).astype(int)
L["result"] = np.where(L.gf > L.ga, "W", np.where(L.gf == L.ga, "D", "L"))
L["margin"] = L.gf - L.ga
L["resid"] = L.win - L.imp

g = L.groupby("team")
for k in (1, 2):
    L[f"p{k}_odds"] = g["odds"].shift(k)
    L[f"p{k}_imp"] = g["imp"].shift(k)
    L[f"p{k}_result"] = g["result"].shift(k)
    L[f"p{k}_venue"] = g["venue"].shift(k)
    L[f"p{k}_margin"] = g["margin"].shift(k)
L["d_odds"] = L.odds - L.p1_odds            # chute (<0) / hausse (>0) de cote vs match précédent
L["d_odds2"] = L.p1_odds - L.p2_odds
L["gap_min"] = (pd.to_datetime(L.ts) - pd.to_datetime(g["ts"].shift(1))).dt.total_seconds() / 60

# ---- FUTUR : cotes du match SUIVANT, capturées avant le match courant ? ----
L["fut_odds"] = g["odds"].shift(-1)
L["fut_imp"] = g["imp"].shift(-1)
L["fut_cap"] = g["cap"].shift(-1)
L["fut_seen_before"] = (pd.to_datetime(L.fut_cap) < pd.to_datetime(L.ts)).astype(int)

out = L.drop(columns=["cap", "fut_cap"]).dropna(subset=["p1_odds"])
out.to_csv("data/vfoot_ml/trajectory.csv", index=False)
print(f"écrit data/vfoot_ml/trajectory.csv : {len(out)} lignes (équipe-match), "
      f"{out.team.nunique()} équipes")
print(f"fuite-futur testable : {int(out.fut_seen_before.sum())} lignes où les cotes du match "
      f"suivant étaient VISIBLES avant le match courant")
print("colonnes:", list(out.columns))
