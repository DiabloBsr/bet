"""Dataset CHASSE DE CLÔTURE — les 4 variantes jamais testées :
  A) chaînes de buts/BTTS par équipe, jugées contre les cotes Over/Under 3.5 et G/NG OFFERTES
  B) chaînes de résultats longueur >=3 (streaks 3, 4+)
  C) composition du round précédent -> match courant
  D) répétition de score exact, jugée contre la cote score-exact OFFERTE du score répété

Produit :
  data/vfoot_ml/closing_team.csv  (1 ligne = équipe-match, chaînes + cotes offertes)
  data/vfoot_ml/closing_round.csv (1 ligne = match, + agrégats du round précédent)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"

eng = create_engine(load_settings().db_url)
df = pd.read_sql(text(f"""
    SELECT e.expected_start ts, e.team_a, e.team_b,
           o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
           r.score_a sa, r.score_b sb
    FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='{LG}'
      AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
    ORDER BY e.expected_start"""), eng)
df = df.drop_duplicates(["ts", "team_a", "team_b"]).reset_index(drop=True)


def gm(xm, name):
    if isinstance(xm, dict):
        if name in xm:
            return xm[name]
        for k, v in xm.items():
            if k.startswith(name):
                return v
    return None


def _odds_ok(v):
    return v if isinstance(v, (int, float)) and v > 1 else np.nan


# ---- parse extra_markets une fois par match ----
o_over, o_under, o_by, o_bn, se_maps = [], [], [], [], []
for raw in df.xm:
    try:
        xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        xm = {}
    pm = gm(xm, "+/-") or {}
    o_over.append(_odds_ok(pm.get("> 3.5"))); o_under.append(_odds_ok(pm.get("< 3.5")))
    gn = gm(xm, "G/NG") or {}
    o_by.append(_odds_ok(gn.get("Oui"))); o_bn.append(_odds_ok(gn.get("Non")))
    se = gm(xm, "Score exact") or {}
    se_maps.append({k.replace(":", "-").replace(" ", ""): v for k, v in se.items()
                    if isinstance(v, (int, float)) and v > 1})
df["o_over35"], df["o_under35"] = o_over, o_under
df["o_btts_oui"], df["o_btts_non"] = o_by, o_bn
df["_se"] = se_maps

df["sa6"] = df.sa.clip(0, 6); df["sb6"] = df.sb.clip(0, 6)
df["total"] = df.sa + df.sb
df["over35"] = (df.total > 3.5).astype(int)
df["over25"] = (df.total > 2.5).astype(int)
df["btts"] = ((df.sa > 0) & (df.sb > 0)).astype(int)
df["exact"] = df.sa6.astype(str) + "-" + df.sb6.astype(str)
inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_h"] = (1/df.oh)/inv; df["imp_d"] = (1/df.od)/inv; df["imp_a"] = (1/df.oa)/inv

# ================= CSV A : équipe-match =================
rows = []
for side in ("H", "A"):
    t = pd.DataFrame({
        "ts": df.ts, "team": df.team_a if side == "H" else df.team_b,
        "venue": side,
        "odds": df.oh if side == "H" else df.oa,
        "imp": df.imp_h if side == "H" else df.imp_a,
        "win": (df.sa > df.sb).astype(int) if side == "H" else (df.sb > df.sa).astype(int),
        "result_c": np.where(df.sa == df.sb, "D",
                             np.where((df.sa > df.sb) == (side == "H"), "W", "L")),
        "total": df.total, "over35": df.over35, "over25": df.over25, "btts": df.btts,
        "exact": df.exact,
        "o_over35": df.o_over35, "o_under35": df.o_under35,
        "o_btts_oui": df.o_btts_oui, "o_btts_non": df.o_btts_non,
        "_se_idx": df.index,
    })
    rows.append(t)
L = pd.concat(rows, ignore_index=True).sort_values(["team", "ts"]).reset_index(drop=True)
g = L.groupby("team")
for k in (1, 2, 3, 4):
    L[f"p{k}_result"] = g["result_c"].shift(k)
for k in (1, 2, 3):
    L[f"p{k}_total"] = g["total"].shift(k)
    L[f"p{k}_over25"] = g["over25"].shift(k)
    L[f"p{k}_btts"] = g["btts"].shift(k)
L["p1_exact"] = g["exact"].shift(1)
L["p2_exact"] = g["exact"].shift(2)

# streaks entrants (cap 4) : longueur de la série identique se terminant AVANT le match
def entering_streak(cols):
    base = L[cols[0]]
    s = pd.Series(0, index=L.index)
    ok = base.notna()
    s[ok] = 1
    for i in range(1, len(cols)):
        ok = ok & (L[cols[i]] == base) & L[cols[i]].notna()
        s[ok] += 1
    return s

L["result_streak"] = entering_streak(["p1_result", "p2_result", "p3_result", "p4_result"])
L["over25_streak"] = entering_streak(["p1_over25", "p2_over25", "p3_over25"])
L["btts_streak"] = entering_streak(["p1_btts", "p2_btts", "p3_btts"])

# cote score-exact OFFERTE (match courant) pour rejouer p1_exact
se_list = df["_se"].tolist()
L["odds_repeat_p1exact"] = [
    (se_list[int(i)].get(px, np.nan) if isinstance(px, str) else np.nan)
    for i, px in zip(L._se_idx, L.p1_exact)]
L = L.drop(columns=["_se_idx"]).dropna(subset=["p1_result"])
L.to_csv("data/vfoot_ml/closing_team.csv", index=False)
print(f"closing_team.csv : {len(L)} lignes | odds_repeat dispo: {int(L.odds_repeat_p1exact.notna().sum())}")

# ================= CSV B : match + agrégats du round précédent =================
M = df[["ts", "imp_h", "imp_d", "imp_a", "od", "o_over35", "o_under35",
        "o_btts_oui", "o_btts_non", "total", "over35", "over25", "btts",
        "sa", "sb"]].copy()
M["draw"] = (M.sa == M.sb).astype(int)
agg = M.groupby("ts").agg(n=("draw", "size"), n_draws=("draw", "sum"),
                          n_over25=("over25", "sum"), n_over35=("over35", "sum"),
                          n_btts=("btts", "sum"), goals=("total", "sum")).reset_index()
agg = agg.sort_values("ts").reset_index(drop=True)
for c in ("n", "n_draws", "n_over25", "n_over35", "n_btts", "goals"):
    agg[f"prev_{c}"] = agg[c].shift(1)
M = M.merge(agg[["ts"] + [f"prev_{c}" for c in ("n", "n_draws", "n_over25", "n_over35", "n_btts", "goals")]],
            on="ts", how="left").dropna(subset=["prev_n"])
M.to_csv("data/vfoot_ml/closing_round.csv", index=False)
print(f"closing_round.csv : {len(M)} matchs, {agg.ts.nunique()} rounds")
