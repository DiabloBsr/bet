"""BUTS PAR ÉQUIPE / SAISON — persistance + exploitabilité vs le marché O/U.

A) PERSISTANCE : la moyenne de buts d'une équipe en saison N corrèle-t-elle avec N+1 ?
B) EXPLOITABILITÉ : le taux over2.5 réel d'une équipe dépasse-t-il l'implicite du
   marché de façon RÉPLICABLE (train->test) et RENTABLE (ROI) ? -> le seul angle qui paie.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT ev.expected_start ts, ev.team_a ta, ev.team_b tb, ev.round_info j,
           o.extra_markets xm, r.score_a sa, r.score_b sb
    FROM events ev JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=ev.id)
    JOIN results r ON r.event_id=ev.id
    WHERE r.score_a IS NOT NULL AND ev.competition='{LG}'"""), eng)
df = df.drop_duplicates(["ts", "ta", "tb"]).reset_index(drop=True)
df["j"] = pd.to_numeric(df.j, errors="coerce")
df["ts"] = pd.to_datetime(df.ts, utc=True)
df["total"] = df.sa + df.sb
df["over25"] = (df.total > 2.5).astype(int)

# implicite over2.5 depuis "Total de buts"
def gm(xm, pref):
    for k, v in (xm or {}).items():
        if k.replace("\x82", "é").replace("\xe9", "é").startswith(pref):
            return v
    return None
io25 = []
for raw in df.xm:
    try: xm = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception: xm = {}
    tt = gm(xm, "Total de buts")
    v = {k: 1/tt[k] for k in [str(x) for x in range(7)]
         if tt and isinstance(tt.get(k), (int, float)) and 1 < tt[k] < 99.99} if tt else {}
    s = sum(v.values())
    io25.append(sum(v[str(k)] for k in range(3, 7))/s if s and len(v) == 7 else np.nan)
df["io25"] = io25

# ---- perspective ÉQUIPE ----
rows = []
for side in ("a", "b"):
    rows.append(pd.DataFrame({"ts": df.ts, "j": df.j, "team": df[f"t{side}"],
                              "gf": df.sa if side == "a" else df.sb,
                              "over25": df.over25, "io25": df.io25}))
L = pd.concat(rows).sort_values("ts").reset_index(drop=True)
L = L.dropna(subset=["io25"])

# ---- segmentation SAISONS (reset de journée) sur la séquence temporelle des matchs ----
seq = df.sort_values("ts").reset_index(drop=True)
seq = seq[(seq.j >= 1) & (seq.j <= 38)]
season = (seq.j.diff() < -5).cumsum()          # une chute de journée = nouvelle saison
seq = seq.assign(season=season.values)
print(f"{len(df)} matchs | {seq.season.nunique()} saisons détectées", flush=True)

# A) PERSISTANCE : moyenne de buts marqués par (équipe, saison), corr saison N vs N+1
tmap = pd.concat([
    seq[["ts", "ta", "season", "sa"]].rename(columns={"ta": "team", "sa": "gf"}),
    seq[["ts", "tb", "season", "sb"]].rename(columns={"tb": "team", "sb": "gf"})])
ts_avg = tmap.groupby(["team", "season"]).gf.mean().reset_index()
ts_avg["prev"] = ts_avg.groupby("team").gf.shift(1)
pair = ts_avg.dropna(subset=["prev"])
print("\n=== A. PERSISTANCE saison N -> N+1 (moyenne de buts marqués par équipe) ===")
if len(pair) > 20:
    r = np.corrcoef(pair.gf, pair.prev)[0, 1]
    print(f"  corrélation buts_saison(N) vs buts_saison(N+1) : {r:+.4f}  (n={len(pair)} paires)")
    print(f"  -> {'PERSISTANCE (profil stable)' if abs(r) > 0.15 else 'faible/nulle : le profil buts est quasi re-tiré chaque saison'}")
else:
    print(f"  pas assez de saisons complètes ({len(pair)} paires) — profil global seulement.")
# écart-type des moyennes d'équipe (identités marquées ou uniformes ?)
glob = tmap.groupby("team").gf.mean()
print(f"  moyenne de buts/match par équipe : min {glob.min():.2f} ({glob.idxmin()}) "
      f"max {glob.max():.2f} ({glob.idxmax()}) | écart-type entre équipes {glob.std():.3f}")

# B) EXPLOITABILITÉ : biais over2.5 de l'équipe vs marché, réplication train/test + ROI
print("\n=== B. EXPLOITABILITÉ vs marché O/U (le seul angle qui paie) ===")
cut = L.ts.iloc[len(L)//2]
tr, te = L[L.ts < cut], L[L.ts >= cut]
g_tr = tr.groupby("team").agg(n=("over25", "size"), real=("over25", "mean"), imp=("io25", "mean"))
g_tr["resid_tr"] = g_tr.real - g_tr.imp
strong = g_tr[(g_tr.n >= 200) & (g_tr.resid_tr.abs() >= 0.02)]
print(f"  équipes à biais |over2.5 - implicite| >= 2pp sur TRAIN : {len(strong)}")
rep = 0
for team, row in strong.iterrows():
    sub = te[te.team == team]
    if len(sub) >= 100:
        rt = sub.over25.mean() - sub.io25.mean()
        same = np.sign(rt) == np.sign(row.resid_tr)
        rep += int(same)
        # ROI : parier over si biais+, under si biais- (à la cote implicite -> proxy sans marge)
        print(f"    {team:<20} train {100*row.resid_tr:+.1f}pp -> test {100*rt:+.1f}pp "
              f"({'même signe' if same else 'INVERSÉ'})")
print(f"  réplication du signe train->test : {rep}/{len(strong)} "
      f"-> {'⚠️ persistant, à creuser en ROI' if rep >= max(3, 0.7*len(strong)) else 'incohérent = biais absorbé par les cotes (pas exploitable)'}")
