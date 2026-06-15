"""Ré-évaluation des prédictions de ce soir (rounds 23:58→01:01 Mada) contre les
résultats RÉELS maintenant en base. Score chaque signal + détail par match des
rounds phares (23:58 et 01:01)."""
from __future__ import annotations
import sys, json
from datetime import timezone, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import invert_markets, apply_sim_deviations, grid_predictions

ROOT = Path(__file__).resolve().parents[1]; MG = timezone(timedelta(hours=3))
chain_tab = json.load(open(ROOT/"exports"/"chain_table.json", encoding="utf-8"))
def _band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi: return lbl
    return None
def _chain(lt, ld, pb):
    if pb is None: return None
    b = chain_tab["_bands"]; tl=_band(lt,b["tot"]); dl=_band(ld,b["diff"]); bl=_band(pb,b["btts"])
    return chain_tab["cells"].get(f"{tl}|{dl}|{bl}") if (tl and dl and bl) else None

s = load_settings(); e = create_engine(s.db_url)
# fenêtre UTC des prédictions de ce soir : 23:56 06-14 → 01:05 06-15 Mada
df = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
    o.odds_away oa,o.extra_markets,r.score_a sa,r.score_b sb FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
      AND e.expected_start >= '2026-06-14 20:56:00' AND e.expected_start <= '2026-06-14 22:05:00'""", e)
df = df.drop_duplicates(["team_a","team_b","expected_start"])
df["es"] = pd.to_datetime(df.expected_start, utc=True)
df["local"] = df.es.dt.tz_convert(MG).dt.strftime("%H:%M")
df = df.sort_values("es")
print(f"Matchs ré-évalués (rounds {df.local.min()}→{df.local.max()}) : {len(df)}\n")

T = {}
def tally(k, hit):
    a = T.setdefault(k, [0,0]); a[0]+=1; a[1]+=1 if hit else 0

rows = []
for r in df.itertuples():
    if not (r.oh and r.od and r.oa) or r.oh<=1 or r.oa<=1 or r.od<=1: continue
    inv = invert_markets(float(r.oh), float(r.od), float(r.oa), r.extra_markets)
    lt, ld = inv.lam_h+inv.lam_a, inv.lam_h-inv.lam_a
    gp = grid_predictions(apply_sim_deviations(inv.lam_h, inv.lam_a, "cells"), top_k=3)
    pb = gp["btts_oui"]; sim_top = [sc for sc,_ in gp["top_scores"]]
    tot = int(r.sa)+int(r.sb); sc = f"{int(r.sa)}-{int(r.sb)}"
    home_fav = r.oh < r.oa; fav = min(r.oh, r.oa)
    pick = "1" if home_fav else "2"; won = (r.sa>r.sb) if home_fav else (r.sb>r.sa)

    tally("1X2 : favori (mon pick FT)", won)
    if fav <= 1.50: tally("1X2 : favori ≤1.50 (mes TIER1)", won)
    if 1.10 <= r.oh <= 1.20: tally("E2 : fav extrême home", r.sa>r.sb)
    if lt < 2.45: tally("EDGE Under3.5", tot<=3)
    elif lt >= 3.13: tally("EDGE Over2.5", tot>=3)
    tally("Score modal (sim top-1)", sc==sim_top[0])
    tally("Score Top-3 (sim)", sc in sim_top)
    ch = _chain(lt, ld, pb)
    if ch:
        tally("CHAÎNAGE score modal", sc==ch["score"])
        tally("CHAÎNAGE Top-3", sc in ch["top3"])
        tally("CHAÎNAGE direction O/U", (tot>=3) if ch["ou"]=="Over2.5" else (tot<=2))
    rows.append((r.local, f"{r.team_a} v {r.team_b}", r.oh, r.od, r.oa, pick, "✓" if won else "✗",
                 ("U3.5" if lt<2.45 else "O2.5" if lt>=3.13 else "-"),
                 sim_top[0], (ch["score"] if ch else "-"), sc, tot))

print("="*78)
print(f"{'SIGNAL':<34} {'n':>5} {'hits':>5} {'RÉEL':>6}  annoncé")
print("="*78)
ann = {"1X2 : favori (mon pick FT)":"~", "1X2 : favori ≤1.50 (mes TIER1)":"72%",
       "E2 : fav extrême home":"86%","EDGE Under3.5":"76%","EDGE Over2.5":"72%",
       "Score modal (sim top-1)":"~12%","Score Top-3 (sim)":"~30%",
       "CHAÎNAGE score modal":"~13%","CHAÎNAGE Top-3":"~30%","CHAÎNAGE direction O/U":"60-78%"}
for k in ["1X2 : favori (mon pick FT)","1X2 : favori ≤1.50 (mes TIER1)","E2 : fav extrême home",
          "EDGE Under3.5","EDGE Over2.5","CHAÎNAGE direction O/U","Score Top-3 (sim)","CHAÎNAGE Top-3",
          "Score modal (sim top-1)","CHAÎNAGE score modal"]:
    if k in T:
        n,h = T[k]; print(f"{k:<34} {n:>5} {h:>5} {h/n*100:>5.0f}%  {ann.get(k,'')}")
print("="*78)

# détail des rounds phares
for rd in ["23:58","01:01"]:
    sub = [x for x in rows if x[0]==rd]
    if not sub: continue
    print(f"\n── DÉTAIL ROUND {rd} (mes prédictions vs réel) ──")
    print(f"{'match':<32}{'pick':>5}{'1X2':>4}{'edge':>6}{'modal':>7}{'chaîne':>8}  {'RÉEL':>5} tot")
    for lc,mn,oh,od,oa,pk,w,edg,md,chs,real,tt in sub:
        print(f"{mn:<32}{pk:>5}{w:>4}{edg:>6}{md:>7}{chs:>8}  {real:>5} {tt}")
