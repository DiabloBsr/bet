"""Backtest LIVE des edges : hit-rate réel des signaux EDGE TOTAL / CHAÎNAGE /
SCORE 1-1 / E2 sur les matchs joués APRÈS la dérivation (vrai forward, jamais vu).
Compare réel vs annoncé."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import invert_markets, apply_sim_deviations, grid_predictions

ROOT = Path(__file__).resolve().parents[1]
def _load(n):
    try: return json.load(open(ROOT/"exports"/n, encoding="utf-8"))
    except Exception: return None
def _band(v, bands):
    for lo, hi, lbl in bands:
        if lo <= v < hi: return lbl
    return None
def _chain(ct, lt, ld, pb):
    if not ct or pb is None: return None
    b = ct["_bands"]; tl=_band(lt,b["tot"]); dl=_band(ld,b["diff"]); bl=_band(pb,b["btts"])
    return ct["cells"].get(f"{tl}|{dl}|{bl}") if (tl and dl and bl) else None

chain_tab = _load("chain_table.json")
# cutoff = max expected_start du CSV de dérivation -> tout après = jamais vu
csv = pd.read_csv(ROOT/"exports"/"combokeys_features.csv", usecols=["expected_start"])
cutoff = csv.expected_start.max()
print(f"Cutoff dérivation : {cutoff}  → on teste UNIQUEMENT les matchs joués après.\n")

s = load_settings(); e = create_engine(s.db_url)
df = pd.read_sql(f"""SELECT e.team_a,e.team_b,e.expected_start,
    o.odds_home oh,o.odds_draw od,o.odds_away oa,o.extra_markets,
    r.score_a sa,r.score_b sb FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id
    WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
      AND e.expected_start > '{cutoff}'""", e)
df = df.drop_duplicates(["team_a","team_b","expected_start"])
print(f"Matchs frais (hors-échantillon strict) : {len(df)}\n")
if len(df) < 30:
    print("Trop peu de matchs frais — j'élargis aux 2500 derniers résultats (dont une partie chevauche la dérivation, à titre indicatif).")
    df = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,
        o.odds_away oa,o.extra_markets,r.score_a sa,r.score_b sb FROM events e
        JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
        JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
        ORDER BY e.expected_start DESC LIMIT 2500""", e)
    df = df.drop_duplicates(["team_a","team_b","expected_start"])
    print(f"→ {len(df)} matchs (récents).\n")

# tallies
T = {}
def tally(k, hit):
    a = T.setdefault(k, [0,0]); a[0]+=1; a[1]+=1 if hit else 0

for r in df.itertuples():
    if not (r.oh and r.od and r.oa) or r.oh<=1 or r.oa<=1 or r.od<=1: continue
    inv = invert_markets(float(r.oh), float(r.od), float(r.oa), r.extra_markets)
    lt, ld = inv.lam_h+inv.lam_a, inv.lam_h-inv.lam_a
    gp = grid_predictions(apply_sim_deviations(inv.lam_h, inv.lam_a, "cells"), top_k=3)
    pb = gp["btts_oui"]
    tot = int(r.sa)+int(r.sb); sc = f"{int(r.sa)}-{int(r.sb)}"
    fav = min(r.oh, r.oa)

    # EDGE TOTAL
    if lt < 2.45: tally("EDGE Under3.5 (λ<2.45) [annoncé 76%]", tot<=3)
    elif lt >= 3.13: tally("EDGE Over2.5 (λ≥3.13) [annoncé 72%]", tot>=3)
    # SCORE 1-1 rule
    if abs(ld) < 0.74 and lt < 2.45: tally("SCORE 1-1 (équil+faible tot) [annoncé 14%]", sc=="1-1")
    # CHAÎNAGE
    ch = _chain(chain_tab, lt, ld, pb)
    if ch:
        tally("CHAÎNAGE score modal [annoncé ~13%]", sc==ch["score"])
        tally("CHAÎNAGE Top-3 [annoncé ~30%]", sc in ch["top3"])
        tally("CHAÎNAGE direction O/U [annoncé ~60-78%]", (tot>=3) if ch["ou"]=="Over2.5" else (tot<=2))
    # E2 : favori extrême home ∈[1.10,1.20]
    if 1.10 <= r.oh <= 1.20: tally("E2 1X2 '1' fav extrême (oh∈[1.10,1.20]) [annoncé ~86%]", r.sa>r.sb)
    # référence : favori 1X2 gagne
    if fav <= 1.50:
        home_fav = r.oh < r.oa
        won = (r.sa>r.sb) if home_fav else (r.sb>r.sa)
        tally("Réf : favori ≤1.50 gagne", won)

print("="*86)
print(f"{'SIGNAL':<48} {'n':>6} {'hits':>6} {'RÉEL':>7}")
print("="*86)
for k, (n, h) in sorted(T.items()):
    print(f"{k:<48} {n:>6} {h:>6} {h/n*100:>6.0f}%")
print("="*86)
