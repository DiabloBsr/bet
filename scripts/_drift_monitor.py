"""MONITEUR DE DÉRIVE RNG — surveille si la réalité décroche des cotes offertes.
Marché efficient => gap (réel - implicite_devig) ≈ 0 par marché. Si le RNG change
ou si les cotes lag, le gap se creuse => fenêtre +EV. Le moniteur :
  1) tendance temporelle (par jour) : calibration favori, Over2.5, BTTS réel vs book ;
  2) fenêtre récente vs baseline : z-test du gap par marché ;
  3) verdict VERT (efficient) / JAUNE (à surveiller) / ROUGE (edge détecté).
Cron-able : à lancer périodiquement. Usage: ./.venv/Scripts/python.exe scripts/_drift_monitor.py [window]
"""
from __future__ import annotations
import sys, math
from datetime import timezone, timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.market_inversion import parse_extra_markets, total_buts_odds, devig_market

WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
MG = timezone(timedelta(hours=3))

s = load_settings(); e = create_engine(s.db_url)
df = pd.read_sql("""SELECT e.team_a,e.team_b,e.expected_start,o.odds_home oh,o.odds_draw od,o.odds_away oa,
    o.extra_markets, r.score_a sa, r.score_b sb FROM events e
    JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
    JOIN results r ON r.event_id=e.id WHERE r.score_a IS NOT NULL AND e.competition='InstantLeague-8035'
    ORDER BY e.expected_start""", e)
df = df.drop_duplicates(["team_a","team_b","expected_start"]).reset_index(drop=True)
df["es"] = pd.to_datetime(df.expected_start, utc=True)
df["day"] = df.es.dt.tz_convert(MG).dt.strftime("%m-%d")
df["sa"] = df.sa.astype(int); df["sb"] = df.sb.astype(int)
df["tot"] = df.sa + df.sb

# --- implicites devig depuis les cotes offertes ---
inv = 1/df.oh + 1/df.od + 1/df.oa
df["imp_h"] = (1/df.oh)/inv; df["imp_d"] = (1/df.od)/inv; df["imp_a"] = (1/df.oa)/inv
df["real_h"] = (df.sa>df.sb).astype(int); df["real_d"]=(df.sa==df.sb).astype(int); df["real_a"]=(df.sa<df.sb).astype(int)
# favori : proba implicite max
df["imp_fav"] = df[["imp_h","imp_d","imp_a"]].max(axis=1)
df["real_fav"] = df.apply(lambda r: [r.real_h,r.real_d,r.real_a][int(np.argmax([r.imp_h,r.imp_d,r.imp_a]))], axis=1)
# totaux : book P(over2.5) devig depuis l'échelle, real over2.5
def book_o25(em):
    tb = total_buts_odds(parse_extra_markets(em))
    if len(tb) < 4: return np.nan
    d = devig_market(tb); return sum(v for k,v in d.items() if k.isdigit() and int(k)>=3)
df["book_o25"] = df.extra_markets.apply(book_o25); df["real_o25"]=(df.tot>=3).astype(int)
# BTTS
def book_btts(em):
    m = parse_extra_markets(em).get("G/NG") or {}
    if m.get("Oui") and m.get("Non"):
        return devig_market({"Oui":m["Oui"],"Non":m["Non"]}).get("Oui", np.nan)
    return np.nan
df["book_btts"] = df.extra_markets.apply(book_btts); df["real_btts"]=((df.sa>=1)&(df.sb>=1)).astype(int)

def z_gap(real, imp, n):
    p = imp.mean()
    if n<=0 or p<=0 or p>=1: return 0.0, 0.0
    gap = real.mean()-p
    return gap, gap/math.sqrt(p*(1-p)/n)

print("="*84)
print(f"MONITEUR DÉRIVE RNG — {len(df)} matchs | fenêtre récente = {WINDOW} derniers")
print("="*84)

# 1) TENDANCE par jour (les derniers jours)
print("\n■ TENDANCE PAR JOUR (réel vs cote-implicite ; gap ~0 = efficient)")
print(f"  {'jour':<7}{'n':>5}{'favWR':>8}{'favImp':>8}{'gap':>7} | {'O2.5r':>7}{'O2.5b':>7}{'gap':>7} | {'BTTSr':>7}{'BTTSb':>7}{'gap':>7}")
for day, g in df.groupby("day"):
    if len(g) < 80: continue
    fr=g.real_fav.mean(); fi=g.imp_fav.mean()
    o2r=g.real_o25.mean(); o2b=g.book_o25.mean(); br=g.real_btts.mean(); bb=g.book_btts.mean()
    print(f"  {day:<7}{len(g):>5}{fr*100:>7.1f}%{fi*100:>7.1f}%{(fr-fi)*100:>+6.1f} | "
          f"{o2r*100:>6.0f}%{o2b*100:>6.0f}%{(o2r-o2b)*100:>+6.1f} | {br*100:>6.0f}%{bb*100:>6.0f}%{(br-bb)*100:>+6.1f}")

# 2) FENÊTRE RÉCENTE vs BASELINE : z du gap par marché
print(f"\n■ DÉRIVE : fenêtre récente ({WINDOW}) vs baseline (reste)")
recent = df.tail(WINDOW); base = df.iloc[:-WINDOW] if len(df)>WINDOW else df
markets = [
  ("Favori 1X2", "real_fav","imp_fav"),
  ("Over 2.5", "real_o25","book_o25"),
  ("BTTS Oui", "real_btts","book_btts"),
  ("Home 1", "real_h","imp_h"), ("Away 2","real_a","imp_a"), ("Draw X","real_d","imp_d"),
]
alerts = []
print(f"  {'marché':<14}{'gap récent':>12}{'z récent':>10}{'gap base':>11}{'EV récent*':>12}{'statut':>9}")
for nm, rc, ic in markets:
    r = recent.dropna(subset=[ic]); b = base.dropna(subset=[ic])
    gr, zr = z_gap(r[rc], r[ic], len(r))
    gb, _ = z_gap(b[rc], b[ic], len(b))
    # EV approx à la cote offerte : gap / imp (le surplus de proba réelle converti en rendement brut, marge non incluse ici)
    ev = gr / r[ic].mean() if r[ic].mean()>0 else 0
    # statut : ROUGE si gap récent franchement >0 et z fort ET au-dessus de la baseline
    drift = gr - gb
    if zr >= 3 and gr > 0.03 and drift > 0.02:
        st = "🔴 EDGE?"; alerts.append((nm, gr, zr))
    elif zr >= 2.3 and drift > 0.015:
        st = "🟡 watch"
    else:
        st = "🟢 ok"
    print(f"  {nm:<14}{gr*100:>+11.1f}%{zr:>+10.1f}{gb*100:>+10.1f}%{ev*100:>+11.0f}%{st:>9}")

print("\n" + "="*84)
if alerts:
    print("🔴 ALERTE DÉRIVE — marchés où le réel décroche des cotes (potentiel +EV à vérifier) :")
    for nm,gr,zr in alerts: print(f"   • {nm}: réel +{gr*100:.1f}pp au-dessus de l'implicite (z {zr:+.1f})")
    print("   -> Lancer une vérif OOS dédiée avant tout pari. Peut être le début d'une fenêtre de mispricing.")
else:
    print("🟢 RAS — marché efficient, aucun décrochage. Le RNG et les cotes sont synchrones (normal).")
print("   Note: *EV récent = surplus de proba réelle / implicite (marge ~12% NON déduite). Pour un vrai +EV net, il")
print("   faut gap récent > marge du marché. Relancer ce moniteur régulièrement (cron) pour capter une dérive future.")
