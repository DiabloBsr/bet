"""OÙ PLAFONNE-T-ON LE PLUS ? — étude round-par-round (38 journées) + par marché.

Q1 : la prévisibilité (confiance max atteignable) varie-t-elle selon la JOURNÉE ?
Q2 : quel MARCHÉ monte le plus haut (plafond de réussite réelle) ?
-> stratégie "quand jouer quel marché".
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings
import predict_trio as pt

LG = "InstantLeague-8035"
eng = create_engine(load_settings().db_url, connect_args={"timeout": 30})
df = pd.read_sql(text(f"""
    SELECT ev.round_info j, o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm,
           r.score_a sa, r.score_b sb, r.ht_score_a ha, r.ht_score_b hb
    FROM events ev JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=ev.id)
    JOIN results r ON r.event_id=ev.id
    WHERE r.ht_score_a IS NOT NULL AND ev.competition='{LG}' AND o.odds_home>1"""), eng)
print(f"{len(df)} matchs", flush=True)


def settle(mkt, sel, sa, sb, ha, hb):
    tot = sa + sb; res = "1" if sa > sb else ("2" if sb > sa else "X")
    hres = "1" if ha > hb else ("2" if hb > ha else "X")
    if mkt == "1X2": return sel == res
    if mkt == "Mi-tps 1X2": return sel == hres
    if mkt == "Double Chance": return res in sel
    if mkt == "Mi-tps DC": return hres in sel
    if mkt == "+/-": return (tot > 3.5) if ">" in sel else (tot < 3.5)
    if mkt == "G/NG": return (sa > 0 and sb > 0) == (sel == "Oui")
    if mkt == "Total de buts":
        k = int(sel); return (tot >= 6) if k == 6 else (tot == k)
    if mkt == "Multi-Buts":
        if "0, 1 ou 2" in sel: return tot <= 2
        if "1, 2 ou 3" in sel: return 1 <= tot <= 3
        if "2, 3 ou 4" in sel: return 2 <= tot <= 4
        return tot > 4
    return None


rows = []
for r in df.itertuples():
    try:
        xm = json.loads(r.xm) if isinstance(r.xm, str) else (r.xm or {})
    except Exception:
        continue
    board = pt.market_board(xm, r.oh, r.od, r.oa)
    best = pt.top_confidence_pick(board)              # meilleur pari sûr du match
    if not best:
        continue
    mkt, sel, p, o = best
    h = settle(mkt, sel, int(r.sa), int(r.sb), int(r.ha), int(r.hb))
    if h is None:
        continue
    j = pd.to_numeric(r.j, errors="coerce")
    rows.append((j, p, int(h), mkt))
    # par marché : meilleur pick de CE marché
    for mk in pt.CONF_MARKETS:
        rr = board.get(mk) or []
        if rr:
            s2, p2, o2 = max(rr, key=lambda x: x[1])
            h2 = settle(mk, s2, int(r.sa), int(r.sb), int(r.ha), int(r.hb))
            if h2 is not None:
                rows.append((j, p2, int(h2), "MK::" + mk))
D = pd.DataFrame(rows, columns=["j", "p", "hit", "mkt"])
best_only = D[~D.mkt.str.startswith("MK::")]

print("\n=== Q1 : PRÉVISIBILITÉ PAR JOURNÉE (meilleur pari sûr par match) ===", flush=True)
jd = best_only[(best_only.j >= 1) & (best_only.j <= 38)]
by_j = jd.groupby("j").agg(n=("hit", "size"), conf=("p", "mean"), reel=("hit", "mean"))
print(f"  moyenne toutes journées : confiance {100*jd.p.mean():.1f}% | réussite réelle {100*jd.hit.mean():.1f}%")
print(f"  réussite réelle par journée : min {100*by_j.reel.min():.1f}% (J{by_j.reel.idxmin()}) | "
      f"max {100*by_j.reel.max():.1f}% (J{by_j.reel.idxmax()}) | écart-type {100*by_j.reel.std():.2f}pp")
se = np.sqrt(0.72*0.28/by_j.n.mean())
print(f"  bruit théorique attendu (SE) à n~{int(by_j.n.mean())}/journée : {100*se:.2f}pp")
print(f"  corrélation journée <-> réussite : {np.corrcoef(jd.j, jd.hit)[0,1]:+.4f}")
top5 = by_j.sort_values('reel', ascending=False).head(5)
print("  top 5 journées 'les plus prévisibles' :", [f'J{i}({100*v:.0f}%)' for i, v in top5.reel.items()])

print("\n=== Q2 : PLAFOND PAR MARCHÉ (meilleur pick du marché, réussite réelle) ===", flush=True)
mk = D[D.mkt.str.startswith("MK::")].copy(); mk["mkt"] = mk.mkt.str[4:]
res = mk.groupby("mkt").agg(n=("hit", "size"), conf=("p", "mean"), reel=("hit", "mean")).sort_values("reel", ascending=False)
print(f"  {'marché':<24}{'confiance moy':>13}{'réussite réelle':>17}")
for m, row in res.iterrows():
    print(f"  {m:<24}{100*row.conf:>12.1f}%{100*row.reel:>16.1f}%")
print("\n  -> Le marché le plus 'haut' = celui du haut du tableau. Calibré si conf ≈ réel.")
