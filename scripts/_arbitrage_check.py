"""IDÉE #3 — Arbitrage intra-coupon : les marchés d'un même event sont-ils
cohérents, ou existe-t-il un profit garanti SANS prédire le RNG ?

Teste : (1) overround back-all par marché (somme 1/cote < 1.0 = arbitrage pur) ;
(2) cohérence croisée P(Over3.5) via '+/-' vs via 'Total de buts' (incohérence
arbitrable). Sur tous les events, toutes ligues. Lecture seule.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sqlalchemy import create_engine, text
from scraper.config import load_settings

_SQL = """
SELECT e.competition lg, o.odds_home oh, o.odds_draw od, o.odds_away oa, o.extra_markets xm
FROM events e
JOIN odds_snapshots o ON o.id=(SELECT MIN(id) FROM odds_snapshots WHERE event_id=e.id)
WHERE e.competition LIKE 'InstantLeague-%'
  AND o.extra_markets IS NOT NULL AND o.extra_markets NOT IN ('','{}','null')
  AND o.odds_home>1 AND o.odds_draw>1 AND o.odds_away>1
"""


def gm(xm, name):
    if name in xm:
        return xm[name]
    for k, v in xm.items():
        if k.startswith(name):
            return v
    return None


def overround(odds_list):
    """Somme des 1/cote sur un set d'issues exhaustives. <1.0 = arbitrage."""
    vals = [1.0 / o for o in odds_list if isinstance(o, (int, float)) and o > 1]
    return sum(vals) if vals else None


def main():
    df = pd.read_sql(text(_SQL), create_engine(load_settings().db_url))
    print(f"{len(df)} events avec marchés")
    rec = {"market_1x2": [], "market_ou35": [], "market_btts": [],
           "market_total": [], "market_score": [], "xmarket_ou35_gap": []}
    arb_events = 0
    for r in df.itertuples():
        try:
            xm = json.loads(r.xm) if isinstance(r.xm, str) else r.xm
        except Exception:
            continue
        if not isinstance(xm, dict):
            continue
        # 1X2
        o = overround([r.oh, r.od, r.oa]);  rec["market_1x2"].append(o) if o else None
        # +/- 3.5
        pm = gm(xm, "+/-")
        ou = None
        if isinstance(pm, dict):
            o = overround([pm.get("> 3.5"), pm.get("< 3.5")])
            if o:
                rec["market_ou35"].append(o)
                if o < 1.0:
                    arb_events += 1
            # P(Over3.5) dé-margé via +/-
            a, b = pm.get("> 3.5"), pm.get("< 3.5")
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > 1 and b > 1:
                ou = (1 / a) / (1 / a + 1 / b)
        # G/NG
        gn = gm(xm, "G/NG")
        if isinstance(gn, dict):
            o = overround([gn.get("Oui"), gn.get("Non")])
            if o:
                rec["market_btts"].append(o)
        # Total de buts (exhaustif 0..max)
        tb = gm(xm, "Total de buts")
        tot_over = None
        if isinstance(tb, dict):
            ks = {k: v for k, v in tb.items() if k.isdigit() and isinstance(v, (int, float)) and v > 1}
            o = overround(list(ks.values()))
            if o:
                rec["market_total"].append(o)
            # P(Over3.5) via total exact = P(k>=4) dé-margé
            if ks:
                inv = {int(k): 1 / v for k, v in ks.items()}
                Z = sum(inv.values())
                tot_over = sum(p for k, p in inv.items() if k >= 4) / Z
        # Score exact
        se = gm(xm, "Score exact")
        if isinstance(se, dict):
            o = overround([v for v in se.values() if isinstance(v, (int, float))])
            if o:
                rec["market_score"].append(o)
        # cohérence croisée Over3.5
        if ou is not None and tot_over is not None:
            rec["xmarket_ou35_gap"].append(ou - tot_over)

    print("\n===== OVERROUND BACK-ALL PAR MARCHÉ (>1 = marge ; <1 = ARBITRAGE) =====")
    print(f"{'marché':<16}{'n':>8}{'médiane':>10}{'min':>10}{'p1%':>10}{'n_arb(<1)':>11}")
    for name, arr in [("1X2", rec["market_1x2"]), ("Over/Under 3.5", rec["market_ou35"]),
                      ("BTTS", rec["market_btts"]), ("Total de buts", rec["market_total"]),
                      ("Score exact", rec["market_score"])]:
        a = np.array(arr)
        if len(a) == 0:
            continue
        n_arb = int((a < 1.0).sum())
        print(f"{name:<16}{len(a):>8}{np.median(a):>10.4f}{a.min():>10.4f}"
              f"{np.percentile(a,1):>10.4f}{n_arb:>11}")

    g = np.array(rec["xmarket_ou35_gap"])
    print(f"\n===== COHÉRENCE CROISÉE P(Over3.5) : '+/-' vs 'Total de buts' =====")
    if len(g):
        print(f"  n={len(g)} | écart moyen={100*g.mean():+.3f}pp | médian={100*np.median(g):+.3f}pp "
              f"| std={100*g.std():.3f}pp | |écart|>5pp : {100*(np.abs(g)>0.05).mean():.1f}% des events")
        print(f"  -> si les 2 marchés étaient incohérents on verrait un écart systématique non nul + arbitrable.")
    print(f"\n===== ARBITRAGE PUR DÉTECTÉ : {arb_events} events (Over/Under back-all < 100%) =====")
    print("-> Si 0 : aucun profit sans risque ; le book est cohérent (marge partout).")


if __name__ == "__main__":
    main()
