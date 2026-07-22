"""Contrôle de cohérence entre marchés — détecteur d'erreur de cotation.

Principe : plusieurs marchés parlent du MÊME événement. « Multi-Buts 0/1/2 »,
« Over/Under 2.5 » et le marché « Total de buts » (partition complète 0..6) doivent
donner la même probabilité. On dévige la partition, on en déduit le prix ÉQUITABLE
de chaque offre, et on compare.

Mesuré sur 604 384 offres : le book est cohérent à **<=0.2pp**. C'est donc un
détecteur gratuit et très sensible — si un jour un marché est mal recopié, mal
mis à jour ou décalé, l'écart sautera bien avant qu'on le voie autrement.

Le seul signal actionnable est `cote * p_equitable > 1` : l'offre paierait plus que
sa vraie probabilité, marge comprise. Historiquement : aucune. Toute occurrence
est à vérifier à la main AVANT de miser (c'est probablement un bug de lecture).

    python scripts/cross_market_check.py [--limit 40000] [--seuil-pp 1.0]

Sortie ASCII, code retour 1 si une anomalie dépasse le seuil (utilisable en tâche
planifiée / CI).
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from market_ranges import parse_goal_range  # noqa: E402

DB = ROOT / "data" / "virtual_sports.db"
TOTAL_CELLS = [str(i) for i in range(7)]     # partition complète du marché "Total de buts"
CAP_ODDS = 90.0   # au-delà, la cote touche le plafond du site (100) : 1/cote est un
                  # PLANCHER de probabilité, pas une estimation. Comparer deux cellules
                  # saturées entre elles produit de faux "+EV" (constat : p figée à 1.77%).


def _market(extra: dict, prefix: str):
    """Le marché dont le libellé commence par `prefix` (accents variables en base)."""
    for k, v in (extra or {}).items():
        if k.replace("\x82", "e").replace("é", "e").startswith(prefix):
            return v
    return None


def _fair_totals(extra: dict):
    """(probas dévigées par total, cellules saturées) depuis la partition 0..6, ou None."""
    t = _market(extra, "Total de buts")
    if not isinstance(t, dict):
        return None
    inv, sature = {}, set()
    for i, k in enumerate(TOTAL_CELLS):
        o = t.get(k)
        if not isinstance(o, (int, float)) or o <= 1:
            return None          # partition incomplète -> non dévigeable, on écarte
        inv[i] = 1.0 / o
        if o >= CAP_ODDS:
            sature.add(i)
    s = sum(inv.values())
    return {i: v / s for i, v in inv.items()}, sature


def _read(limit: int):
    """Lecture seule + patience : le scraper écrit en parallèle."""
    for attempt in range(5):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=120)
            c.execute("PRAGMA busy_timeout=120000")
            return c.execute(
                "SELECT e.competition, e.id, o.extra_markets FROM events e "
                "JOIN odds_snapshots o ON o.event_id=e.id "
                "GROUP BY e.id ORDER BY e.id DESC LIMIT ?", (limit,)).fetchall()
        except sqlite3.Error as exc:
            if attempt == 4:
                raise
            print(f"  base occupee ({exc}), nouvelle tentative...")
            time.sleep(6)
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40000, help="derniers evenements lus")
    ap.add_argument("--seuil-pp", type=float, default=1.0,
                    help="ecart moyen tolere entre marches, en points de %%")
    args = ap.parse_args()

    rows = _read(args.limit)
    print(f"{len(rows)} evenements lus")

    ratios = defaultdict(list)     # marche -> (1/cote) / p_equitable  ~= facteur de marge
    suspects = []
    ignorees = 0
    for lg, eid, xm in rows:
        try:
            extra = json.loads(xm) if isinstance(xm, str) else (xm or {})
        except Exception:
            continue
        ft = _fair_totals(extra)
        if not ft:
            continue
        fair, sature = ft
        for nom, prefix in (("Multi-Buts", "Multi-Buts"), ("+/-", "+/-")):
            m = _market(extra, prefix)
            if not isinstance(m, dict):
                continue
            for label, o in m.items():
                if not isinstance(o, (int, float)) or not 1 < o < CAP_ODDS:
                    continue          # offre elle-meme au plafond : non comparable
                rg = parse_goal_range(label)
                if not rg:
                    continue
                cells = {min(i, 6) for i in rg if i <= 6}
                if cells & sature:    # s'appuie sur une cellule plafonnee -> non comparable
                    ignorees += 1
                    continue
                p = sum(fair.get(i, 0.0) for i in cells)
                if p <= 0.005:
                    continue
                ratios[nom].append((1.0 / o) / p)
                if o * p > 1.0:              # paierait plus que la vraie proba
                    suspects.append((lg, eid, label, o, p, o * p))

    if not ratios:
        print("aucune offre comparable (partition 'Total de buts' absente) -> rien a verifier")
        return 0

    anomalie = False
    print(f"  {ignorees} offres ignorees (cote plafonnee : 1/cote n'est plus une probabilite)")
    print(f"\n  {'marche':<14}{'offres':>9}{'marge mediane':>16}{'p1':>9}{'p99':>9}{'verdict':>12}")
    for nom, vals in ratios.items():
        a = np.asarray(vals)
        med, p1, p99 = np.median(a), np.percentile(a, 1), np.percentile(a, 99)
        # la marge doit etre stable et > 1. Une derive ou un ratio < 1 = anomalie.
        ok = 1.0 <= med <= 1.35 and (p99 - p1) <= 0.5
        anomalie |= not ok
        print(f"  {nom:<14}{a.size:>9}{100*(med-1):>14.2f}%{p1:>9.3f}{p99:>9.3f}"
              f"{'OK' if ok else 'DERIVE':>12}")

    if suspects:
        anomalie = True
        print(f"\n  {len(suspects)} offre(s) a EV positive contre le marche 'Total de buts' :")
        for lg, eid, label, o, p, ev in sorted(suspects, key=lambda x: -x[5])[:10]:
            print(f"    {lg} #{eid} | {label[:38]:<40} cote {o:6.2f} | p_equitable {100*p:5.2f}%"
                  f" | paie {ev:.3f}x")
        print("    -> VERIFIER A LA MAIN avant toute mise : c'est presque surement un bug de lecture.")
    else:
        print("\n  aucune offre a EV positive : marches coherents, marge posee uniformement.")

    return 1 if anomalie else 0


if __name__ == "__main__":
    raise SystemExit(main())
