"""Contrôle de cohérence entre marchés — détecteur d'erreur de cotation.

Principe : le marché « Score exact » compte **28 issues** — exactement tous les
scores de total <= 6, c'est-à-dire TOUTE la surface des résultats possibles
(0 dépassement sur 58 083 matchs). C'est donc une partition complète : la déviger
donne la loi jointe (buts domicile, buts extérieur) telle que le book la price.

Tous les autres marchés s'en déduisent exactement. On compare donc chaque offre
à son prix équitable calculé depuis cette référence :

    1X2 · Double Chance · G/NG · Pair/Impair · Total de buts · +/- · Multi-Buts
    Total equipe domicile · Total equipe extérieur · 1X2 & Total · 1X2 & G/NG

Deux usages :
  - **marge par marché** : elle doit être stable. Sa dérive signale un changement
    de pricing (nouvelle version du moteur, marché mal recopié).
  - **offres à EV positive** : `cote * p_equitable > 1`. Historiquement AUCUNE.
    Toute occurrence est à vérifier à la main — c'est presque sûrement un bug de
    lecture de libellé, pas une opportunité (cf. market_ranges.check_roi).

LIMITE MESURÉE — les niveaux absolus sont surestimés d'environ 2 à 3 points.
Les cases de queue du « Score exact » sont plafonnées à la cote 100, et sur
40 000 événements il n'en existe **aucun** sans au moins une case plafonnée. Leur
1/cote plancher (1%) capte de la masse à la renormalisation, ce qui rabote les
cases principales et gonfle tous les ratios. Contrôle : l'overround 1X2 mesuré en
direct (somme des 1/cote) vaut **6.00%**, quand la référence en annonce 8.47%.
Cet outil sert donc à détecter une DÉRIVE (même statistique comparée dans le
temps), pas à chiffrer une marge dans l'absolu.

BASELINE au 2026-07-22 (40 000 événements, marge médiane) :
    Double Chance 12.1% · G/NG 7.9% · Pair/Impair 10.3% · Total de buts 17.9%
    +/- 11.7% · Multi-Buts 15.8% · Total dom. 10.8% · Total ext. 7.1%
    1X2 & Total 15.9% · 1X2 & G/NG 17.0% · 1X2 10.6% — 0 offre a EV positive.

    python scripts/cross_market_check.py [--limit 40000] [--json rapport.json]

Sortie ASCII, code retour 1 si anomalie (utilisable en tâche planifiée / CI).
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
CAP_ODDS = 90.0   # au-delà, la cote touche le plafond du site (100) : 1/cote devient un
                  # PLANCHER de probabilité, pas une estimation. Comparer deux valeurs
                  # saturées produit de faux "+EV" (constaté : p figée à 1.77%).
MAX_TOTAL = 6     # plafond dur du RNG


def _norm(s: str) -> str:
    return str(s).replace("\x82", "e").replace("é", "e").replace("è", "e").lower()


def _market(extra: dict, nom: str):
    """Le marché nommé EXACTEMENT `nom` (aux accents près), sinon None.

    Surtout pas de correspondance par préfixe : « 1X2 » attrapait « 1X2 & Total »,
    et l'on comparait alors un marché à un autre. Un nom inconnu doit être ignoré,
    jamais deviné.
    """
    cible = _norm(nom)
    for k, v in (extra or {}).items():
        if _norm(k) == cible:
            return v
    return None


# --- prédicats : (buts_domicile, buts_exterieur) -> l'offre gagne-t-elle ? --------
# Écrits une fois, testés (tests/test_cross_market.py). Toute erreur ici fabrique un
# faux edge : c'est exactement ce qui s'est produit avec "supérieur à 4".

def _dc(lbl):
    l = lbl.strip()
    return {"1X": lambda h, a: h >= a, "12": lambda h, a: h != a,
            "X2": lambda h, a: h <= a}.get(l)


def _gng(lbl):
    l = _norm(lbl).strip()
    if l.startswith("oui"):
        return lambda h, a: h > 0 and a > 0
    if l.startswith("non"):
        return lambda h, a: h == 0 or a == 0
    return None


def _parite(lbl):
    l = _norm(lbl).strip()
    if l.startswith("pair"):
        return lambda h, a: (h + a) % 2 == 0
    if l.startswith("impair"):
        return lambda h, a: (h + a) % 2 == 1
    return None


def _total_equipe(lbl, cote_domicile: bool):
    rg = parse_goal_range(lbl)
    if not rg:
        return None
    return (lambda h, a: h in rg) if cote_domicile else (lambda h, a: a in rg)


def _total_global(lbl):
    rg = parse_goal_range(lbl)
    return (lambda h, a: (h + a) in rg) if rg else None


def _cell_total(lbl):
    """Marché « Total de buts » : les clés sont des entiers nus ('0'..'6')."""
    s = str(lbl).strip()
    if not s.isdigit():
        return None
    n = int(s)
    return lambda h, a: (h + a) == n


def _x12_total(lbl):
    """'1 / > 3.5' — issue 1X2 ET plage de buts."""
    if "/" not in lbl:
        return None
    issue, plage = lbl.split("/", 1)
    issue = issue.strip()
    rg = parse_goal_range(plage)
    if not rg or issue not in ("1", "X", "2"):
        return None
    res = {"1": lambda h, a: h > a, "X": lambda h, a: h == a, "2": lambda h, a: h < a}[issue]
    return lambda h, a: res(h, a) and (h + a) in rg


def _x12_gng(lbl):
    l = _norm(lbl)
    deux = "les deux equipes marquent" in l
    if l.startswith("1 gagne"):
        return (lambda h, a: h > a and a > 0) if deux else (lambda h, a: h > a and a == 0)
    if l.startswith("2 gagne"):
        return (lambda h, a: h < a and h > 0) if deux else (lambda h, a: h < a and h == 0)
    if l.startswith("x et"):
        if "aucun but" in l:
            return lambda h, a: h == 0 and a == 0
        return (lambda h, a: h == a and h > 0) if deux else None
    return None


def _x12(lbl):
    return {"1": lambda h, a: h > a, "X": lambda h, a: h == a,
            "2": lambda h, a: h < a}.get(str(lbl).strip())


# (nom affiché, nom EXACT en base, fabricant de prédicat)
# Le 1X2 sec ne figure pas dans extra_markets : il vit dans les colonnes de cotes
# de la table odds_snapshots, il est donc traité à part.
MARCHES = [
    ("Double Chance",    "Double Chance",           _dc),
    ("G/NG",             "G/NG",                    _gng),
    ("Pair/Impair",      "Pair/Impair",             _parite),
    ("Total de buts",    "Total de buts",           _cell_total),
    ("+/-",              "+/-",                     _total_global),
    ("Multi-Buts",       "Multi-Buts",              _total_global),
    ("Total dom.",       "Total equipe domicile",   lambda l: _total_equipe(l, True)),
    ("Total ext.",       "Total equipe extérieur",  lambda l: _total_equipe(l, False)),
    ("1X2 & Total",      "1X2 & Total",             _x12_total),
    ("1X2 & G/NG",       "1X2 & G/NG",              _x12_gng),
]

SCORES = [(h, a) for h in range(MAX_TOTAL + 1) for a in range(MAX_TOTAL + 1)
          if h + a <= MAX_TOTAL]          # les 28 issues


def _reference(extra: dict):
    """Loi jointe DÉVIGÉE depuis « Score exact » (28 issues) + cellules saturées.

    Une cellule au plafond (cote >= 90) n'a pas de probabilité lisible : son 1/cote
    est un PLANCHER. En CAN, une quinzaine de cases de queue sont plafonnées et leur
    faux 1% chacune s'accumule en ~13.8% de masse fantôme — pile la zone couverte par
    « > 3.5 ». Une offre qui s'appuie dessus est donc incomparable, pas rentable.
    """
    se = _market(extra, "Score exact")
    if not isinstance(se, dict) or len(se) < len(SCORES):
        return None
    inv, sature = {}, set()
    for h, a in SCORES:
        o = se.get(f"{h}-{a}")
        if not isinstance(o, (int, float)) or o <= 1:
            return None                    # partition incomplète -> non dévigeable
        inv[(h, a)] = 1.0 / o
        if o >= CAP_ODDS:
            sature.add((h, a))
    s = sum(inv.values())
    return {k: v / s for k, v in inv.items()}, sature


def _read(limit: int):
    """Lecture seule + patience : le scraper écrit en parallèle."""
    for attempt in range(5):
        try:
            c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=120)
            c.execute("PRAGMA busy_timeout=120000")
            return c.execute(
                "SELECT e.competition, e.id, o.extra_markets, o.odds_home, "
                "o.odds_draw, o.odds_away FROM events e "
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
    ap.add_argument("--json", type=str, default="", help="ecrire le rapport en JSON")
    ap.add_argument("--marge-max", type=float, default=0.35,
                    help="facteur de marge median tolere (1.35 = 35%%)")
    ap.add_argument("--dispersion-max", type=float, default=0.5,
                    help="etendue p1-p99 toleree du facteur de marge")
    args = ap.parse_args()

    rows = _read(args.limit)
    print(f"{len(rows)} evenements lus")

    ratios = defaultdict(list)      # marche -> (1/cote) / p_equitable
    suspects, illisibles, refs, ignorees_sat = [], defaultdict(int), 0, 0
    for lg, eid, xm, oh, od, oa in rows:
        try:
            extra = json.loads(xm) if isinstance(xm, str) else (xm or {})
        except Exception:
            continue
        ref = _reference(extra)
        if not ref:
            continue
        loi, sature = ref
        refs += 1
        offres = [(nom, m, fab) for nom, cle, fab in MARCHES
                  if isinstance((m := _market(extra, cle)), dict)]
        if oh and od and oa:      # le 1X2 sec vient des colonnes, pas d'extra_markets
            offres.append(("1X2", {"1": oh, "X": od, "2": oa}, _x12))
        for nom, m, fabrique in offres:
            for label, o in m.items():
                if not isinstance(o, (int, float)) or not 1 < o < CAP_ODDS:
                    continue         # offre au plafond : 1/cote n'est plus une probabilite
                pred = fabrique(label)
                if pred is None:
                    illisibles[f"{nom} :: {label}"] += 1
                    continue
                gagnantes = [(h, a) for h, a in loi if pred(h, a)]
                p = sum(loi[c] for c in gagnantes)
                if p <= 0.005:
                    continue
                # une offre dont la zone gagnante repose sur des cases plafonnees
                # n'est pas comparable : leur 1/cote est un plancher, pas une proba.
                p_sat = sum(loi[c] for c in gagnantes if c in sature)
                if p_sat > 0.05 * p:
                    ignorees_sat += 1
                    continue
                ratios[nom].append((1.0 / o) / p)
                if o * p > 1.0:
                    suspects.append((lg, eid, nom, label, o, p, o * p))

    if not ratios:
        print("aucune reference 'Score exact' complete -> rien a verifier")
        return 0

    anomalie = False
    print(f"  {refs} references 'Score exact' completes (28 issues, devigees)\n")
    print(f"  {'marche':<16}{'offres':>9}{'marge mediane':>16}{'p1':>8}{'p99':>8}{'verdict':>10}")
    rapport = {}
    for nom in list(dict.fromkeys([m[0] for m in MARCHES] + ["1X2"])):
        vals = ratios.get(nom)
        if not vals:
            continue
        a = np.asarray(vals)
        med, p1, p99 = float(np.median(a)), float(np.percentile(a, 1)), float(np.percentile(a, 99))
        ok = 1.0 <= med <= 1.0 + args.marge_max and (p99 - p1) <= args.dispersion_max
        anomalie |= not ok
        rapport[nom] = {"n": int(a.size), "marge": round(100 * (med - 1), 3),
                        "p1": round(p1, 4), "p99": round(p99, 4), "ok": ok}
        print(f"  {nom:<16}{a.size:>9}{100*(med-1):>14.2f}%{p1:>8.3f}{p99:>8.3f}"
              f"{'OK' if ok else 'DERIVE':>10}")

    if illisibles:
        print(f"\n  {len(illisibles)} libelle(s) non reconnu(s) -> IGNORES (jamais devines) :")
        for k, v in sorted(illisibles.items(), key=lambda x: -x[1])[:8]:
            print(f"    x{v:<7} {k}")

    if suspects:
        anomalie = True
        print(f"\n  {len(suspects)} offre(s) a EV positive contre 'Score exact' :")
        for lg, eid, nom, label, o, p, ev in sorted(suspects, key=lambda x: -x[6])[:10]:
            print(f"    {lg} #{eid} [{nom}] {label[:34]:<36} cote {o:6.2f}"
                  f" | p_equitable {100*p:5.2f}% | paie {ev:.3f}x")
        print("    -> VERIFIER A LA MAIN avant toute mise : presque surement un bug de lecture.")
    else:
        print("\n  aucune offre a EV positive : marches coherents entre eux.")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"evenements": len(rows), "references": refs, "marches": rapport,
             "suspects": len(suspects), "illisibles": dict(illisibles)},
            indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n  rapport -> {args.json}")

    return 1 if anomalie else 0


if __name__ == "__main__":
    raise SystemExit(main())
