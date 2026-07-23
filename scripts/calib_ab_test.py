"""La table de calibration aide-t-elle vraiment ? — arbitrage hors échantillon, par ligue.

Pourquoi cette question se pose. `refresh_calibration.py` ajuste la table comme
`empirique / modèle`, où **modèle** est la grille d'inversion Poisson du marché
(`exact_invert_1x2` + `apply_sim_deviations`). Mais `predict_one` l'applique au
**consensus V2+V5+marché**, qui est un autre objet. La table corrige donc une chose
et sert sur une autre : rien ne garantit qu'elle aide, et une mesure sur la seule
grille de marché montrait même une dégradation (CAN 2.40pp -> 3.73pp).

Protocole anti-mirage (cf. THEORIES_TESTED.md) :
  - découpage CHRONOLOGIQUE par ligue : 70% train / 30% test ;
  - la table est RÉAJUSTÉE sur le seul train (sinon on teste sur ses propres données) ;
  - les deux bras partagent exactement le même consensus, seule l'application de la
    table diffère -> comparaison APPARIÉE, et le biais des modèles d'équipes
    (ajustés sur tout l'historique) s'annule entre les bras ;
  - décision par test de McNemar sur les seules paires discordantes.

    python scripts/calib_ab_test.py [--limit 1500] [--leagues 8060,8035]

Sortie ASCII. Ne modifie rien : c'est un outil de décision, pas un correctif.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import predict_trio as pt                      # noqa: E402
from refresh_calibration import SQL, _calibrate  # noqa: E402
from scraper.config import load_settings        # noqa: E402

TRAIN_FRAC = 0.70
_DB = ""                                   # extrait fige eventuel (--db)
NOMS = {"InstantLeague-8060": "CAN", "InstantLeague-8035": "ANG",
        "InstantLeague-8036": "FRA", "InstantLeague-8037": "ESP",
        "InstantLeague-8042": "ITA", "InstantLeague-8043": "ALL",
        "InstantLeague-8044": "POR", "InstantLeague-8056": "UCL",
        "InstantLeague-8065": "CDM"}


def _lire(sql: str) -> pd.DataFrame:
    """Lecture SEULE + patience : le scraper ecrit dans la meme base sqlite, et
    rencontrer un verrou est normal (un run precedent est mort dessus)."""
    import sqlite3
    import time
    chemin = _DB or load_settings().db_url.split("///")[-1]
    for essai in range(6):
        try:
            c = sqlite3.connect(f"file:{chemin}?mode=ro", uri=True, timeout=180)
            c.execute("PRAGMA busy_timeout=180000")
            try:
                return pd.read_sql(sql, c)
            finally:
                c.close()
        except Exception as exc:
            if essai == 5:
                raise
            print(f"  base occupee ({str(exc)[:60]}…), nouvel essai dans 10s")
            time.sleep(10)
    return pd.DataFrame()


def _mcnemar(n01: int, n10: int) -> float:
    """p bilatéral, correction de continuité. n10 = calibré gagne, n01 = brut gagne."""
    n = n01 + n10
    if n == 0:
        return 1.0
    from math import erfc, sqrt
    chi = (abs(n10 - n01) - 1) ** 2 / n
    return erfc(sqrt(chi / 2))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1500, help="matchs de test par ligue")
    ap.add_argument("--leagues", type=str, default="8060,8035")
    ap.add_argument("--db", type=str, default="",
                    help="extrait fige (evite de se battre avec le scraper qui ecrit)")
    args = ap.parse_args()
    cibles = [f"InstantLeague-{x.strip()}" for x in args.leagues.split(",") if x.strip()]

    global _DB
    _DB = args.db
    if _DB:
        # lecture seule + longue patience : l extrait est fige, mais on garde le
        # meme contrat que sur la base vive.
        eng = create_engine(f"sqlite:///file:{_DB}?mode=ro&uri=true",
                            connect_args={"timeout": 180, "uri": True})
        print(f"base : extrait fige {_DB}")
    else:
        eng = create_engine(load_settings().db_url)
    print("lecture de l'historique cote…")
    df = _lire(SQL)                             # deja trie par expected_start
    print(f"  {len(df)} matchs, {df.lg.nunique()} ligues")

    print("ajustement des moteurs V2/V5…")
    m5, v2, nfit = pt.fit(eng)
    print(f"  fit sur {nfit} matchs")

    print(f"\n  {'ligue':<7}{'test':>7}{'Top-1 brut':>12}{'Top-1 calibre':>15}"
          f"{'ecart':>9}{'brut+':>7}{'cal+':>6}{'p':>9}   verdict")
    for lg in cibles:
        d = df[df.lg == lg].reset_index(drop=True)
        if len(d) < 4000:
            print(f"  {NOMS.get(lg, lg):<7} {len(d)} matchs -> trop peu")
            continue
        coupe = int(len(d) * TRAIN_FRAC)
        train, test = d.iloc[:coupe], d.iloc[coupe:].tail(args.limit)

        corr, _emp, _mod, ninv = _calibrate(train)     # table AJUSTEE SUR LE TRAIN SEUL
        pt._CALIB_BY_LG = {lg: corr}                   # on force la production a l'utiliser
        pt._CALIB = corr

        # les cotes du test doivent etre rejointes a leurs marches + resultats
        ids = _lire(f"""
            SELECT e.id ev, e.team_a, e.team_b, o.odds_home oh, o.odds_draw od,
                   o.odds_away oa, o.extra_markets, r.score_a sa, r.score_b sb
            FROM events e
            JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f
              ON f.event_id = e.id
            JOIN odds_snapshots o ON o.id = f.mid
            JOIN results r ON r.event_id = e.id
            WHERE e.competition = '{lg}' AND r.score_a IS NOT NULL AND o.odds_home > 1
            ORDER BY e.expected_start DESC LIMIT {len(test)}""")

        brut = cal = n = n01 = n10 = 0
        for r in ids.itertuples():
            try:
                p = pt.predict_one(eng, m5, v2, r.team_a, r.team_b,
                                   r.oh, r.od, r.oa, r.extra_markets, lg)
            except Exception:
                continue
            ctop, tcal = p.get("consensus_top3"), p.get("top1_calibre")
            if not ctop or not tcal:
                continue
            reel = f"{int(r.sa)}-{int(r.sb)}"
            hb, hc = int(ctop[0][0] == reel), int(tcal[0] == reel)
            brut += hb; cal += hc; n += 1
            n01 += (hb and not hc); n10 += (hc and not hb)
        if not n:
            print(f"  {NOMS.get(lg, lg):<7} aucun match evaluable")
            continue

        pb, pc = brut / n, cal / n
        p = _mcnemar(n01, n10)
        if p >= 0.05:
            verdict = "EQUIVALENT (garder le plus simple : sans table)"
        else:
            verdict = "la table AIDE" if pc > pb else "la table NUIT -> desactiver"
        print(f"  {NOMS.get(lg, lg):<7}{n:>7}{100*pb:>11.2f}%{100*pc:>14.2f}%"
              f"{100*(pc-pb):>+8.2f}pp{n01:>7}{n10:>6}{p:>9.3f}   {verdict}")

    print("\n  brut+ = cas ou seul le brut trouve le score ; cal+ = seul le calibre.")
    print("  Seules ces paires discordantes portent l'information (McNemar).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
