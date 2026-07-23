"""Bulletin de notes du prédicteur corrigé — par domaine et par ligue.

Le tracker live a cessé d'écrire le 2026-07-06, bien avant les corrections du
2026-07-22 : il n'existe donc AUCUNE prédiction enregistrée postérieure aux
corrections. Pour juger le moteur tel qu'il tourne AUJOURD'HUI, on le rejoue sur
des matchs récents dont on connaît déjà le score réel, et on note chaque domaine.

Domaines mesurés, chacun comparé à sa borne théorique (Bayes, cf. THEORIES_TESTED) :
  - Score exact Top-1 (brut et calibré) — plafond 11.9 %
  - Score exact Top-3                    — plafond 31.6 %
  - Résultat 1X2                          — plafond 55 %  + lift vs "toujours le favori"
  - Over/Under 2.5 buts                   — précision directionnelle + Brier (calibrage)

HONNÊTETÉ MÉTHODO. Les modèles d'équipes V2/V5 sont ajustés sur tout l'historique,
matchs de test inclus : léger optimisme in-sample sur la composante ÉQUIPE. En
revanche l'arbitre MARCHÉ, le 1X2 et l'Over/Under dérivent des cotes (rien n'est
ajusté) : ces trois-là sont propres. Le biais joue donc en faveur du Score-exact,
pas des marchés cote-dérivés — on en tient compte dans la lecture.

    python scripts/eval_predictor.py [--db data/calib_ab_extract.db] [--n 1500]
                                     [--leagues 8060,8035]
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

import predict_trio as pt                    # noqa: E402
from scraper.config import load_settings     # noqa: E402

NOMS = {"InstantLeague-8060": "CAN", "InstantLeague-8035": "ANG",
        "InstantLeague-8036": "FRA", "InstantLeague-8037": "ESP",
        "InstantLeague-8042": "ITA", "InstantLeague-8043": "ALL",
        "InstantLeague-8044": "POR", "InstantLeague-8056": "UCL",
        "InstantLeague-8065": "CDM"}
# PAS de plafond fixe : mesurer une ligue avec le plafond d'une AUTRE dé-calibre
# (bug attrapé : CAN 27% jugée contre le plafond anglais 11.9% => faux "+124%",
# alors que le plafond PROPRE de la CAN est ~24.7% car elle marque peu). Chaque
# repère est donc recalculé pour la ligue évaluée, à partir de ses propres cotes.


def _ic(p: float, n: int) -> float:
    return 1.96 * np.sqrt(p * (1 - p) / n) if n else 0.0


def _norm(x: dict, prefix: str):
    for k, v in (x or {}).items():
        if k.replace("é", "e").startswith(prefix):
            return v
    return None


def _market_ceiling(extra) -> float | None:
    """Plafond de Bayes local : proba dévigée du score le plus probable selon le marché."""
    import json
    try:
        mk = json.loads(extra) if isinstance(extra, str) else (extra or {})
    except Exception:
        return None
    se = _norm(mk, "Score exact")
    if not isinstance(se, dict):
        return None
    inv = [1 / v for v in se.values() if isinstance(v, (int, float)) and v > 1]
    s = sum(inv)
    return max(inv) / s if s > 0 else None


def _res(sa: int, sb: int) -> str:
    return "1" if sa > sb else ("2" if sb > sa else "X")


def _eval_league(eng, m5, v2, lg: str, n: int) -> dict | None:
    ids = pd.read_sql(f"""
        SELECT e.team_a, e.team_b, o.odds_home oh, o.odds_draw od, o.odds_away oa,
               o.extra_markets, r.score_a sa, r.score_b sb
        FROM events e
        JOIN (SELECT event_id, MIN(id) mid FROM odds_snapshots GROUP BY event_id) f
          ON f.event_id = e.id
        JOIN odds_snapshots o ON o.id = f.mid
        JOIN results r ON r.event_id = e.id
        WHERE e.competition = '{lg}' AND r.score_a IS NOT NULL AND o.odds_home > 1
        ORDER BY e.expected_start DESC LIMIT {n}""", eng)

    acc = {k: [] for k in ("h1", "h1c", "h3", "hx", "hx_fav", "ou_dir")}
    brier_x = []
    brier_ou = []
    ceilings = []                 # plafond de Bayes local, par match
    actual_scores = []            # pour la baseline "toujours le score modal"
    under_flags = []              # pour la baseline O/U (classe majoritaire)
    for r in ids.itertuples():
        try:
            p = pt.predict_one(eng, m5, v2, r.team_a, r.team_b,
                               r.oh, r.od, r.oa, r.extra_markets, lg)
        except Exception:
            continue
        ctop = p.get("consensus_top3") or []
        tcal = p.get("top1_calibre")
        if not ctop:
            continue
        reel = f"{int(r.sa)}-{int(r.sb)}"
        actual_scores.append(reel)
        under_flags.append(int((r.sa + r.sb) <= 2))
        cl = _market_ceiling(r.extra_markets)
        if cl is not None:
            ceilings.append(cl)
        acc["h1"].append(int(ctop[0][0] == reel))
        if tcal:
            acc["h1c"].append(int(tcal[0] == reel))
        acc["h3"].append(int(any(s == reel for s, _ in ctop[:3])))

        # 1X2 : le modele choisit l'issue la plus probable
        ph, pdr, pa = p.get("x12", [None, None, None])
        rr = _res(int(r.sa), int(r.sb))
        if ph is not None:
            pick = ["1", "X", "2"][int(np.argmax([ph, pdr, pa]))]
            acc["hx"].append(int(pick == rr))
            fav = ["1", "X", "2"][int(np.argmin([r.oh, r.od, r.oa]))]
            acc["hx_fav"].append(int(fav == rr))
            pgt = {"1": ph, "X": pdr, "2": pa}[rr]
            brier_x.append(sum((({"1": ph, "X": pdr, "2": pa}[k]) - (k == rr)) ** 2
                               for k in ("1", "X", "2")))

        # Over/Under 2.5
        ov = p.get("over25_pct")
        if ov is not None:
            povr = ov / 100.0
            over_reel = int((r.sa + r.sb) >= 3)
            acc["ou_dir"].append(int((povr >= 0.5) == bool(over_reel)))
            brier_ou.append((povr - over_reel) ** 2)

    if not acc["h1"]:
        return None
    from collections import Counter
    out = {"lg": lg, "n": len(acc["h1"])}
    for k in acc:
        v = acc[k]
        out[k] = (np.mean(v), len(v)) if v else (None, 0)
    out["brier_x"] = float(np.mean(brier_x)) if brier_x else None
    out["brier_ou"] = float(np.mean(brier_ou)) if brier_ou else None
    # repères PROPRES à la ligue
    out["ceiling"] = float(np.mean(ceilings)) if ceilings else None
    modal_freq = Counter(actual_scores).most_common(1)[0][1] / len(actual_scores)
    out["base_modal"] = modal_freq                         # toujours le score le + fréquent
    ur = float(np.mean(under_flags))
    out["base_ou"] = max(ur, 1 - ur)                       # classe majoritaire O/U
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="data/calib_ab_extract.db")
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--leagues", type=str, default="8060,8035")
    args = ap.parse_args()
    cibles = [f"InstantLeague-{x.strip()}" for x in args.leagues.split(",") if x.strip()]

    dbp = Path(args.db)
    if dbp.exists():
        eng = create_engine(f"sqlite:///file:{dbp.as_posix()}?mode=ro&uri=true",
                            connect_args={"timeout": 180, "uri": True})
        print(f"base : extrait fige {dbp}  ({dbp.stat().st_size/1e6:.0f} Mo)")
    else:
        eng = create_engine(load_settings().db_url)
        print("base : live")

    print("ajustement V2/V5…")
    m5, v2, nfit = pt.fit(eng)
    print(f"  fit sur {nfit} matchs\n")

    res = [r for lg in cibles if (r := _eval_league(eng, m5, v2, lg, args.n))]

    print("=" * 82)
    print("REUSSITE PAR DOMAINE  (modele vs REPERE PROPRE A LA LIGUE — lift = vraie valeur)")
    print("=" * 82)
    excel = {}
    for r in res:
        nom = NOMS.get(r["lg"], r["lg"])
        print(f"\n### {nom}  (n={r['n']}, plafond de Bayes local {100*(r['ceiling'] or 0):.1f}%)")
        lifts = []

        # Score exact : repere = toujours le score modal (0-0 en CAN, 1-1 en ANG)
        (h1c, n1c) = r["h1c"]
        if h1c is not None:
            base = r["base_modal"]
            lift = h1c - base
            sig = _ic(h1c, n1c)
            flag = "  <<< bat la baseline" if lift - sig > 0 else ""
            lifts.append(("Score exact Top-1", lift))
            print(f"  {'Score exact Top-1 (cal.)':<26}{100*h1c:5.1f}% (+-{100*sig:.1f})"
                  f"  vs toujours-modal {100*base:4.1f}%  {100*lift:+5.1f}pp{flag}")
        (h3, n3) = r["h3"]
        if h3 is not None:
            print(f"  {'Score exact Top-3':<26}{100*h3:5.1f}% (+-{100*_ic(h3,n3):.1f})"
                  f"  (info : 3 scores couverts)")

        # 1X2 : repere = toujours le favori (cote la plus basse)
        (hx, nx) = r["hx"]; (hf, _nf) = r["hx_fav"]
        if hx is not None and hf is not None:
            lift = hx - hf
            sig = _ic(hx, nx)
            flag = "  <<< bat le favori" if lift - sig > 0 else "  = le favori (aucun edge)"
            lifts.append(("Resultat 1X2", lift))
            print(f"  {'Resultat 1X2':<26}{100*hx:5.1f}% (+-{100*sig:.1f})"
                  f"  vs favori-cote {100*hf:4.1f}%  {100*lift:+5.1f}pp{flag}"
                  f"  Brier {r['brier_x']:.3f}")

        # Over/Under 2.5 : repere = classe majoritaire (under en CAN, over en ANG)
        (od, nod) = r["ou_dir"]
        if od is not None:
            base = r["base_ou"]
            lift = od - base
            sig = _ic(od, nod)
            flag = "  <<< bat la classe majoritaire" if lift - sig > 0 else "  = taux de base"
            lifts.append(("Over/Under 2.5", lift))
            print(f"  {'Over/Under 2.5 (sens)':<26}{100*od:5.1f}% (+-{100*sig:.1f})"
                  f"  vs majorite {100*base:4.1f}%  {100*lift:+5.1f}pp{flag}"
                  f"  Brier {r['brier_ou']:.3f}")

        excel[nom] = sorted(lifts, key=lambda x: -x[1])

    print("\n" + "=" * 82)
    print("OU LE PREDICTEUR EXCELLE — lift sur un repere BETE, par ligue")
    print("=" * 82)
    for nom, lifts in excel.items():
        if not lifts:
            continue
        best = lifts[0]
        verdict = (f"{best[0]} ({100*best[1]:+.1f}pp sur la baseline)"
                   if best[1] > 0 else "AUCUN domaine ne bat son repere bete")
        print(f"  {nom} : {verdict}")
    print("\n  Rappel : battre la baseline en TAUX ne veut pas dire +EV — la cote"
          " integre deja\n  ce taux (Under 2.5 CAN ~77% se paie ~1.25 -> 0.96 < 1).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
