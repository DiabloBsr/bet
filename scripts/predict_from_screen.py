"""Predict the 10 fixtures from a screenshot using the already-trained V2 model.

Pas de scraping live : on charge l'historique resolu de la base, on fit le
modele V2, puis on predit une liste de matchs codee en dur (cotes du screenshot).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v2 import fit_model_v2, predict_match_v2

# (home, away, odds_home, odds_draw, odds_away) — exactement le screenshot
FIXTURES = [
    ("N. Forest", "Fulham", 2.18, 3.44, 3.21),
    ("A. Villa", "London Reds", 3.48, 3.64, 2.01),
    ("Spurs", "London Blues", 1.91, 4.14, 3.39),
    ("Brighton", "Liverpool", 2.58, 3.96, 2.38),
    ("Everton", "Leeds", 1.68, 3.43, 5.82),
    ("West Ham", "Wolverhampton", 1.80, 3.97, 3.96),
    ("C. Palace", "Burnley", 1.61, 4.42, 4.73),
    ("Newcastle", "Manchester Blue", 2.28, 3.56, 2.93),
    ("Sunderland", "Manchester Red", 6.07, 4.06, 1.54),
    ("Bournemouth", "Brentford", 2.57, 3.20, 2.79),
]


def _pick(probs):
    return max(probs, key=lambda x: x[1])


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)
    history = pd.read_sql(
        """
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        JOIN results r ON r.event_id = e.id
        """,
        engine,
    )
    if history.empty:
        print("aucun match resolu pour calibrer")
        return 1

    model = fit_model_v2(history, half_life=None, score_market_weight=0.5)
    print(f"=== Modele V2 — n={model.n_train} | rho_DC={model.rho:+.4f} ===\n")

    rows = []
    for home, away, oh, od, oa in FIXTURES:
        pred = predict_match_v2(model, home, away, oh, od, oa, score_exact_market=None)

        # 1X2 cote calibree
        c = [("1", pred["p_h_cote"], oh), ("X", pred["p_d_cote"], od), ("2", pred["p_a_cote"], oa)]
        pc, p_c, odds_c = _pick(c)

        # 1X2 blend (=Poisson+DC ici, pas de marche score exact)
        if pred["p_h_blend"] is not None:
            b = [("1", pred["p_h_blend"], oh), ("X", pred["p_d_blend"], od), ("2", pred["p_a_blend"], oa)]
            pb, p_b, odds_b = _pick(b)
            edge_b = p_b * odds_b - 1
            n_a, n_b = pred["team_a_n"], pred["team_b_n"]
        else:
            pb, p_b, edge_b, n_a, n_b = "—", 0.0, 0.0, 0, 0

        score_pred = pred["score_blend"] or pred["score_pois"] or "—"
        top3 = pred.get("top3_blend") or pred.get("top3_pois") or []
        top3_str = " / ".join(s for s, _ in top3[:3]) if top3 else "—"
        accord = "OUI" if pb == pc and pb != "—" else " - "

        rows.append({
            "match": f"{home} vs {away}",
            "cotes": f"{oh:.2f}/{od:.2f}/{oa:.2f}",
            "pick_cote": pc, "p_cote%": f"{p_c*100:.0f}",
            "pick_blend": pb, "p_blend%": f"{p_b*100:.0f}" if pb != "—" else "—",
            "edge": f"{edge_b*100:+.1f}%" if pb != "—" else "—",
            "score": score_pred, "top3": top3_str,
            "n_hist": f"{n_a}/{n_b}", "accord": accord,
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print("\npick_blend = Poisson+Dixon-Coles (pas de marche Score exact ici)")
    print("edge = p_blend*cote - 1 (>0 = +EV theorique) | accord OUI = blend==cote")
    return 0


if __name__ == "__main__":
    sys.exit(main())
