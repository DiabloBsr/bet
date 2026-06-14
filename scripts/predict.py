"""Predict upcoming matches with the V3 model.

V3 ameliorations vs V2 :
  - Multi-market IPF (Score exact + Total de buts + G/NG contraintes)
    -> Top-1 blend 10.6% -> 12.0%   |  Top-3 market 32.9% -> 34.5%
  - X-aware pick rule (bascule X si p_X >= 27% et |p_h - p_a| < 10%)
    -> ROI global -3.2% -> -0.8%    |  picks X enfin non-nul

Backtest V3 (n=426 test) :
  - 1X2 global         : 56.3% acc, ROI -0.8%
  - 1X2 p>=70%         : 87.2% acc, ROI +5.0%   (vs 85.7% V2)
  - Score Top-1 market : 12.7%                    (plateau)
  - Score Top-3 market : 34.5%                    (vs 32.9% V2, le plus gros gain)
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v3 import fit_model_v3, predict_match_v3

MG_TZ = timezone(timedelta(hours=3))


def _pick(probs):
    return max(probs, key=lambda x: x[1])


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    history = pd.read_sql(
        """
        SELECT e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
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

    # form_alpha=0 = pas de form integration (test V3 montre que ca n'aide pas)
    model = fit_model_v3(history, engine=engine, form_alpha=0.0)
    print(f"=== Modele V3 — n={model.n_train} ===")
    print(f"  Poisson H/A split | rho_DC={model.rho:+.4f}")
    print(f"  Multi-market IPF (Score exact + Total de buts + G/NG)")
    print(f"  X-aware rule : p_X >= 27% et |p_h - p_a| < 10%")

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql(
        """
        SELECT e.team_a, e.team_b, e.round_info, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL
        ORDER BY e.expected_start, e.team_a
        """,
        engine,
    )
    if upcoming.empty:
        print("aucun match a venir")
        return 1
    upcoming["expected_start"] = pd.to_datetime(upcoming["expected_start"], utc=True, errors="coerce")
    window_start = now_utc - timedelta(minutes=5)
    upcoming = upcoming[
        upcoming["expected_start"].notna()
        & (upcoming["expected_start"] >= window_start)
    ].copy()
    if upcoming.empty:
        print("Aucun match dans la fenetre [now -5min ; +inf].")
        return 1

    rows = []
    for _, m in upcoming.iterrows():
        pred = predict_match_v3(
            model, m.team_a, m.team_b,
            m.odds_home, m.odds_draw, m.odds_away,
            extra_markets=m.extra_markets,
        )

        if pd.notna(m.expected_start):
            local_time = m.expected_start.tz_convert(MG_TZ).strftime("%H:%M:%S")
            delta_s = (m.expected_start - now_utc).total_seconds()
            statut = f"dans {int(delta_s)}s" if delta_s > 0 else (
                f"en cours ({int(-delta_s)}s)" if delta_s > -180 else "passe"
            )
        else:
            local_time = "—"; statut = "—"

        if pred["p_h_blend"] is not None:
            # Argmax pick
            blend = [("1", pred["p_h_blend"], m.odds_home),
                     ("X", pred["p_d_blend"], m.odds_draw),
                     ("2", pred["p_a_blend"], m.odds_away)]
            pb, p_b, odds_b = _pick(blend)
            edge_b = p_b * odds_b - 1
            # X-aware pick
            px = pred["pick_xaware"]
            score_pred = pred["score_market"] or pred["score_blend"] or "—"
            top3 = pred.get("top3_market") or pred.get("top3_blend") or []
            top3_str = " / ".join(s for s, _ in top3[:3]) if top3 else "—"
        else:
            pb, p_b, edge_b, px = "—", 0.0, 0.0, "—"
            score_pred = "—"
            top3_str = "—"

        c = [("1", pred["p_h_cote"], m.odds_home),
             ("X", pred["p_d_cote"], m.odds_draw),
             ("2", pred["p_a_cote"], m.odds_away)]
        pc, p_c, odds_c = _pick(c)

        accord = "OUI" if pb == pc and pb != "—" else " - "
        x_flag = "X!" if px == "X" and pb != "X" else ""

        rows.append({
            "heure": local_time,
            "statut": statut,
            "round": int(m.round_info),
            "match": f"{m.team_a} vs {m.team_b}",
            "cotes": f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}",
            "pick_cote": pc,
            "p_cote%": f"{p_c*100:.0f}",
            "pick_v3": pb,
            "p_v3%": f"{p_b*100:.0f}" if pb != "—" else "—",
            "edge_v3": f"{edge_b*100:+.1f}%" if pb != "—" else "—",
            "x_aware": x_flag,
            "score": score_pred,
            "top3_scores": top3_str,
            "accord": accord,
        })

    out = pd.DataFrame(rows).sort_values(["heure", "match"]).reset_index(drop=True)
    now_local = now_utc.astimezone(MG_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n=== {len(out)} matchs (Madagascar UTC+3, maintenant : {now_local}) ===\n")
    print(out.to_string(index=False))
    print()
    print("Lecture :")
    print("  - pick_v3    = argmax du modele V3 Blend (Poisson + Multi-market IPF)")
    print("  - pick_cote  = argmax des cotes calibrees seules")
    print("  - x_aware X! = la regle X-aware suggere X au lieu du pick principal")
    print("                 (match equilibre avec proba nul >=27%)")
    print("  - score      = score modal du marche bookmaker (Top-1 acc ~12.7%)")
    print("  - top3_scores= 3 scores plus probables (Top-3 acc ~34.5%)")
    print("  - accord OUI = blend V3 et cote sont d'accord (signal renforce)")

    out_path = Path("exports") / "predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nCSV : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
