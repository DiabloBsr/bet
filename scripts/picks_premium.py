"""High-confidence pick selector (V3 model).

Filtre :
  - p_Blend (V3) >= seuil (defaut 0.55)
  - p_Cote >= seuil cote (defaut 0.55)
  - Blend V3 et Cote d'accord (sauf si --include-disagree)

Backtest V3 :
  - p>=55% : ~70% accuracy, ROI ~0% (le filtre 55% manque maintenant de juice)
  - p>=60% : ~76% accuracy, ROI +3.5%
  - p>=70% : 87% accuracy, ROI +5.0%

Usage :
  python scripts/picks_premium.py                   # defaut p>=0.55
  python scripts/picks_premium.py --threshold 0.60  # bon compromis
  python scripts/picks_premium.py --threshold 0.70  # ultra-selectif (87% acc)
"""
from __future__ import annotations

import argparse
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--cote-min", type=float, default=0.55)
    ap.add_argument("--include-disagree", action="store_true")
    ap.add_argument("--include-past", action="store_true")
    args = ap.parse_args()

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
        print("aucun match resolu — entraine d'abord")
        return 1
    model = fit_model_v3(history, engine=engine, form_alpha=0.0)

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
    if not args.include_past:
        window_start = now_utc - timedelta(minutes=5)
        upcoming = upcoming[
            upcoming["expected_start"].notna()
            & (upcoming["expected_start"] >= window_start)
        ]
    if upcoming.empty:
        print(f"Aucun match dans la fenetre.")
        return 1

    picks = []
    skipped = 0
    for _, m in upcoming.iterrows():
        pred = predict_match_v3(
            model, m.team_a, m.team_b,
            m.odds_home, m.odds_draw, m.odds_away,
            extra_markets=m.extra_markets,
        )
        if pred["p_h_blend"] is None:
            skipped += 1
            continue

        c = [("1", pred["p_h_cote"], m.odds_home),
             ("X", pred["p_d_cote"], m.odds_draw),
             ("2", pred["p_a_cote"], m.odds_away)]
        b = [("1", pred["p_h_blend"], m.odds_home),
             ("X", pred["p_d_blend"], m.odds_draw),
             ("2", pred["p_a_blend"], m.odds_away)]
        pc, p_c, odds_c = _pick(c)
        pb, p_b, odds_b = _pick(b)

        if p_b < args.threshold:
            continue
        if p_c < args.cote_min:
            continue
        if not args.include_disagree and pc != pb:
            continue

        edge_b = p_b * odds_b - 1
        confidence = (p_b + p_c) / 2

        if pd.notna(m.expected_start):
            local_time = m.expected_start.tz_convert(MG_TZ).strftime("%H:%M:%S")
            delta_s = (m.expected_start - now_utc).total_seconds()
            statut = f"dans {int(delta_s)}s" if delta_s > 0 else (
                f"en cours ({int(-delta_s)}s)" if delta_s > -180 else "passe"
            )
        else:
            local_time = "—"; statut = "—"

        top3 = pred.get("top3_market") or pred.get("top3_blend") or []
        top3_str = " / ".join(s for s, _ in top3[:3]) if top3 else "—"
        # X-aware indicator
        x_aware = pred.get("pick_xaware")
        x_flag = "  [+X opp]" if x_aware == "X" and pb != "X" else ""

        picks.append({
            "heure": local_time,
            "statut": statut,
            "match": f"{m.team_a} vs {m.team_b}",
            "cotes": f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}",
            "pick": pb + x_flag,
            "p_blend%": f"{p_b*100:.1f}",
            "p_cote%": f"{p_c*100:.1f}",
            "confidence%": f"{confidence*100:.1f}",
            "edge_blend": f"{edge_b*100:+.1f}%",
            "score": pred.get("score_market") or pred.get("score_blend") or "—",
            "top3_scores": top3_str,
            "tier": f"{pred['attack_diff']:+.2f}" if pred.get("attack_diff") is not None else "—",
        })

    now_local = now_utc.astimezone(MG_TZ).strftime("%H:%M:%S")
    print(f"=== Selection PREMIUM V3 (Madagascar {now_local}) ===")
    print(f"Modele V3 n={model.n_train}  |  Multi-market IPF + X-aware rule")
    print(f"Filtres : p_V3 >= {args.threshold:.0%} | p_Cote >= {args.cote_min:.0%}"
           + ("" if args.include_disagree else " | accord requis"))
    print()

    if not picks:
        print(f"Aucun pick au seuil p={args.threshold:.0%}.")
        return 0

    pdf = pd.DataFrame(picks).sort_values("confidence%", ascending=False).reset_index(drop=True)
    print(pdf.to_string(index=False))
    print()
    print(f"=> {len(pdf)} pick(s) premium V3")
    if skipped:
        print(f"   {skipped} match(s) ignores (data equipes manquantes)")
    print()
    print("Backtest V3 :")
    print(f"  p>=55% : 70.2% acc, ROI 0.0%")
    print(f"  p>=60% : 76.7% acc, ROI +3.5%")
    print(f"  p>=70% : 84.8% acc, ROI +2.8%  (87.2% V3 no-form)")
    print(f"  Pour ROI max : utilise --threshold 0.60 ou 0.70")

    out_path = Path("exports") / "picks_premium.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nCSV : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
