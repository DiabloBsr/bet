"""V4 — predictions completes pour TOUS les matchs upcoming.

Avec capture multi-rounds (9 rounds futurs ~20 min), on voit toute la grille
en une commande. Chaque match a sa prediction V4 complete + categorie.

Categories :
  A. ULTRA SAFE (V3 p >= 70%)              -> 87% accuracy, ROI +5-6%
  B. X-VALUE (H2H n>=5, X_rate >= 30%)     -> 32% accuracy, ROI +7.5%
  C. NEUTRAL (prediction infos mais pas recommande pour pari)

Usage :
  python scripts/picks_v4.py
  python scripts/picks_v4.py --watch
  python scripts/picks_v4.py --p-threshold 0.65 --h2h-min 4
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v4 import fit_model_v4, predict_match_v4

MG_TZ = timezone(timedelta(hours=3))


def _pick_blend(pred):
    """Argmax du V3 blend."""
    probs = [
        ("1", pred.get("p_h_blend") or 0),
        ("X", pred.get("p_d_blend") or 0),
        ("2", pred.get("p_a_blend") or 0),
    ]
    return max(probs, key=lambda x: x[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-threshold", type=float, default=0.70)
    ap.add_argument("--h2h-min", type=int, default=5)
    ap.add_argument("--x-rate", type=float, default=0.30)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--watch-interval", type=int, default=30)
    args = ap.parse_args()

    if args.watch:
        return _watch(args)
    return _run(args)


def _watch(args):
    print(f"=== MODE WATCH (refresh chaque {args.watch_interval}s, Ctrl+C pour stop) ===\n")
    last_sig = None
    while True:
        sig = _run(args, return_signature=True)
        if sig and sig != last_sig and last_sig is not None:
            print(f"\n{'!' * 70}")
            print(f"  NOUVELLE GRILLE DETECTEE")
            print(f"{'!' * 70}\n")
        last_sig = sig
        time.sleep(args.watch_interval)
        print(f"\n--- {datetime.now(MG_TZ).strftime('%H:%M:%S')} re-check... ---\n")


def _run(args, return_signature: bool = False):
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
        print("aucun historique")
        return None if return_signature else 1

    model = fit_model_v4(
        history, engine=engine,
        h2h_min_n=args.h2h_min, h2h_x_threshold=args.x_rate, form_alpha=0.0,
    )

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
    upcoming["expected_start"] = pd.to_datetime(upcoming["expected_start"], utc=True, errors="coerce")
    upcoming = upcoming[
        upcoming["expected_start"].notna()
        & (upcoming["expected_start"] > now_utc)
    ].copy()
    if upcoming.empty:
        print("Aucun match strictement a venir.")
        return None if return_signature else 0

    rows = []
    for _, m in upcoming.iterrows():
        pred = predict_match_v4(
            model, m.team_a, m.team_b,
            m.odds_home, m.odds_draw, m.odds_away,
            extra_markets=m.extra_markets,
        )
        if pred["primary_p"] is None or pred["p_h_blend"] is None:
            continue

        local = m.expected_start.tz_convert(MG_TZ).strftime("%H:%M:%S")
        delta_s = (m.expected_start - now_utc).total_seconds()
        statut = f"+{int(delta_s)}s" if delta_s > 0 else "<live>"

        pick_blend, p_blend = _pick_blend(pred)
        cote_blend = m.odds_home if pick_blend == "1" else (m.odds_draw if pick_blend == "X" else m.odds_away)
        edge_blend = p_blend * cote_blend - 1
        score = pred.get("score_market") or pred.get("score_blend") or "—"
        top3 = pred.get("top3_market") or pred.get("top3_blend") or []
        top3_str = " ".join(s for s, _ in top3[:3]) if top3 else "—"

        h2h_n = pred.get("h2h_n", 0)
        h2h_x = pred.get("h2h_x_rate") or 0
        h2h_str = f"n={h2h_n} X={h2h_x*100:.0f}%" if h2h_n > 0 else "—"

        in_a = p_blend >= args.p_threshold
        in_b = h2h_n >= args.h2h_min and h2h_x >= args.x_rate
        if in_a and in_b:
            category = "A+B"
        elif in_a:
            category = "A"
        elif in_b:
            category = "B"
        else:
            category = "—"

        x_aware = pred.get("pick_xaware") == "X"

        rows.append({
            "heure": local,
            "T-": statut,
            "match": f"{m.team_a} vs {m.team_b}",
            "cotes": f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}",
            "pick": pick_blend,
            "p%": f"{p_blend*100:.0f}",
            "cote_pick": f"{cote_blend:.2f}",
            "edge": f"{edge_blend*100:+.1f}%",
            "H2H": h2h_str,
            "Xaw": "X!" if x_aware else "",
            "tier": f"{pred['attack_diff']:+.2f}" if pred.get("attack_diff") is not None else "—",
            "score": score,
            "top3": top3_str,
            "cat": category,
        })

    df = pd.DataFrame(rows)
    df["round_group"] = df["heure"].astype(str)

    now_local = now_utc.astimezone(MG_TZ).strftime("%Y-%m-%d %H:%M:%S")
    print(f"=== V4 GRILLE COMPLETE — {len(df)} matchs upcoming | n_train={model.n_train} ===")
    print(f"Heure Madagascar : {now_local}\n")

    # Compteurs categories
    n_a = (df["cat"].str.contains("A", na=False)).sum()
    n_b = (df["cat"].str.contains("B", na=False)).sum()
    n_x_aware = (df["Xaw"] == "X!").sum()
    print(f"Categorie A (Ultra Safe p>={args.p_threshold:.0%}) : {n_a}")
    print(f"Categorie B (X-Value H2H n>={args.h2h_min} X>={args.x_rate:.0%}) : {n_b}")
    print(f"Alertes X-aware (match equilibre p_X>=27%) : {n_x_aware}")
    print()

    # Affichage groupe par round (heure)
    for heure in sorted(df["heure"].unique()):
        sub = df[df["heure"] == heure].copy()
        first = sub.iloc[0]
        print(f"=== Round @ {heure} ({first['T-']}) — {len(sub)} matchs ===")
        sub_display = sub[["match", "cotes", "pick", "p%", "cote_pick", "edge", "H2H", "Xaw", "tier", "score", "top3", "cat"]]
        print(sub_display.to_string(index=False))
        print()

    # Sauve CSV
    out_path = Path("exports") / "predictions_full.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"CSV complet : {out_path}")

    # Legende
    print()
    print("Legende :")
    print("  pick       = V4 argmax (1=home, X=nul, 2=away)")
    print("  cote_pick  = cote du pick")
    print("  edge       = p_V4 * cote - 1 (>0 = profitable theorique)")
    print("  H2H        = historique paire (n, X_rate)")
    print("  Xaw X!     = X-aware rule declenchee (match equilibre)")
    print("  tier       = ecart de force home-away (+ = home fort)")
    print("  score      = score modal predit")
    print("  top3       = 3 scores les plus probables")
    print("  cat        = A=Ultra Safe | B=X Value | A+B=both | —=ne pas parier")

    signature = tuple(sorted(f"{r.team_a}|{r.team_b}|{r.expected_start}" for _, r in upcoming.iterrows()))
    return signature if return_signature else 0


if __name__ == "__main__":
    sys.exit(main())
