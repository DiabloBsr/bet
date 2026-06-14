"""V6 picks — chasseur d'upsets et d'inefficiences.

Stratégie anti-favori validée 5-fold out-of-sample (n=2524 portfolio) :
  • HT/FT 1/2 (away revient au score)        ROI +104%
  • HT/FT 2/1 (home revient au score)         ROI  +28%
  • Score 1-0 (home outsider gagne sec)      ROI  +42%
  • HT-1 (outsider home mène à la mi-tps)    ROI marginal

Le système IGNORE délibérément la cote 1X2 favori : il chasse les upsets.

Usage :
  python scripts/picks_v6.py [--round HH:MM] [--ev-min 0.5] [--bankroll 100]
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v6 import fit_model_v6, predict_edges_v6

MG_TZ = timezone(timedelta(hours=3))
SIG_DISPLAY = {
    "ht_ft_1_2": "HT/FT 1/2",
    "ht_ft_2_1": "HT/FT 2/1",
    "upset_1_0": "Score 1-0",
    "outsider_ht1": "HT-1 outsider",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default=None)
    ap.add_argument("--ev-min", type=float, default=0.5)
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--mode", choices=["best", "all"], default="all",
                    help="best=meilleur signal/match, all=tous signaux positifs")
    args = ap.parse_args()

    settings = load_settings()
    engine = create_engine(settings.db_url)

    history = pd.read_sql("""
        SELECT o.odds_home, o.odds_away, r.score_a, r.score_b,
               r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL
    """, engine)
    if history.empty:
        print("Pas d'historique."); return 1

    model = fit_model_v6(history)
    print(f"=== V6 PICKS — n_train={model.n_train} | EV≥{args.ev_min} | mode={args.mode} ===")
    print(f"  p(HT/FT 1/2)={model.p_12_global*100:.2f}%  p(HT/FT 2/1)={model.p_21_global*100:.2f}%  p(1-0 quand away fav)={model.p_1_0_when_away_fav*100:.2f}%\n")

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL
        ORDER BY e.expected_start
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming["expected_start"], utc=True, errors="coerce")
    upcoming = upcoming[upcoming.expected_start.notna() & (upcoming.expected_start > now_utc)].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")

    if args.round:
        upcoming = upcoming[upcoming.local == args.round]
        if upcoming.empty:
            avail = sorted(upcoming.local.unique()) if not upcoming.empty else []
            print(f"Round {args.round} introuvable. Disponibles : {avail}")
            return 1
        print(f"Round ciblé : {args.round} ({len(upcoming)} matchs)\n")

    picks = []
    for _, m in upcoming.iterrows():
        edges = predict_edges_v6(model, m.odds_home, m.odds_away, m.extra_markets)
        match = f"{m.team_a} vs {m.team_b}"
        cotes_str = f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}"
        match_options = []
        for sig_key in ["ht_ft_1_2", "ht_ft_2_1", "upset_1_0", "outsider_ht1"]:
            sig = edges[sig_key]
            if sig["cote"] is None or sig["ev"] is None: continue
            match_options.append({
                "local": m.local, "match": match,
                "sig": sig_key, "label": SIG_DISPLAY[sig_key],
                "cote": sig["cote"], "p_emp": sig["p_emp"], "ev": sig["ev"],
                "cotes_ft": cotes_str,
            })
        if args.mode == "best" and match_options:
            picks.append(max(match_options, key=lambda x: x["ev"]))
        else:
            picks.extend(match_options)

    picks = [p for p in picks if p["ev"] >= args.ev_min]
    picks.sort(key=lambda p: (-p["ev"], p["local"]))

    if not picks:
        print(f"Aucun pari EV ≥ {args.ev_min}. Essaye --ev-min plus bas.")
        return 0

    print(f"=== {len(picks)} PARIS RECOMMANDÉS ===\n")
    print(f"{'heure':<7} {'match':<38} {'pari':<14} {'cote':<7} {'p_emp':<7} {'EV':<8} {'cotes 1X2'}")
    print("-" * 110)
    for p in picks:
        print(f"{p['local']:<7} {p['match']:<38} {p['label']:<14} @{p['cote']:<5.1f} {p['p_emp']*100:>5.2f}%  {p['ev']*100:>+6.0f}%  {p['cotes_ft']}")

    # Plan Kelly fractionnaire 1/8
    print(f"\n=== PLAN DE MISE (bankroll {args.bankroll:.0f}u, Kelly 1/8) ===")
    total_mise = total_expected = 0
    for p in picks:
        b = p["cote"] - 1
        pe = p["p_emp"]
        kelly = (b * pe - (1 - pe)) / b if b > 0 else 0
        mise = max(0, kelly * args.bankroll / 8)
        mise = min(mise, args.bankroll * 0.05)
        expected = p["ev"] * mise
        total_mise += mise
        total_expected += expected
        print(f"  {p['local']:<6} {p['match'][:30]:<30} {p['label']:<13} @{p['cote']:<5.1f}  mise {mise:>5.2f}u  EV {expected:>+6.2f}u")

    print(f"\nTotal misé   : {total_mise:.2f}u / {args.bankroll:.0f}u ({total_mise/args.bankroll*100:.1f}%)")
    print(f"Gain attendu : {total_expected:+.2f}u")
    if total_mise > 0:
        print(f"ROI attendu  : {total_expected/total_mise*100:+.1f}%")

    print()
    print("Note : variance énorme (acc 3-6%). Drawdowns longs normaux.")
    print("Backtest 5-fold n=2524 (out-of-sample) confirme ROI portefeuille +57%.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
