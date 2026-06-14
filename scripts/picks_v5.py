"""V5 final avec 5 categories optimisees.

ROI backtest n=575 :
  A. ULTRA SAFE FT (V3 p>=70%)                            -> ROI +5%
  B. H2H FT X-Value (n>=5, X>=30%)                        -> ROI +7%
  C1. MEGA HT-X (V5 p>=40% + cote[1.5;1.9] + tier<0.3)   -> ROI +30.6% TOP
  C2. H2H HT-X (n>=5, HT-X>=40%, cote[1.5;2.5])          -> ROI +18.9%
  C3. BALANCED HT-X (cote[1.5;2.0] + tier<0.15)          -> ROI +19.0%
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5

MG_TZ = timezone(timedelta(hours=3))


def compute_h2h_ht(history_df):
    """Compute H2H HT outcomes per pair."""
    out = defaultdict(lambda: {"n": 0, "1": 0, "X": 0, "2": 0})
    for r in history_df.itertuples():
        key = (r.team_a, r.team_b)
        ht_outcome = ("1" if r.ht_score_a > r.ht_score_b
                       else "X" if r.ht_score_a == r.ht_score_b
                       else "2")
        out[key]["n"] += 1
        out[key][ht_outcome] += 1
    return dict(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p-ft", type=float, default=0.70)
    args = ap.parse_args()

    settings = load_settings()
    engine = create_engine(settings.db_url)

    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e JOIN odds_snapshots o ON o.event_id = e.id
                       JOIN results r ON r.event_id = e.id
    """, engine)
    ht_history = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b,
               r.ht_score_a, r.ht_score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL
    """, engine)
    if history.empty:
        print("aucun historique"); return 1

    model = fit_model_v5(history, ht_history=ht_history, engine=engine, form_alpha=0.0)
    h2h_ht_stats = compute_h2h_ht(ht_history)

    print(f"=== V5 OPTIMISE — n_train={model.n_train} | HT samples={len(ht_history)} ===\n")

    now_utc = datetime.now(timezone.utc)
    # On garde uniquement le DERNIER odds_snapshot par event pour eviter les doublons
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
        FROM events e
        JOIN odds_snapshots o ON o.id = (
            SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id
        )
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL
        ORDER BY e.expected_start
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming["expected_start"], utc=True, errors="coerce")
    upcoming = upcoming[
        upcoming["expected_start"].notna()
        & (upcoming["expected_start"] > now_utc)
    ].copy()
    if upcoming.empty:
        print("Aucun upcoming."); return 0

    cat_a, cat_b, c1_mega, c2_h2h, c3_balanced = [], [], [], [], []

    for _, m in upcoming.iterrows():
        pred = predict_match_v5(model, m.team_a, m.team_b,
                                  m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)
        if pred.get("primary_p") is None: continue

        em = m.extra_markets
        if isinstance(em, str):
            try: em = json.loads(em)
            except Exception: em = {}
        ht_x_cote = em.get("Mi-tps 1X2", {}).get("X") if isinstance(em.get("Mi-tps 1X2"), dict) else None

        local = m.expected_start.tz_convert(MG_TZ).strftime("%H:%M:%S")
        delta_s = (m.expected_start - now_utc).total_seconds()
        statut = f"+{int(delta_s)}s"

        primary_pick = pred["primary_pick"]
        primary_p = pred["primary_p"]
        cote_primary = m.odds_home if primary_pick == "1" else (m.odds_draw if primary_pick == "X" else m.odds_away)
        h2h_n = pred.get("h2h_n", 0)
        h2h_x = pred.get("h2h_x_rate") or 0
        ht_p_x = pred.get("p_d_ht") or 0
        tier_abs = abs(pred.get("attack_diff") or 0)

        # H2H HT stats
        ph = h2h_ht_stats.get((m.team_a, m.team_b), {"n": 0, "X": 0})
        h2h_ht_n = ph["n"]
        h2h_ht_x_rate = ph["X"] / ph["n"] if ph["n"] > 0 else 0

        common = {
            "heure": local, "T-": statut,
            "match": f"{m.team_a} vs {m.team_b}",
            "tier": f"{(pred.get('attack_diff') or 0):+.2f}",
        }

        # Cat A — Ultra Safe FT
        if primary_p >= args.p_ft:
            cat_a.append({**common, "pick": primary_pick, "cote": f"{cote_primary:.2f}",
                           "p": f"{primary_p*100:.0f}%"})

        # Cat B — H2H FT X-Value
        if h2h_n >= 5 and h2h_x >= 0.30:
            cat_b.append({**common, "pari": "X (FT)", "cote": f"{m.odds_draw:.2f}",
                           "H2H_FT": f"n={h2h_n} X={h2h_x*100:.0f}%"})

        # Cat C1 — MEGA HT-X (ROI +30.6%)
        if (ht_x_cote is not None and 1.5 <= ht_x_cote < 1.9
                and ht_p_x >= 0.40 and tier_abs < 0.3):
            c1_mega.append({**common, "pari": "HT X", "cote": f"{ht_x_cote:.2f}",
                             "V5_p_HT_X": f"{ht_p_x*100:.0f}%"})

        # Cat C2 — H2H HT-X (ROI +18.9%)
        if (h2h_ht_n >= 5 and h2h_ht_x_rate >= 0.40
                and ht_x_cote is not None and 1.5 <= ht_x_cote < 2.5):
            c2_h2h.append({**common, "pari": "HT X", "cote": f"{ht_x_cote:.2f}",
                            "H2H_HT": f"n={h2h_ht_n} X={h2h_ht_x_rate*100:.0f}%"})

        # Cat C3 — BALANCED HT-X (ROI +19%)
        if (ht_x_cote is not None and 1.5 <= ht_x_cote < 2.0
                and tier_abs < 0.15):
            c3_balanced.append({**common, "pari": "HT X", "cote": f"{ht_x_cote:.2f}",
                                 "tier_abs": f"{tier_abs:.2f}"})

    print(f"=== ⭐⭐⭐ CAT C1 — MEGA HT-X (ROI BACKTEST +30.63%) ===")
    print(f"   Triple filtre : V5 p_HT_X>=40% + cote HT-X∈[1.5;1.9] + tier<0.3\n")
    if c1_mega:
        print(pd.DataFrame(c1_mega).to_string(index=False))
    else:
        print("   Aucun match qualifie.")
    print()

    print(f"=== ⭐⭐ CAT C2 — H2H HT-X (ROI BACKTEST +18.89%) ===")
    print(f"   Paire historique H2H HT-X >=40% (n>=5) + cote∈[1.5;2.5]\n")
    if c2_h2h:
        print(pd.DataFrame(c2_h2h).to_string(index=False))
    else:
        print("   Aucun match qualifie.")
    print()

    print(f"=== ⭐⭐ CAT C3 — BALANCED HT-X (ROI BACKTEST +19.00%) ===")
    print(f"   cote HT-X∈[1.5;2.0] + match TRES equilibre (|tier|<0.15)\n")
    if c3_balanced:
        print(pd.DataFrame(c3_balanced).to_string(index=False))
    else:
        print("   Aucun match qualifie.")
    print()

    print(f"=== ⭐ CAT A — Ultra Safe FT (V3 p>=70%, ROI +5%) ===")
    if cat_a:
        print(pd.DataFrame(cat_a).to_string(index=False))
    else:
        print("   Aucun.")
    print()

    print(f"=== ⭐ CAT B — H2H FT X-Value (ROI +7%) ===")
    if cat_b:
        print(pd.DataFrame(cat_b).to_string(index=False))
    else:
        print("   Aucun.")
    print()

    print(f"=== RECAP ===")
    print(f"   Cat C1 (MEGA HT-X +30%)        : {len(c1_mega):>3}  matchs")
    print(f"   Cat C2 (H2H HT-X +18.9%)       : {len(c2_h2h):>3}  matchs")
    print(f"   Cat C3 (Balanced HT-X +19%)    : {len(c3_balanced):>3}  matchs")
    print(f"   Cat A (Ultra Safe FT +5%)      : {len(cat_a):>3}  matchs")
    print(f"   Cat B (H2H FT X-Value +7%)     : {len(cat_b):>3}  matchs")
    print(f"   Total picks                     : {len(c1_mega)+len(c2_h2h)+len(c3_balanced)+len(cat_a)+len(cat_b):>3}")
    print(f"   Total upcoming                  : {len(upcoming):>3}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
