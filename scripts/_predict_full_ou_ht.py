"""Analyse COMPLÈTE Under/Over 0.5/1.5/2.5/3.5 + HT pour TOUS les matchs."""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5

MG_TZ = timezone(timedelta(hours=3))


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL
    """, engine)
    print(f"V5 chargé (n={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)

    # Historique pour H2H
    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    h_all["ft_o"] = np.where(h_all.score_a > h_all.score_b, "1",
                       np.where(h_all.score_a == h_all.score_b, "X", "2"))
    h_all["ht_o"] = np.where(h_all.ht_score_a > h_all.ht_score_b, "1",
                       np.where(h_all.ht_score_a == h_all.ht_score_b, "X", "2"))
    h_all["total"] = h_all.score_a + h_all.score_b
    h_all["ht_total"] = h_all.ht_score_a + h_all.ht_score_b

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets, e.id as ev_id
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming.expected_start, utc=True)
    upcoming = upcoming[upcoming.expected_start > now_utc].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    upcoming = upcoming.sort_values("ev_id", ascending=False).drop_duplicates(["team_a", "team_b", "local"])
    rounds = sorted(upcoming.local.unique())[:6]
    print(f"Rounds : {rounds}\n")

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time].head(10)
        print(f"\n{'═' * 130}")
        print(f"  ⏰ ROUND {round_time} Mada — {len(matches)} matchs — Analyse Under/Over + HT complète")
        print(f"{'═' * 130}\n")

        # Header
        print(f"{'Match':<38} {'Cotes':<17} {'Buts att':<8} {'O0.5':<6} {'O1.5':<6} {'O2.5':<6} {'O3.5':<6} {'U2.5':<6} {'U3.5':<6} {'BTTS':<6} {'HT pick':<8} {'HT 1/X/2':<14} {'Score HT'}")
        print("-" * 150)

        for _, m in matches.iterrows():
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)

            # Lambdas from V5
            if pred5.get("lam_h_ht"):
                lam_h_ht = pred5["lam_h_ht"]
                lam_a_ht = pred5["lam_a_ht"]
                lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
                lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
                lam_total_ft = lam_h_ft + lam_a_ft
                lam_total_ht = lam_h_ht + lam_a_ht

                # Over/Under probabilities (Poisson on total)
                p_over_05 = 1 - poisson.cdf(0, lam_total_ft)
                p_over_15 = 1 - poisson.cdf(1, lam_total_ft)
                p_over_25 = 1 - poisson.cdf(2, lam_total_ft)
                p_over_35 = 1 - poisson.cdf(3, lam_total_ft)
                p_under_25 = poisson.cdf(2, lam_total_ft)
                p_under_35 = poisson.cdf(3, lam_total_ft)

                # BTTS
                p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft))

                # HT predictions
                p_h_ht = pred5.get("p_h_ht", 0)
                p_d_ht = pred5.get("p_d_ht", 0)
                p_a_ht = pred5.get("p_a_ht", 0)
                ht_pick = pred5.get("ht_pick", "—")
                ht_score = (int(round(lam_h_ht)), int(round(lam_a_ht)))
            else:
                # Fallback
                lam_total_ft = 0; p_over_05 = p_over_15 = p_over_25 = p_over_35 = 0
                p_under_25 = p_under_35 = p_btts = 0
                p_h_ht = p_d_ht = p_a_ht = 0
                ht_pick = "—"; ht_score = (0, 0)

            # H2H stats (real history pour boost confiance)
            h2h = h_all[(h_all.team_a == m.team_a) & (h_all.team_b == m.team_b)]
            h2h_n = len(h2h)
            if h2h_n >= 10:
                h2h_over_25 = (h2h.total > 2.5).mean()
                h2h_btts = ((h2h.score_a >= 1) & (h2h.score_b >= 1)).mean()
                # Blend model + h2h
                p_over_25 = 0.5 * p_over_25 + 0.5 * h2h_over_25
                p_btts = 0.5 * p_btts + 0.5 * h2h_btts

            match_str = f"{m.team_a} vs {m.team_b}"[:36]
            cotes_str = f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}"
            ht_dist = f"{p_h_ht*100:.0f}/{p_d_ht*100:.0f}/{p_a_ht*100:.0f}"
            score_ht_str = f"{ht_score[0]}-{ht_score[1]}"

            # Highlight columns above 80% / 90%
            def fmt(p):
                pct = p * 100
                if pct >= 90: return f"{pct:>4.0f}🔥"
                if pct >= 80: return f"{pct:>4.0f}⭐"
                if pct >= 70: return f"{pct:>4.0f}✓"
                if pct >= 50: return f"{pct:>4.0f} "
                return f"{pct:>4.0f} "

            print(f"{match_str:<38} {cotes_str:<17} {lam_total_ft:>5.1f}    {fmt(p_over_05):<6} {fmt(p_over_15):<6} {fmt(p_over_25):<6} {fmt(p_over_35):<6} {fmt(p_under_25):<6} {fmt(p_under_35):<6} {fmt(p_btts):<6} {ht_pick:<8} {ht_dist:<14} {score_ht_str}")

        print()
        print("  📝 LÉGENDE :")
        print("     🔥 ≥90%   ⭐ ≥80%   ✓ ≥70%   (sans marqueur) <70%")
        print()
        print(f"  💡 Recommandations pour {round_time}:")

        # Picks recommandés par catégorie
        for _, m in matches.iterrows():
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            if not pred5.get("lam_h_ht"): continue
            lam_h_ht = pred5["lam_h_ht"]
            lam_a_ht = pred5["lam_a_ht"]
            lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
            lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
            lam_total = lam_h_ft + lam_a_ft

            p_over_05 = 1 - poisson.cdf(0, lam_total)
            p_over_15 = 1 - poisson.cdf(1, lam_total)
            p_over_25 = 1 - poisson.cdf(2, lam_total)
            p_under_25 = poisson.cdf(2, lam_total)
            p_under_35 = poisson.cdf(3, lam_total)
            p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft))

            # H2H
            h2h = h_all[(h_all.team_a == m.team_a) & (h_all.team_b == m.team_b)]
            recs = []
            if h2h.shape[0] >= 10:
                h2h_over_25 = (h2h.total > 2.5).mean()
                h2h_over_15 = (h2h.total > 1.5).mean()
                h2h_under_35 = (h2h.total <= 3.5).mean()
                h2h_btts = ((h2h.score_a >= 1) & (h2h.score_b >= 1)).mean()
                if h2h_over_15 >= 0.90:
                    recs.append(f"Over 1.5 ({h2h_over_15*100:.0f}% H2H n={h2h.shape[0]})")
                if h2h_over_25 >= 0.80:
                    recs.append(f"Over 2.5 ({h2h_over_25*100:.0f}% H2H)")
                if h2h_under_35 >= 0.80:
                    recs.append(f"Under 3.5 ({h2h_under_35*100:.0f}% H2H)")
                if h2h_btts >= 0.80:
                    recs.append(f"BTTS Oui ({h2h_btts*100:.0f}% H2H)")
                if h2h_btts <= 0.30:
                    recs.append(f"BTTS Non ({100-h2h_btts*100:.0f}% H2H)")
            if p_over_15 >= 0.85: recs.append(f"Over 1.5 (modèle {p_over_15*100:.0f}%)")
            if p_under_35 >= 0.70: recs.append(f"Under 3.5 (modèle {p_under_35*100:.0f}%)")

            if recs:
                print(f"     • {m.team_a} vs {m.team_b}: {' | '.join(recs[:3])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
