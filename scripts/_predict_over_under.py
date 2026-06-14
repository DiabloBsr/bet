"""Prédictions UNDER / OVER pour les rounds à venir.

Pour chaque match :
- Buts attendus (modèle Poisson)
- Probabilités Over 0.5, 1.5, 2.5, 3.5, 4.5
- Probabilités Under 1.5, 2.5, 3.5
- Signaux OVER_GOLD / UNDER_GOLD historiques (avec filtres calibrage)
- BTTS oui / non
"""
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
from scraper.team_gold_data import OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD

MG_TZ = timezone(timedelta(hours=3))
TARGETS = ["19:12", "19:14", "19:16", "19:18", "19:20", "19:22", "19:24"]


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
    print(f"Modèle V5 (n_train={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start, e.round_info,
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

    all_picks = []

    for TARGET in TARGETS:
        round_matchs = upcoming[upcoming.local == TARGET]
        if len(round_matchs) == 0:
            continue
        print(f"\n{'═'*115}")
        print(f"  ⏰ ROUND {TARGET} — UNDER / OVER")
        print(f"{'═'*115}")
        print(f"  {'MATCH':<38} {'Buts':<6} {'O1.5':<6} {'O2.5':<6} {'O3.5':<6} {'U2.5':<6} {'U3.5':<6} {'BTTS':<6} {'SIGNAL'}")
        print(f"  {'-'*115}")

        for _, m in round_matchs.iterrows():
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                      extra_markets=m.extra_markets)
            if pred5.get("lam_h_ht"):
                lam_h_ht = pred5["lam_h_ht"]; lam_a_ht = pred5["lam_a_ht"]
                lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
                lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
                lam_total = lam_h_ft + lam_a_ft
                p_over_05 = (1 - poisson.cdf(0, lam_total)) * 100
                p_over_15 = (1 - poisson.cdf(1, lam_total)) * 100
                p_over_25 = (1 - poisson.cdf(2, lam_total)) * 100
                p_over_35 = (1 - poisson.cdf(3, lam_total)) * 100
                p_over_45 = (1 - poisson.cdf(4, lam_total)) * 100
                p_under_15 = poisson.cdf(1, lam_total) * 100
                p_under_25 = poisson.cdf(2, lam_total) * 100
                p_under_35 = poisson.cdf(3, lam_total) * 100
                p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft)) * 100
            else:
                continue

            signals = []
            # OVER GOLD (filter: lam_total >= 3.5 for validity)
            if (m.team_a, m.team_b) in OVER_GOLD:
                og = OVER_GOLD[(m.team_a, m.team_b)]
                if lam_total >= 3.5:
                    signals.append(f"💎 OVER GOLD {og['rate']*100:.0f}% (n={og.get('n',0)})")
                else:
                    signals.append(f"⚠️ OG hist {og['rate']*100:.0f}% mais buts {lam_total:.1f}<3.5")
            # UNDER GOLD
            if (m.team_a, m.team_b) in UNDER_GOLD:
                ug = UNDER_GOLD[(m.team_a, m.team_b)]
                rate = 1 - ug['over_rate']
                signals.append(f"💎💎 UNDER GOLD {rate*100:.0f}% (n={ug.get('n',0)})")
            # BTTS GOLD
            if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
                bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
                signals.append(f"💎 BTTS OUI {bg['rate']*100:.0f}%")
            if (m.team_a, m.team_b) in BTTS_NON_GOLD:
                bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
                rate = 1 - bn['bts_rate']
                signals.append(f"💎 BTTS NON {rate*100:.0f}%")

            match_lbl = f"{m.team_a} vs {m.team_b}"[:36]
            print(f"  {match_lbl:<38} {lam_total:<6.2f} {p_over_15:<5.0f}% {p_over_25:<5.0f}% {p_over_35:<5.0f}% {p_under_25:<5.0f}% {p_under_35:<5.0f}% {p_btts:<5.0f}% {' '.join(signals)}")

            all_picks.append({
                "round": TARGET, "match": match_lbl,
                "buts": lam_total,
                "p_over_15": p_over_15, "p_over_25": p_over_25, "p_over_35": p_over_35,
                "p_under_25": p_under_25, "p_under_35": p_under_35,
                "p_btts": p_btts,
                "over_gold": (m.team_a, m.team_b) in OVER_GOLD,
                "under_gold": (m.team_a, m.team_b) in UNDER_GOLD,
                "btts_oui_gold": (m.team_a, m.team_b) in BTTS_OUI_GOLD,
                "btts_non_gold": (m.team_a, m.team_b) in BTTS_NON_GOLD,
            })

    # ============ TOP PICKS ============
    print(f"\n\n{'═'*115}")
    print(f"  🎯 TOP PICKS UNDER / OVER")
    print(f"{'═'*115}\n")
    df = pd.DataFrame(all_picks)
    if df.empty:
        return 0

    # OVER 2.5 — buts ≥ 3.5 (calibré)
    print("  🔥 OVER 2.5 — Buts attendus ≥3.5 + GOLD (priorité MAX) :")
    safe_over = df[(df.buts >= 3.5) & (df.p_over_25 >= 65)].sort_values("p_over_25", ascending=False)
    for _, r in safe_over.head(10).iterrows():
        marker = " 💎" if r.over_gold else ""
        print(f"     {r['round']}  {r['match']:<38}  Buts {r.buts:.2f}  O2.5 {r.p_over_25:.0f}%{marker}")

    print()
    print("  🛡️  UNDER 2.5 — Buts ≤2.5 + GOLD (priorité MAX) :")
    safe_under = df[(df.buts <= 2.6) & (df.p_under_25 >= 60)].sort_values("p_under_25", ascending=False)
    for _, r in safe_under.head(10).iterrows():
        marker = " 💎💎" if r.under_gold else ""
        print(f"     {r['round']}  {r['match']:<38}  Buts {r.buts:.2f}  U2.5 {r.p_under_25:.0f}%{marker}")

    print()
    print("  🛡️  UNDER 3.5 — Très défensif (≥70%) :")
    safe_under35 = df[df.p_under_35 >= 70].sort_values("p_under_35", ascending=False)
    for _, r in safe_under35.head(10).iterrows():
        marker = " 💎" if r.under_gold else ""
        print(f"     {r['round']}  {r['match']:<38}  Buts {r.buts:.2f}  U3.5 {r.p_under_35:.0f}%{marker}")

    print()
    print("  🔥 OVER 3.5 — High scoring (≥45%) :")
    safe_over35 = df[df.p_over_35 >= 45].sort_values("p_over_35", ascending=False)
    for _, r in safe_over35.head(10).iterrows():
        marker = " 💎" if r.over_gold else ""
        print(f"     {r['round']}  {r['match']:<38}  Buts {r.buts:.2f}  O3.5 {r.p_over_35:.0f}%{marker}")

    print()
    print("  💎 BTTS OUI — Probable ≥60% :")
    safe_btts = df[df.p_btts >= 60].sort_values("p_btts", ascending=False)
    for _, r in safe_btts.head(10).iterrows():
        marker = " 💎" if r.btts_oui_gold else ""
        print(f"     {r['round']}  {r['match']:<38}  BTTS {r.p_btts:.0f}%{marker}")

    print()
    print("  🛡️  BTTS NON — Pas de buts pour les 2 (≤45%) :")
    safe_btts_non = df[df.p_btts <= 45].sort_values("p_btts", ascending=True)
    for _, r in safe_btts_non.head(10).iterrows():
        marker = " 💎" if r.btts_non_gold else ""
        print(f"     {r['round']}  {r['match']:<38}  BTTS {r.p_btts:.0f}%{marker}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
