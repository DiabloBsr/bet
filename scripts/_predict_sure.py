"""Analyse COMPLÈTE - tous types de paris à confiance ≥70%."""
from __future__ import annotations
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.predictor_v10 import fit_model_v10, predict_v10
from scraper.team_gold_data import (
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD, PAIR_TRAP_HOME,
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD, SCORE_DOMINANT_GOLD,
)

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
    print(f"Modèles chargés (n={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    h_all["ft_o"] = np.where(h_all.score_a > h_all.score_b, "1",
                       np.where(h_all.score_a == h_all.score_b, "X", "2"))
    h_all["ht_o"] = np.where(h_all.ht_score_a > h_all.ht_score_b, "1",
                       np.where(h_all.ht_score_a == h_all.ht_score_b, "X", "2"))
    h_all["score"] = h_all.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    h_all["total"] = h_all.score_a + h_all.score_b
    h_all["btts"] = ((h_all.score_a >= 1) & (h_all.score_b >= 1)).astype(int)

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

    # Catégorisation : 98%+ / 90-97% / 80-89% / 70-79%
    sure_98 = []      # ≥98%
    sure_90 = []      # 90-97%
    sure_80 = []      # 80-89%
    sure_70 = []      # 70-79%

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time].head(10)

        for i, (_, m) in enumerate(matches.iterrows(), 1):
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            pred10 = predict_v10(model_v10, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)

            h2h = h_all[(h_all.team_a == m.team_a) & (h_all.team_b == m.team_b)]
            h2h_n = len(h2h)
            cotes_str = f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}"
            match_str = f"{m.team_a} vs {m.team_b}"

            # === BUILD ALL SIGNALS ===
            signals = []

            # 1. 1X2 PICK
            p_h = pred10["p_model"]["1"] * 100
            p_x = pred10["p_model"]["X"] * 100
            p_a = pred10["p_model"]["2"] * 100
            if p_h >= 70:
                signals.append({"type": "1X2 pick", "bet": f"1 @{m.odds_home:.2f}", "rate": p_h/100, "source": "modèle"})
            if p_a >= 70:
                signals.append({"type": "1X2 pick", "bet": f"2 @{m.odds_away:.2f}", "rate": p_a/100, "source": "modèle"})
            # X rarement >=70%

            # 2. PAIRE OR HOME (validé)
            if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
                p = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
                if p["win"] >= 0.70:
                    signals.append({"type": "PAIRE OR", "bet": f"1 @{m.odds_home:.2f}", "rate": p["win"],
                                     "source": f"paire OR n={p['n']}"})

            # 3. PAIRE OR AWAY (filtré strict)
            if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
                p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
                if p["win"] >= 0.70 and m.odds_away <= p["cote"] * p.get("max_cote_factor", 1.05):
                    signals.append({"type": "PAIRE OR away", "bet": f"2 @{m.odds_away:.2f}", "rate": p["win"],
                                     "source": f"paire OR away n={p['n']}"})

            # 4. H2H Domination (≥70% sur n≥10)
            if h2h_n >= 10:
                w_rate = (h2h.ft_o == "1").mean()
                x_rate = (h2h.ft_o == "X").mean()
                a_rate = (h2h.ft_o == "2").mean()
                # Double Chance 1X
                dc_1x = (h2h.ft_o.isin(["1", "X"])).mean()
                dc_x2 = (h2h.ft_o.isin(["X", "2"])).mean()
                dc_12 = (h2h.ft_o.isin(["1", "2"])).mean()
                if dc_1x >= 0.85:
                    signals.append({"type": "Double Chance", "bet": f"1X (cote ~1.2-1.4)", "rate": dc_1x,
                                     "source": f"H2H n={h2h_n}"})
                if dc_x2 >= 0.85:
                    signals.append({"type": "Double Chance", "bet": f"X2", "rate": dc_x2,
                                     "source": f"H2H n={h2h_n}"})
                if dc_12 >= 0.85:
                    signals.append({"type": "Double Chance", "bet": f"12", "rate": dc_12,
                                     "source": f"H2H n={h2h_n}"})
                # 1X2 H2H
                if w_rate >= 0.70:
                    signals.append({"type": "1X2 H2H", "bet": f"1 @{m.odds_home:.2f}", "rate": w_rate,
                                     "source": f"H2H {int(w_rate*h2h_n)}/{h2h_n}"})
                if a_rate >= 0.70:
                    signals.append({"type": "1X2 H2H", "bet": f"2 @{m.odds_away:.2f}", "rate": a_rate,
                                     "source": f"H2H {int(a_rate*h2h_n)}/{h2h_n}"})

            # 5. OVER 2.5 / UNDER 2.5
            if (m.team_a, m.team_b) in OVER_GOLD:
                og = OVER_GOLD[(m.team_a, m.team_b)]
                if og["rate"] >= 0.70:
                    signals.append({"type": "OVER 2.5", "bet": "Plus de 2.5 buts", "rate": og["rate"],
                                     "source": f"GOLD n={og['n']}"})
            if (m.team_a, m.team_b) in UNDER_GOLD:
                ug = UNDER_GOLD[(m.team_a, m.team_b)]
                if (1-ug["over_rate"]) >= 0.70:
                    signals.append({"type": "UNDER 2.5", "bet": "Moins de 2.5 buts", "rate": 1-ug["over_rate"],
                                     "source": f"GOLD n={ug['n']}"})

            # 6. Test OVER 1.5 / UNDER 4.5 via H2H
            if h2h_n >= 10:
                over_15 = (h2h.total > 1.5).mean()
                over_25 = (h2h.total > 2.5).mean()
                over_35 = (h2h.total > 3.5).mean()
                under_45 = (h2h.total <= 4.5).mean()
                if over_15 >= 0.85:
                    signals.append({"type": "OVER 1.5", "bet": "Plus de 1.5 buts", "rate": over_15,
                                     "source": f"H2H n={h2h_n}"})
                if over_25 >= 0.70:
                    signals.append({"type": "OVER 2.5", "bet": "Plus de 2.5 buts", "rate": over_25,
                                     "source": f"H2H n={h2h_n}"})
                if under_45 >= 0.85:
                    signals.append({"type": "UNDER 4.5", "bet": "Moins de 4.5 buts", "rate": under_45,
                                     "source": f"H2H n={h2h_n}"})

            # 7. BTTS
            if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
                bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
                if bg["rate"] >= 0.70 and m.odds_home >= bg.get("min_cote_h", 1.8):
                    signals.append({"type": "BTTS Oui", "bet": "BTTS Oui", "rate": bg["rate"],
                                     "source": f"GOLD n={bg['n']}"})
            if (m.team_a, m.team_b) in BTTS_NON_GOLD:
                bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
                if (1-bn["bts_rate"]) >= 0.70:
                    signals.append({"type": "BTTS Non", "bet": "BTTS Non", "rate": 1-bn["bts_rate"],
                                     "source": f"GOLD n={bn['n']}"})

            # 8. HT (mi-temps)
            if h2h_n >= 10:
                ht_1 = (h2h.ht_o == "1").mean()
                ht_x = (h2h.ht_o == "X").mean()
                ht_2 = (h2h.ht_o == "2").mean()
                if ht_1 >= 0.70:
                    signals.append({"type": "HT pick", "bet": "HT 1", "rate": ht_1,
                                     "source": f"H2H n={h2h_n}"})
                if ht_2 >= 0.70:
                    signals.append({"type": "HT pick", "bet": "HT 2", "rate": ht_2,
                                     "source": f"H2H n={h2h_n}"})

            # 9. Score Exact GOLD
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                sg = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                if sg["rate"] >= 0.35 and sg["n"] >= 11:
                    signals.append({"type": "Score Exact", "bet": f"Score {sg['score']}", "rate": sg["rate"],
                                     "source": f"GOLD n={sg['n']}"})

            # Catégoriser
            for sig in signals:
                rate = sig["rate"]
                entry = {**sig, "round": round_time, "match": match_str, "cotes": cotes_str}
                if rate >= 0.98: sure_98.append(entry)
                elif rate >= 0.90: sure_90.append(entry)
                elif rate >= 0.80: sure_80.append(entry)
                elif rate >= 0.70: sure_70.append(entry)

    # AFFICHAGE
    print("=" * 110)
    print("  🔥🔥🔥 TIER ULTRA (≥98%) — paris quasi-sûrs")
    print("=" * 110)
    if not sure_98:
        print("  Aucun pick à 98%+")
    for s in sorted(sure_98, key=lambda x: -x["rate"]):
        print(f"  ⭐⭐⭐ [{s['round']}] {s['match']:<40} {s['type']:<15} {s['bet']:<25} {s['rate']*100:.1f}% ({s['source']})")

    print()
    print("=" * 110)
    print("  💎💎 TIER OR (90-97%)")
    print("=" * 110)
    if not sure_90:
        print("  Aucun pick à 90-97%")
    for s in sorted(sure_90, key=lambda x: -x["rate"]):
        print(f"  ⭐⭐ [{s['round']}] {s['match']:<40} {s['type']:<15} {s['bet']:<25} {s['rate']*100:.1f}% ({s['source']})")

    print()
    print("=" * 110)
    print("  💎 TIER FORT (80-89%)")
    print("=" * 110)
    for s in sorted(sure_80, key=lambda x: -x["rate"]):
        print(f"  ⭐ [{s['round']}] {s['match']:<40} {s['type']:<15} {s['bet']:<25} {s['rate']*100:.1f}% ({s['source']})")

    print()
    print("=" * 110)
    print("  ✓ TIER BON (70-79%)")
    print("=" * 110)
    for s in sorted(sure_70, key=lambda x: -x["rate"]):
        print(f"  [{s['round']}] {s['match']:<40} {s['type']:<15} {s['bet']:<25} {s['rate']*100:.1f}% ({s['source']})")

    # Bilan
    print()
    print("=" * 110)
    print(f"  📊 BILAN : {len(sure_98)} ultra + {len(sure_90)} OR + {len(sure_80)} fort + {len(sure_70)} bon")
    print("=" * 110)

    return 0


if __name__ == "__main__":
    sys.exit(main())
