"""Prédictions SCORE EXACT seulement - tous rounds upcoming."""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.team_gold_data import SCORE_DOMINANT_GOLD

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
    print(f"V5 chargé (n_train={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)

    # Historique pour H2H scores
    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    h_all["score"] = h_all.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

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
    rounds = sorted(upcoming.local.unique())[:8]
    print(f"Rounds : {rounds}\n")

    all_score_picks = []

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time].head(10)
        print(f"\n{'═' * 105}")
        print(f"  ⏰ ROUND {round_time} — SCORE EXACT focus")
        print(f"{'═' * 105}")

        for i, (_, m) in enumerate(matches.iterrows(), 1):
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            top5 = pred5.get("top5_scores_enriched") or []

            # H2H scores
            h2h_scores = h_all[(h_all.team_a == m.team_a) & (h_all.team_b == m.team_b)]
            h2h_n = len(h2h_scores)
            h2h_modal = None
            if h2h_n >= 5:
                top_h2h = h2h_scores.score.value_counts().head(3)
                h2h_modal = top_h2h.index[0]
                h2h_rate = top_h2h.iloc[0] / h2h_n

            print(f"\n  MATCH {i}: {m.team_a} vs {m.team_b}")
            print(f"    Cotes : {m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}")

            # Top 5 V5 score predictions
            if top5:
                print(f"    📊 V5 Top 5 scores Poisson+empirique:")
                for sc, p in top5[:5]:
                    print(f"       {sc} → {p*100:.1f}%")

            # H2H modal
            if h2h_modal:
                print(f"    🔍 H2H : {h2h_n} matchs, score le + fréquent {h2h_modal} ({h2h_rate*100:.0f}%)")
                for sc, count in top_h2h.items():
                    print(f"       {sc} → {count}/{h2h_n} = {count/h2h_n*100:.0f}%")

            # GOLD signal
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                sg = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                if sg["rate"] >= 0.40 and sg["n"] >= 12:
                    print(f"    💎💎 SCORE EXACT GOLD : {sg['score']} arrive {sg['rate']*100:.0f}% (n={sg['n']}) ⭐⭐⭐")
                    all_score_picks.append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                              "score": sg["score"], "rate": sg["rate"], "n": sg["n"],
                                              "tier": "GOLD ⭐⭐⭐"})
                elif sg["rate"] >= 0.33:
                    print(f"    💎 SCORE EXACT BON : {sg['score']} arrive {sg['rate']*100:.0f}% (n={sg['n']})")
                    all_score_picks.append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                              "score": sg["score"], "rate": sg["rate"], "n": sg["n"],
                                              "tier": "BON ⭐⭐"})
                else:
                    print(f"    ⚠️ Score {sg['score']} {sg['rate']*100:.0f}% (n={sg['n']}) — confiance moyenne")

            # Best V5 prediction si rate ≥ 11% (sur cote estimée ~9 → EV positive)
            if top5 and top5[0][1] >= 0.11 and not (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                sc, p = top5[0]
                print(f"    ⭐ V5 BEST : {sc} prob {p*100:.1f}% (cote~{1/p*0.9:.0f}) → value bet")
                all_score_picks.append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                          "score": sc, "rate": p, "n": "V5_model", "tier": "V5 ⭐"})

    # SOMMAIRE
    print(f"\n\n{'═' * 105}")
    print(f"  📋 SOMMAIRE SCORE EXACT — TOUS ROUNDS")
    print(f"{'═' * 105}\n")

    if not all_score_picks:
        print("Aucun pick score exact détecté")
        return 0

    gold = [p for p in all_score_picks if "GOLD" in p["tier"]]
    bon = [p for p in all_score_picks if "BON" in p["tier"]]
    v5 = [p for p in all_score_picks if "V5" in p["tier"]]

    if gold:
        print(f"💎💎💎 SCORE EXACT GOLD (priorité ABSOLUE — rate ≥40%, n≥12):")
        for p in sorted(gold, key=lambda x: -x["rate"]):
            cote_est = 1 / p["rate"] * 0.9
            ev = p["rate"] * cote_est - 1
            print(f"   ⭐⭐⭐ [{p['round']}] {p['match']:<40} {p['score']:<6} {p['rate']*100:.0f}% (n={p['n']}, cote~{cote_est:.0f}, EV~+{ev*100:.0f}%)")
        print()

    if bon:
        print(f"💎 SCORE EXACT BON (rate ≥33%):")
        for p in sorted(bon, key=lambda x: -x["rate"]):
            cote_est = 1 / p["rate"] * 0.9
            ev = p["rate"] * cote_est - 1
            print(f"   ⭐⭐ [{p['round']}] {p['match']:<40} {p['score']:<6} {p['rate']*100:.0f}% (n={p['n']}, cote~{cote_est:.0f})")
        print()

    if v5:
        print(f"⭐ V5 BEST SCORES (top1 ≥11%) :")
        for p in sorted(v5, key=lambda x: -x["rate"])[:15]:
            print(f"   [{p['round']}] {p['match']:<40} {p['score']:<6} {p['rate']*100:.1f}% (modèle V5)")

    # PLAN DE MISE
    print(f"\n💰 PLAN DE MISE (bankroll 100u, mise 0.30-0.50u par pick):")
    total_stake = 0
    for p in gold:
        stake = 0.50
        total_stake += stake
        cote_est = 1/p['rate'] * 0.9
        print(f"   💎 [{p['round']}] {p['match'][:30]:<30} {p['score']} → mise {stake:.2f}u (gain potentiel +{stake*(cote_est-1):.1f}u)")
    for p in bon:
        stake = 0.30
        total_stake += stake
        cote_est = 1/p['rate'] * 0.9
        print(f"   ⭐ [{p['round']}] {p['match'][:30]:<30} {p['score']} → mise {stake:.2f}u (gain potentiel +{stake*(cote_est-1):.1f}u)")
    print(f"\n   Total misé : {total_stake:.2f}u")

    return 0


if __name__ == "__main__":
    sys.exit(main())
