"""Prédiction COMBO SCORE TOP 2 (stratégie validée 24% accuracy)."""
from __future__ import annotations
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.team_gold_data import (
    SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD,
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD, OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD,
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
    print(f"V5 chargé (n={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)

    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    h_all = h_all.drop_duplicates(["team_a", "team_b", "score_a", "score_b"], keep="last").copy()
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
    rounds = sorted(upcoming.local.unique())[:6]
    print(f"Now Mada: {now_utc.astimezone(MG_TZ).strftime('%H:%M:%S')}")
    print(f"Rounds : {rounds}\n")

    all_combos = []
    all_singles = []
    all_v5 = []

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time].head(10)
        print(f"\n{'═' * 110}")
        print(f"  ⏰ ROUND {round_time} — {len(matches)} matchs")
        print(f"{'═' * 110}")

        for i, (_, m) in enumerate(matches.iterrows(), 1):
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            top5 = pred5.get("top5_scores_enriched") or []

            print(f"\n  ━━ MATCH {i}  {m.team_a} vs {m.team_b}  ({m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f})")

            # COMBO GOLD (priorité absolue)
            if (m.team_a, m.team_b) in SCORE_COMBO_GOLD:
                c = SCORE_COMBO_GOLD[(m.team_a, m.team_b)]
                print(f"    💎💎 COMBO TOP 2 : {c['top1']} ({c['r1']*100:.0f}%) + {c['top2']} ({c['r2']*100:.0f}%) = {c['combo']*100:.0f}% combo")
                all_combos.append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                     "score1": c['top1'], "r1": c['r1'],
                                     "score2": c['top2'], "r2": c['r2'],
                                     "combo": c['combo'], "n": c['n']})

            # Single GOLD (rate 30-34% = sweet spot validé)
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                s = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                # Filtrer : ne prendre que 30-34% (sweet spot)
                if 0.30 <= s['rate'] <= 0.39:
                    print(f"    💎 SCORE SOLO : {s['score']} {s['rate']*100:.0f}% (n={s['n']}) — SWEET SPOT")
                    if (m.team_a, m.team_b) not in SCORE_COMBO_GOLD:
                        all_singles.append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                              "score": s['score'], "rate": s['rate'], "n": s['n']})
                elif s['rate'] >= 0.40:
                    print(f"    ⚠️  Score {s['score']} {s['rate']*100:.0f}% (overfit possible, attention)")

            # Top 3 V5 fallback (si pas dans GOLD)
            if (m.team_a, m.team_b) not in SCORE_COMBO_GOLD and (m.team_a, m.team_b) not in SCORE_DOMINANT_GOLD:
                if top5:
                    print(f"    ⭐ V5 Top 3 : ", end="")
                    for s, p in top5[:3]:
                        print(f"{s}({p*100:.0f}%) ", end="")
                    print()
                    if top5[0][1] >= 0.12:
                        all_v5.append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                         "score": top5[0][0], "rate": top5[0][1]})

            # H2H récent
            h2h = h_all[(h_all.team_a == m.team_a) & (h_all.team_b == m.team_b)]
            if len(h2h) >= 8:
                top_h2h = Counter(h2h.score).most_common(3)
                print(f"    🔍 H2H récents : ", end="")
                for s, c in top_h2h:
                    print(f"{s}({c}/{len(h2h)}={c/len(h2h)*100:.0f}%) ", end="")
                print()

    # SOMMAIRE
    print(f"\n\n{'═' * 110}")
    print(f"  📋 SOMMAIRE — STRATÉGIE COMBO TOP 2")
    print(f"{'═' * 110}\n")

    if all_combos:
        print(f"💎💎 COMBO GOLD ({len(all_combos)} matchs) — priorité ABSOLUE:")
        for c in sorted(all_combos, key=lambda x: -x['combo']):
            print(f"   [{c['round']}] {c['match']:<40} {c['score1']}({c['r1']*100:.0f}%) + {c['score2']}({c['r2']*100:.0f}%) = {c['combo']*100:.0f}% combo (n={c['n']})")
        print()

    if all_singles:
        print(f"💎 SCORE SWEET SPOT ({len(all_singles)} matchs, rate 30-39%):")
        for s in all_singles:
            print(f"   [{s['round']}] {s['match']:<40} {s['score']:<6} {s['rate']*100:.0f}% (n={s['n']})")
        print()

    if all_v5:
        print(f"⭐ V5 TOP picks (≥12% modèle):")
        for v in sorted(all_v5, key=lambda x: -x['rate'])[:10]:
            print(f"   [{v['round']}] {v['match']:<40} {v['score']:<6} {v['rate']*100:.1f}%")

    # Plan de mise
    if all_combos:
        print(f"\n💰 PLAN DE MISE COMBO (sur 100u bankroll):")
        total = 0
        total_gain = 0
        for c in all_combos:
            stake_per_score = 0.30  # 0.30u par score
            total += stake_per_score * 2  # 2 scores
            # Gain combiné estimé (cote moyenne 8 pour score 30-50%)
            cote_avg = 7
            gain_if_win = stake_per_score * (cote_avg - 1)
            ev_gain = c['combo'] * gain_if_win - (1 - c['combo']) * stake_per_score
            total_gain += ev_gain * 2
            print(f"   💎 [{c['round']}] {c['match'][:30]:<30} {c['score1']} + {c['score2']} → 0.30u × 2 = 0.60u (1 sur 2 chances de win)")
        print(f"\n   Total misé : {total:.2f}u")
        print(f"   Gain attendu (théorique) : +{total_gain:.2f}u")

    return 0


if __name__ == "__main__":
    sys.exit(main())
