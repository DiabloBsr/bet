"""Prédictions SCORE EXACT pour les rounds à venir.

Stratégies :
- TOP 1 : Score le plus probable du modèle (10-15% accuracy)
- COMBO TOP 2 : Top 1 + Top 2 (~20-25% accuracy)
- COMBO TOP 3 : Top 1 + Top 2 + Top 3 (~30-35% accuracy)
- SCORE_COMBO_GOLD historique : si paire historiquement >60% combo
- SCORE_DOMINANT : score qui domine 30-44% (sweet spot)
"""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.team_gold_data import SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD

MG_TZ = timezone(timedelta(hours=3))
TARGETS = ["02:01"]


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

    best_picks = []

    for TARGET in TARGETS:
        round_matchs = upcoming[upcoming.local == TARGET]
        if len(round_matchs) == 0:
            continue
        print(f"\n{'═'*108}")
        print(f"  ⏰ ROUND {TARGET} — SCORES EXACTS")
        print(f"{'═'*108}")
        print(f"  {'MATCH':<42} {'TOP1':<10} {'TOP2':<10} {'TOP3':<10} {'COMBO':<10} {'SIGNAUX'}")
        print(f"  {'-'*108}")

        for _, m in round_matchs.iterrows():
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                      extra_markets=m.extra_markets)
            top5 = pred5.get("top5_scores_enriched") or []
            if not top5:
                continue
            top1 = top5[0]
            top2 = top5[1] if len(top5) > 1 else (("?-?"), 0)
            top3 = top5[2] if len(top5) > 2 else (("?-?"), 0)
            top4 = top5[3] if len(top5) > 3 else (("?-?"), 0)
            combo_top3 = top1[1] + top2[1] + top3[1]
            combo_top2 = top1[1] + top2[1]
            combo_top4 = combo_top3 + top4[1]

            signals = []
            # SCORE_COMBO_GOLD
            if (m.team_a, m.team_b) in SCORE_COMBO_GOLD:
                c = SCORE_COMBO_GOLD[(m.team_a, m.team_b)]
                if c.get('n', 0) >= 8:
                    signals.append(f"💎 GOLD {c['top1']}+{c['top2']}={c['combo']*100:.0f}%(n={c['n']})")
            # SCORE_DOMINANT (sweet spot 30-44%)
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                sd = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                if 0.30 <= sd['rate'] <= 0.44:
                    signals.append(f"💎 DOM {sd['score']}={sd['rate']*100:.0f}%")

            match_lbl = f"{m.team_a} vs {m.team_b}"[:40]
            print(f"  {match_lbl:<42} {top1[0]}({top1[1]*100:.0f}%)  {top2[0]}({top2[1]*100:.0f}%)  {top3[0]}({top3[1]*100:.0f}%)  T3:{combo_top3*100:.0f}%  {' '.join(signals)}")

            # Stocker pour récap
            best_picks.append({
                "round": TARGET, "match": match_lbl,
                "top1_score": top1[0], "top1_p": top1[1],
                "top2_score": top2[0], "top2_p": top2[1],
                "top3_score": top3[0], "top3_p": top3[1],
                "combo_top2": combo_top2, "combo_top3": combo_top3,
                "has_gold": (m.team_a, m.team_b) in SCORE_COMBO_GOLD,
                "is_dominant": (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD,
            })

    # ============ TOP PICKS ============
    print(f"\n\n{'═'*108}")
    print(f"  🎯 TOP PICKS SCORE EXACT — par stratégie")
    print(f"{'═'*108}\n")

    df = pd.DataFrame(best_picks)
    if df.empty:
        print("Aucune prédiction.")
        return 0

    # TOP 1 single
    print("  📊 TOP 1 SINGLE (mise simple, ~10-15% accuracy attendu) :")
    top1_sorted = df.sort_values("top1_p", ascending=False).head(10)
    for _, r in top1_sorted.iterrows():
        marker = " 💎" if r.has_gold else ""
        print(f"     {r['round']}  {r['match']:<42}  → {r['top1_score']}  ({r['top1_p']*100:.0f}%){marker}")

    print()
    print("  📊 COMBO TOP 2 (parier 2 scores, ~20-25% accuracy attendu) :")
    df_sorted = df.sort_values("combo_top2", ascending=False).head(10)
    for _, r in df_sorted.iterrows():
        marker = " 💎" if r.has_gold else ""
        print(f"     {r['round']}  {r['match']:<42}  → {r['top1_score']} + {r['top2_score']}  ({r['combo_top2']*100:.0f}%){marker}")

    print()
    print("  📊 COMBO TOP 3 (parier 3 scores, ~30-35% accuracy attendu) :")
    df_sorted = df.sort_values("combo_top3", ascending=False).head(10)
    for _, r in df_sorted.iterrows():
        marker = " 💎" if r.has_gold else ""
        print(f"     {r['round']}  {r['match']:<42}  → {r['top1_score']} + {r['top2_score']} + {r['top3_score']}  ({r['combo_top3']*100:.0f}%){marker}")

    print()
    print("  💎 SCORE_COMBO_GOLD (paires historiques validées) :")
    gold = df[df.has_gold]
    if len(gold) > 0:
        for _, r in gold.iterrows():
            print(f"     {r['round']}  {r['match']:<42}  → voir signal GOLD ci-dessus")
    else:
        print("     (aucune paire GOLD sur ces rounds)")

    print()
    print("  💎 SCORE DOMINANT (rate 30-44%, sweet spot) :")
    dom = df[df.is_dominant]
    if len(dom) > 0:
        for _, r in dom.iterrows():
            print(f"     {r['round']}  {r['match']}")
    else:
        print("     (aucune paire DOMINANT)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
