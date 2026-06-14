"""Prédiction COMPLÈTE tous rounds futurs : 1X2, score, HT, O/U, BTTS, combos."""
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
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD,
    SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD,
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
    print(f"Modèles chargés (n={len(history)} matchs)")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    h_all = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    h_all = h_all.drop_duplicates(["team_a", "team_b", "score_a", "score_b"], keep="last").copy()
    h_all["ft_o"] = np.where(h_all.score_a > h_all.score_b, "1",
                       np.where(h_all.score_a == h_all.score_b, "X", "2"))
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
    rounds = sorted(upcoming.local.unique())
    print(f"Now Mada: {now_utc.astimezone(MG_TZ).strftime('%H:%M:%S')}")
    print(f"Rounds futurs : {rounds}\n")

    summary = {
        "PAIRE_OR_HOME": [], "PAIRE_OR_AWAY": [], "MULTI": [],
        "OVER": [], "UNDER": [], "BTTS_OUI": [], "BTTS_NON": [],
        "COMBO_SCORE": [], "SCORE_SWEET": [], "V5_SCORE": [],
    }

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time].head(10)
        print(f"\n{'═' * 105}")
        print(f"  ⏰ ROUND {round_time} — {len(matches)} matchs")
        print(f"{'═' * 105}")

        for i, (_, m) in enumerate(matches.iterrows(), 1):
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            pred10 = predict_v10(model_v10, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)

            top5 = pred5.get("top5_scores_enriched") or []
            score_ft = top5[0][0] if top5 else "?"
            ft_pick = pred5.get("primary_pick", "—")
            ft_p = (pred5.get("primary_p") or 0) * 100
            ht_pick = pred5.get("ht_pick", "—")

            # Lambdas pour O/U
            if pred5.get("lam_h_ht"):
                lam_h_ht = pred5["lam_h_ht"]; lam_a_ht = pred5["lam_a_ht"]
                lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
                lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
                lam_total = lam_h_ft + lam_a_ft
                ht_score = (int(round(lam_h_ht)), int(round(lam_a_ht)))
                p_over_15 = 1 - poisson.cdf(1, lam_total)
                p_over_25 = 1 - poisson.cdf(2, lam_total)
                p_under_35 = poisson.cdf(3, lam_total)
                p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft))
            else:
                lam_total = 0
                ht_score = (0, 0)
                p_over_15 = p_over_25 = p_under_35 = p_btts = 0

            print(f"\n  ━━ MATCH {i}  {m.team_a} vs {m.team_b}  ({m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f})")
            print(f"     HT: {ht_pick}({ht_score[0]}-{ht_score[1]})  FT: {ft_pick}({ft_p:.0f}%) {score_ft}  Buts: {lam_total:.1f}")

            # SIGNAUX
            # 1. PAIRE OR HOME
            if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
                p = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
                print(f"     💎 PAIRE OR HOME : {p['win']*100:.0f}% (n={p['n']}, ROI+{p['roi']*100:.0f}%)")
                summary["PAIRE_OR_HOME"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                  "cote": m.odds_home, "win": p['win'], "n": p['n']})
            # 2. PAIRE OR AWAY (avec filtre)
            if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
                p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
                if m.odds_away <= p["cote"] * p.get("max_cote_factor", 1.05):
                    print(f"     💎 PAIRE OR AWAY : {p['win']*100:.0f}% (n={p['n']}, cote {m.odds_away:.2f} OK)")
                    summary["PAIRE_OR_AWAY"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                      "cote": m.odds_away, "win": p['win']})
            # 3. PAIRE TRAP
            if (m.team_a, m.team_b) in PAIR_TRAP_HOME:
                print(f"     ❌❌ PAIRE TRAP HOME — REFUSER 1")

            # 4. OVER/UNDER GOLD
            if (m.team_a, m.team_b) in OVER_GOLD:
                og = OVER_GOLD[(m.team_a, m.team_b)]
                print(f"     💎 OVER 2.5 GOLD : {og['rate']*100:.0f}% (n={og['n']})")
                summary["OVER"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": og['rate']})
            if (m.team_a, m.team_b) in UNDER_GOLD:
                ug = UNDER_GOLD[(m.team_a, m.team_b)]
                print(f"     💎 UNDER 2.5 GOLD : {(1-ug['over_rate'])*100:.0f}% (n={ug['n']})")
                summary["UNDER"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": 1-ug['over_rate']})

            # 5. BTTS
            if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
                bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
                if m.odds_home >= bg.get('min_cote_h', 1.8):
                    print(f"     💎 BTTS OUI : {bg['rate']*100:.0f}% (n={bg['n']})")
                    summary["BTTS_OUI"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": bg['rate']})
            if (m.team_a, m.team_b) in BTTS_NON_GOLD:
                bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
                print(f"     💎 BTTS NON : {(1-bn['bts_rate'])*100:.0f}% (n={bn['n']})")
                summary["BTTS_NON"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": 1-bn['bts_rate']})

            # 6. SCORE COMBO
            if (m.team_a, m.team_b) in SCORE_COMBO_GOLD:
                c = SCORE_COMBO_GOLD[(m.team_a, m.team_b)]
                print(f"     💎💎 COMBO SCORE : {c['top1']}({c['r1']*100:.0f}%) + {c['top2']}({c['r2']*100:.0f}%) = {c['combo']*100:.0f}%")
                summary["COMBO_SCORE"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                "s1": c['top1'], "s2": c['top2'], "combo": c['combo']})
            # 7. SCORE SWEET SPOT (30-44%)
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD and (m.team_a, m.team_b) not in SCORE_COMBO_GOLD:
                s = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                if 0.30 <= s['rate'] <= 0.44:
                    print(f"     💎 SCORE SWEET : {s['score']} {s['rate']*100:.0f}% (n={s['n']})")
                    summary["SCORE_SWEET"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                    "score": s['score'], "rate": s['rate']})

            # 8. Multi-signal V10
            for outcome in ["1", "X", "2"]:
                agg = pred10["agg"][outcome]
                ev = pred10["ev_1x2"][outcome]
                conf = pred10["confidence"][outcome]
                cote = pred10["cotes"][outcome]
                if agg.get("has_pair_trap"): continue
                if agg["n_pos"] >= 2 and ev > 0.05 and not agg.get("has_pair_gold") and cote < 3:
                    print(f"     🔥🔥 MULTI {outcome}@{cote:.2f} EV+{ev*100:.0f}% conf {conf}/10")
                    summary["MULTI"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                              "outcome": outcome, "cote": cote, "ev": ev, "conf": conf})

            # 9. V5 score top si rien d'autre
            if (m.team_a, m.team_b) not in SCORE_COMBO_GOLD and \
               (m.team_a, m.team_b) not in SCORE_DOMINANT_GOLD and top5:
                if top5[0][1] >= 0.12:
                    summary["V5_SCORE"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                 "score": top5[0][0], "rate": top5[0][1]})

    # SOMMAIRE FINAL
    print(f"\n\n{'═' * 105}")
    print(f"  📋 SOMMAIRE FINAL — TOUS ROUNDS")
    print(f"{'═' * 105}\n")

    if summary["PAIRE_OR_HOME"]:
        print(f"💎 PAIRES OR HOME ({len(summary['PAIRE_OR_HOME'])}):")
        for p in summary["PAIRE_OR_HOME"]:
            print(f"   [{p['round']}] {p['match']:<40} 1 @{p['cote']:.2f}  ({p['win']*100:.0f}% hist, n={p['n']})")
        print()

    if summary["PAIRE_OR_AWAY"]:
        print(f"💎 PAIRES OR AWAY ({len(summary['PAIRE_OR_AWAY'])}):")
        for p in summary["PAIRE_OR_AWAY"]:
            print(f"   [{p['round']}] {p['match']:<40} 2 @{p['cote']:.2f}  ({p['win']*100:.0f}% hist)")
        print()

    if summary["MULTI"]:
        print(f"🔥 MULTI-SIGNAL ({len(summary['MULTI'])}):")
        for p in sorted(summary["MULTI"], key=lambda x: -x['conf']):
            print(f"   [{p['round']}] {p['match']:<40} {p['outcome']} @{p['cote']:.2f}  EV+{p['ev']*100:.0f}% conf {p['conf']}/10")
        print()

    if summary["OVER"]:
        print(f"💎 OVER 2.5 GOLD ({len(summary['OVER'])}):")
        for p in summary["OVER"]:
            print(f"   [{p['round']}] {p['match']:<40} Over 2.5  ({p['rate']*100:.0f}%)")
        print()

    if summary["UNDER"]:
        print(f"💎 UNDER 2.5 GOLD ({len(summary['UNDER'])}):")
        for p in summary["UNDER"]:
            print(f"   [{p['round']}] {p['match']:<40} Under 2.5  ({p['rate']*100:.0f}%)")
        print()

    if summary["BTTS_OUI"]:
        print(f"💎 BTTS OUI ({len(summary['BTTS_OUI'])}):")
        for p in summary["BTTS_OUI"]:
            print(f"   [{p['round']}] {p['match']:<40} BTTS Oui  ({p['rate']*100:.0f}%)")
        print()

    if summary["BTTS_NON"]:
        print(f"💎 BTTS NON ({len(summary['BTTS_NON'])}):")
        for p in summary["BTTS_NON"]:
            print(f"   [{p['round']}] {p['match']:<40} BTTS Non  ({p['rate']*100:.0f}%)")
        print()

    if summary["COMBO_SCORE"]:
        print(f"💎💎 SCORE COMBO TOP 2 ({len(summary['COMBO_SCORE'])}):")
        for p in sorted(summary["COMBO_SCORE"], key=lambda x: -x['combo']):
            print(f"   [{p['round']}] {p['match']:<40} {p['s1']} + {p['s2']}  ({p['combo']*100:.0f}%)")
        print()

    if summary["SCORE_SWEET"]:
        print(f"💎 SCORE SWEET SPOT ({len(summary['SCORE_SWEET'])}):")
        for p in summary["SCORE_SWEET"]:
            print(f"   [{p['round']}] {p['match']:<40} {p['score']}  ({p['rate']*100:.0f}%)")
        print()

    if summary["V5_SCORE"]:
        print(f"⭐ V5 SCORE TOP ({len(summary['V5_SCORE'])}):")
        for p in sorted(summary["V5_SCORE"], key=lambda x: -x['rate'])[:10]:
            print(f"   [{p['round']}] {p['match']:<40} {p['score']}  V5 {p['rate']*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
