"""Deep analyse multi-rounds — V10 + 5 nouveaux signaux GOLD."""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
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
    BRACKET_GOLD_HOME, BRACKET_GOLD_AWAY, BRACKET_TRAP_HOME,
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD, SCORE_DOMINANT_GOLD,
    bracket_match,
)

MG_TZ = timezone(timedelta(hours=3))


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    history = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    history["ft_o"] = np.where(history.score_a > history.score_b, "1",
                       np.where(history.score_a == history.score_b, "X", "2"))
    print(f"Loading V10 + all GOLD signals (n_train={len(history)})...\n")

    history_ht = history[history.ht_score_a.notna()].copy()
    model_v5 = fit_model_v5(history_ht, ht_history=history_ht.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history_ht)

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
        ORDER BY e.expected_start
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming.expected_start, utc=True)
    upcoming = upcoming[upcoming.expected_start > now_utc].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    rounds = sorted(upcoming.local.unique())
    print(f"Rounds : {rounds}\n")

    all_picks_summary = {"PAIRE_OR": [], "MULTI": [], "OVER": [], "UNDER": [],
                           "BTTS_OUI": [], "BTTS_NON": [], "SCORE": [], "SPEC": []}

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time]
        print(f"\n╔{'═' * 113}╗")
        print(f"║  ⏰ ROUND {round_time} Mada — {len(matches)} matchs  ║")
        print(f"╚{'═' * 113}╝\n")

        round_recs = []

        for i, (_, m) in enumerate(matches.iterrows(), 1):
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            pred10 = predict_v10(model_v10, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)

            if pred5.get("lam_h_ht"):
                lam_h_ht, lam_a_ht = pred5["lam_h_ht"], pred5["lam_a_ht"]
                lam_h_ft = lam_h_ht / model_v5.ht_lambda_ratio
                lam_a_ft = lam_a_ht / model_v5.ht_lambda_ratio
                ht_score = (int(round(lam_h_ht)), int(round(lam_a_ht)))
                ft_total = lam_h_ft + lam_a_ft
            else:
                ht_score = ("?", "?"); ft_total = 0; lam_h_ft = lam_a_ft = 0

            top5 = pred5.get("top5_scores_enriched") or []
            score_ft = top5[0][0] if top5 else "?"
            ft_pick = pred5.get("primary_pick", "—")
            ft_p = (pred5.get("primary_p") or 0) * 100
            ht_pick = pred5.get("ht_pick", "—")
            htft = pred5.get("htft_pick") or "—"
            htft_p = (pred5.get("htft_p") or 0) * 100

            # H2H
            h2h = history[(history.team_a == m.team_a) & (history.team_b == m.team_b)]
            h2h_n = len(h2h)
            h2h_str = "—"
            if h2h_n >= 5:
                w = (h2h.ft_o == "1").sum(); d = (h2h.ft_o == "X").sum(); l = (h2h.ft_o == "2").sum()
                h2h_str = f"n={h2h_n}, {w}W/{d}D/{l}L ({w/h2h_n*100:.0f}% home)"

            # Stats Over/BTTS H2H
            h2h_over = ((h2h.score_a + h2h.score_b) > 2.5).mean()*100 if h2h_n > 0 else None
            h2h_btts = ((h2h.score_a >= 1) & (h2h.score_b >= 1)).mean()*100 if h2h_n > 0 else None

            # Performance Home/Away
            h_team = history[history.team_a == m.team_a]
            a_team = history[history.team_b == m.team_b]
            h_w_rate = (h_team.ft_o == "1").mean()*100 if len(h_team) > 0 else 0
            a_w_rate = (a_team.ft_o == "2").mean()*100 if len(a_team) > 0 else 0

            # Forme récente
            h_form = "".join("W" if r.ft_o == "1" else "L" if r.ft_o == "2" else "D"
                              for _, r in h_team.tail(5).iterrows())
            a_form = "".join("W" if r.ft_o == "2" else "L" if r.ft_o == "1" else "D"
                              for _, r in a_team.tail(5).iterrows())

            print(f"┌─ MATCH {i}  {m.team_a} vs {m.team_b}")
            print(f"│  Cotes 1X2     : {m.odds_home:.2f} / {m.odds_draw:.2f} / {m.odds_away:.2f}")
            print(f"│  H2H           : {h2h_str}", end="")
            if h2h_n >= 5:
                print(f"   Over_H2H={h2h_over:.0f}%  BTTS_H2H={h2h_btts:.0f}%")
            else:
                print()
            print(f"│  Performance   : {m.team_a} home {h_w_rate:.0f}%W  |  {m.team_b} away {a_w_rate:.0f}%W")
            print(f"│  Forme récente : {m.team_a} {h_form}  |  {m.team_b} {a_form}")
            print(f"│  Prédiction    : HT {ht_pick}({ht_score[0]}-{ht_score[1]})  FT {ft_pick}({ft_p:.0f}%)  Score {score_ft}")
            print(f"│  Buts attendus : {ft_total:.2f}    HT/FT : {htft} ({htft_p:.0f}%)")

            recommendations = []

            # PAIRE OR HOME
            if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
                p = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 PAIRE OR HOME : n={p['n']}, {p['win']*100:.0f}% wins, ROI+{p['roi']*100:.0f}%")
                recommendations.append(("PAIRE_OR_HOME", "1", m.odds_home, p['win']))
                all_picks_summary["PAIRE_OR"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                       "pari": "1", "cote": m.odds_home, "win": p['win']})

            # PAIRE OR AWAY
            if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
                p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 PAIRE OR AWAY : n={p['n']}, {p['win']*100:.0f}% wins, ROI+{p['roi']*100:.0f}%")
                recommendations.append(("PAIRE_OR_AWAY", "2", m.odds_away, p['win']))
                all_picks_summary["PAIRE_OR"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                       "pari": "2", "cote": m.odds_away, "win": p['win']})

            # PAIRE TRAP
            if (m.team_a, m.team_b) in PAIR_TRAP_HOME:
                print(f"│  ❌❌ PAIRE TRAP HOME — JAMAIS parier 1")

            # OVER GOLD
            if (m.team_a, m.team_b) in OVER_GOLD:
                og = OVER_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 OVER 2.5 GOLD : {og['rate']*100:.0f}% historique sur n={og['n']}")
                recommendations.append(("OVER", "Over 2.5", None, og['rate']))
                all_picks_summary["OVER"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                    "rate": og['rate'], "n": og['n']})

            # UNDER GOLD
            if (m.team_a, m.team_b) in UNDER_GOLD:
                ug = UNDER_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 UNDER 2.5 GOLD : seulement {ug['over_rate']*100:.0f}% Over, n={ug['n']}")
                recommendations.append(("UNDER", "Under 2.5", None, 1-ug['over_rate']))
                all_picks_summary["UNDER"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                     "rate": 1-ug['over_rate'], "n": ug['n']})

            # BTTS OUI GOLD
            if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
                bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 BTTS OUI GOLD : {bg['rate']*100:.0f}% historique n={bg['n']}")
                recommendations.append(("BTTS_OUI", "BTTS Oui", None, bg['rate']))
                all_picks_summary["BTTS_OUI"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                       "rate": bg['rate'], "n": bg['n']})

            # BTTS NON GOLD
            if (m.team_a, m.team_b) in BTTS_NON_GOLD:
                bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 BTTS NON GOLD : seulement {bn['bts_rate']*100:.0f}% BTTS, n={bn['n']}")
                recommendations.append(("BTTS_NON", "BTTS Non", None, 1-bn['bts_rate']))
                all_picks_summary["BTTS_NON"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                       "rate": 1-bn['bts_rate'], "n": bn['n']})

            # SCORE EXACT GOLD
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                sg = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 SCORE EXACT GOLD : {sg['score']} arrive {sg['rate']*100:.0f}% n={sg['n']}")
                recommendations.append(("SCORE", sg['score'], None, sg['rate']))
                all_picks_summary["SCORE"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                    "score": sg['score'], "rate": sg['rate'], "n": sg['n']})

            # V10 multi-signal
            for outcome in ["1", "X", "2"]:
                sigs = pred10["signals"][outcome]
                agg = pred10["agg"][outcome]
                ev = pred10["ev_1x2"][outcome]
                conf = pred10["confidence"][outcome]
                cote = pred10["cotes"][outcome]
                if agg.get("has_pair_trap"): continue
                if agg["n_pos"] >= 2 and ev > 0.05 and not agg.get("has_pair_gold"):
                    sig_str = ", ".join(s[0] for s in sigs if s[2] in ("+", "++"))[:75]
                    print(f"│  🔥🔥 MULTI-SIGNAL {outcome} @{cote:.2f}  EV={ev*100:+.0f}%  conf={conf}/10")
                    print(f"│       Signaux : {sig_str}")
                    if cote < 3:
                        all_picks_summary["MULTI"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                            "pari": outcome, "cote": cote, "ev": ev, "conf": conf})

            # Spéculatifs HT/FT
            for name, data in pred10["exotics"].items():
                if data["cote"] and data["ev"] and data["ev"] > 1.0:
                    print(f"│  🎰 {name} @{data['cote']:.0f}  EV+{data['ev']*100:.0f}% spéculatif")
                    all_picks_summary["SPEC"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                        "pari": name, "cote": data["cote"], "ev": data["ev"]})

            print(f"└{'─' * 113}")

            if recommendations:
                round_recs.append({"match": f"{m.team_a} vs {m.team_b}", "recs": recommendations})

        # Récap round
        print(f"\n📋 RÉCAP ROUND {round_time}")
        print(f"   {'─' * 105}")
        for r in round_recs:
            print(f"   {r['match']}:")
            for type_, pari, cote, conf in r["recs"]:
                cote_str = f"@{cote:.2f}" if cote else ""
                print(f"      • [{type_}] {pari} {cote_str}  conf={conf*100:.0f}%")
        if not round_recs:
            print(f"   ⚠️ Aucune recommandation forte")
        print()

    # SOMMAIRE GLOBAL
    print(f"\n╔{'═' * 113}╗")
    print(f"║  📊 SOMMAIRE TOUS ROUNDS ║")
    print(f"╚{'═' * 113}╝\n")

    print(f"💎💎 PAIRES OR ({len(all_picks_summary['PAIRE_OR'])} matchs):")
    for p in all_picks_summary['PAIRE_OR']:
        print(f"   ✅ [{p['round']}] {p['match']:<40} {p['pari']} @{p['cote']:.2f}  hist {p['win']*100:.0f}% wins")
    print()
    print(f"💎 OVER 2.5 GOLD ({len(all_picks_summary['OVER'])} matchs):")
    for p in all_picks_summary['OVER']:
        print(f"   ✅ [{p['round']}] {p['match']:<40} Over 2.5  hist {p['rate']*100:.0f}% (n={p['n']})")
    print()
    print(f"💎 UNDER 2.5 GOLD ({len(all_picks_summary['UNDER'])} matchs):")
    for p in all_picks_summary['UNDER']:
        print(f"   ✅ [{p['round']}] {p['match']:<40} Under 2.5  hist {p['rate']*100:.0f}% (n={p['n']})")
    print()
    print(f"💎 BTTS OUI GOLD ({len(all_picks_summary['BTTS_OUI'])} matchs):")
    for p in all_picks_summary['BTTS_OUI']:
        print(f"   ✅ [{p['round']}] {p['match']:<40} BTTS Oui  hist {p['rate']*100:.0f}% (n={p['n']})")
    print()
    print(f"💎 BTTS NON GOLD ({len(all_picks_summary['BTTS_NON'])} matchs):")
    for p in all_picks_summary['BTTS_NON']:
        print(f"   ✅ [{p['round']}] {p['match']:<40} BTTS Non  hist {p['rate']*100:.0f}% (n={p['n']})")
    print()
    print(f"💎 SCORE EXACT GOLD ({len(all_picks_summary['SCORE'])} matchs):")
    for p in all_picks_summary['SCORE']:
        print(f"   ✅ [{p['round']}] {p['match']:<40} Score {p['score']}  hist {p['rate']*100:.0f}% (n={p['n']})")
    print()
    print(f"🔥🔥 MULTI-SIGNAL ({len(all_picks_summary['MULTI'])} matchs):")
    for p in sorted(all_picks_summary['MULTI'], key=lambda x: -x["conf"])[:10]:
        print(f"   ✅ [{p['round']}] {p['match']:<40} {p['pari']} @{p['cote']:.2f}  EV+{p['ev']*100:.0f}% conf {p['conf']}/10")
    print()
    print(f"🎰 SPÉCULATIFS TOP 10 EV (mise 0.10u):")
    for p in sorted(all_picks_summary['SPEC'], key=lambda x: -x["ev"])[:10]:
        print(f"   🎰 [{p['round']}] {p['match']:<40} {p['pari']:<20} @{p['cote']:.0f}  EV+{p['ev']*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
