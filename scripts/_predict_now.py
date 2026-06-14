"""V10 prédiction tous rounds disponibles à partir de MAINTENANT."""
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
    print(f"V10 FINAL chargé (n_train={len(history)})\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    history_all = pd.read_sql("""
        SELECT e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    history_all["ft_o"] = np.where(history_all.score_a > history_all.score_b, "1",
                            np.where(history_all.score_a == history_all.score_b, "X", "2"))

    now_utc = datetime.now(timezone.utc)
    # Dédupliquer : 1 entrée par (team_a, team_b, expected_start)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               e.id as ev_id
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL AND e.expected_start IS NOT NULL
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming.expected_start, utc=True)
    upcoming = upcoming[upcoming.expected_start > now_utc].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    # Garde 1 entrée par (team_a, team_b, local)
    upcoming = upcoming.sort_values("ev_id", ascending=False).drop_duplicates(["team_a", "team_b", "local"])
    rounds = sorted(upcoming.local.unique())
    print(f"Rounds détectés : {rounds}\n")

    summary = {"PAIRE_OR": [], "MULTI": [], "OVER": [], "UNDER": [], "BTTS_OUI": [], "BTTS_NON": [], "SCORE": [], "SPEC": []}

    for round_time in rounds[:8]:
        matches = upcoming[upcoming.local == round_time].head(10)  # max 10 matchs/round
        print(f"\n╔{'═' * 113}╗")
        print(f"║  ⏰ ROUND {round_time}  —  {len(matches)} matchs  ║")
        print(f"╚{'═' * 113}╝\n")

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
            top3 = " · ".join(f"{s}({p*100:.0f}%)" for s, p in top5[:3])
            ft_pick = pred5.get("primary_pick", "—")
            ft_p = (pred5.get("primary_p") or 0) * 100
            ht_pick = pred5.get("ht_pick", "—")
            htft = pred5.get("htft_pick") or "—"
            htft_p = (pred5.get("htft_p") or 0) * 100

            h2h = history_all[(history_all.team_a == m.team_a) & (history_all.team_b == m.team_b)]
            h2h_str = "—"
            if len(h2h) >= 5:
                w = (h2h.ft_o == "1").sum(); d = (h2h.ft_o == "X").sum(); l = (h2h.ft_o == "2").sum()
                h2h_str = f"n={len(h2h)}, {w}W/{d}D/{l}L"

            print(f"┌─ MATCH {i}  {m.team_a} vs {m.team_b}")
            print(f"│  Cotes : {m.odds_home:.2f} / {m.odds_draw:.2f} / {m.odds_away:.2f}")
            print(f"│  H2H   : {h2h_str}")
            print(f"│  HT : {ht_pick}({ht_score[0]}-{ht_score[1]})  FT : {ft_pick}({ft_p:.0f}%) score {score_ft}")
            print(f"│  Top 3 : {top3}")
            print(f"│  Buts : {ft_total:.2f}  HT/FT : {htft} ({htft_p:.0f}%)")

            # SIGNAUX
            recs = []
            if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
                p = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 PAIRE OR HOME : {p['win']*100:.0f}% wins n={p['n']} ROI+{p['roi']*100:.0f}%")
                summary["PAIRE_OR"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                              "pari": "1", "cote": m.odds_home, "win": p['win']})
            if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
                p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
                # CORRECTIF : filtrer cote actuelle ≤ cote moyenne × factor
                max_factor = p.get('max_cote_factor', 1.05)
                if m.odds_away <= p['cote'] * max_factor:
                    print(f"│  💎 PAIRE OR AWAY : {p['win']*100:.0f}% n={p['n']} (cote {m.odds_away:.2f} ≤ seuil {p['cote']*max_factor:.2f}) → OK")
                    summary["PAIRE_OR"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                  "pari": "2", "cote": m.odds_away, "win": p['win']})
                else:
                    print(f"│  ⚠️ PAIRE OR AWAY mais cote {m.odds_away:.2f} > seuil {p['cote']*max_factor:.2f} (bookmaker corrige) → SKIP")
            if (m.team_a, m.team_b) in PAIR_TRAP_HOME:
                print(f"│  ❌❌ PAIRE TRAP HOME")
            if (m.team_a, m.team_b) in OVER_GOLD:
                og = OVER_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 OVER 2.5 GOLD : {og['rate']*100:.0f}% n={og['n']}")
                summary["OVER"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": og['rate'], "n": og['n']})
            if (m.team_a, m.team_b) in UNDER_GOLD:
                ug = UNDER_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 UNDER 2.5 GOLD : {(1-ug['over_rate'])*100:.0f}% n={ug['n']}")
                summary["UNDER"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": 1-ug['over_rate'], "n": ug['n']})
            if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
                bg = BTTS_OUI_GOLD[(m.team_a, m.team_b)]
                # CORRECTIF : seulement si cote home >= 1.8 (sinon favori écrase)
                min_cote_h = bg.get('min_cote_h', 1.8)
                if m.odds_home >= min_cote_h:
                    print(f"│  💎 BTTS OUI GOLD : {bg['rate']*100:.0f}% n={bg['n']} (cote home {m.odds_home:.2f} ≥ {min_cote_h}) → OK")
                    summary["BTTS_OUI"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": bg['rate'], "n": bg['n']})
                else:
                    print(f"│  ⚠️ BTTS OUI mais cote home {m.odds_home:.2f} < {min_cote_h} (favori écraserait) → SKIP")
            if (m.team_a, m.team_b) in BTTS_NON_GOLD:
                bn = BTTS_NON_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 BTTS NON GOLD : {(1-bn['bts_rate'])*100:.0f}% n={bn['n']}")
                summary["BTTS_NON"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "rate": 1-bn['bts_rate'], "n": bn['n']})
            if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
                sg = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
                if sg['rate'] >= 0.40 and sg['n'] >= 12:
                    print(f"│  💎 SCORE EXACT GOLD : {sg['score']} {sg['rate']*100:.0f}% n={sg['n']}")
                    summary["SCORE"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}", "score": sg['score'], "rate": sg['rate']})
                else:
                    print(f"│  ⚠️ Score {sg['score']} {sg['rate']*100:.0f}% mais n={sg['n']} → faible confiance")

            # V10 multi-signal
            for outcome in ["1", "X", "2"]:
                sigs = pred10["signals"][outcome]
                agg = pred10["agg"][outcome]
                ev = pred10["ev_1x2"][outcome]
                conf = pred10["confidence"][outcome]
                cote = pred10["cotes"][outcome]
                if agg.get("has_pair_trap"): continue
                if agg["n_pos"] >= 2 and ev > 0.05 and not agg.get("has_pair_gold"):
                    sig_str = ", ".join(s[0] for s in sigs if s[2] in ("+", "++"))[:60]
                    print(f"│  🔥🔥 MULTI {outcome} @{cote:.2f} EV+{ev*100:.0f}% conf {conf}/10  ({sig_str})")
                    if cote < 3:
                        summary["MULTI"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                  "pari": outcome, "cote": cote, "ev": ev, "conf": conf})

            for name, data in pred10["exotics"].items():
                if data["cote"] and data["ev"] and data["ev"] > 1.0:
                    print(f"│  🎰 {name} @{data['cote']:.0f}  EV+{data['ev']*100:.0f}%")
                    if data["ev"] > 1.5:
                        summary["SPEC"].append({"round": round_time, "match": f"{m.team_a} vs {m.team_b}",
                                                  "pari": name, "cote": data["cote"], "ev": data["ev"]})

            print(f"└{'─' * 113}")

    # Sommaire
    print(f"\n╔{'═' * 113}╗")
    print(f"║  📋 SOMMAIRE ALL ROUNDS  ║")
    print(f"╚{'═' * 113}╝\n")
    for cat, items in [("PAIRE OR", summary["PAIRE_OR"]),
                         ("MULTI-SIGNAL", summary["MULTI"]),
                         ("OVER 2.5 GOLD", summary["OVER"]),
                         ("UNDER 2.5 GOLD", summary["UNDER"]),
                         ("BTTS OUI", summary["BTTS_OUI"]),
                         ("BTTS NON", summary["BTTS_NON"]),
                         ("SCORE EXACT", summary["SCORE"])]:
        if items:
            print(f"\n💎 {cat} ({len(items)} picks):")
            for p in items:
                if "cote" in p:
                    print(f"   [{p['round']}] {p['match']:<40} pari {p['pari']} @{p['cote']:.2f}  hist {p.get('win', p.get('rate', 0))*100:.0f}%")
                elif "score" in p:
                    print(f"   [{p['round']}] {p['match']:<40} score {p['score']}  hist {p['rate']*100:.0f}%")
                else:
                    print(f"   [{p['round']}] {p['match']:<40} hist {p['rate']*100:.0f}% (n={p.get('n', '?')})")

    if summary["SPEC"]:
        print(f"\n🎰 SPÉCULATIFS TOP 10 EV (mise 0.10u):")
        for p in sorted(summary["SPEC"], key=lambda x: -x["ev"])[:10]:
            print(f"   [{p['round']}] {p['match']:<40} {p['pari']:<20} @{p['cote']:.0f}  EV+{p['ev']*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
