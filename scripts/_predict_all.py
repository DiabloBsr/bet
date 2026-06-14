"""V10 FINAL — prédiction détaillée TOUS rounds à venir."""
from __future__ import annotations
import sys
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.predictor_v10 import fit_model_v10, predict_v10
from scraper.team_gold_data import PAIR_HOME_GOLD, PAIR_AWAY_GOLD, PAIR_TRAP_HOME

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
    print(f"Loading V10 FINAL (n_train={len(history)})...\n")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    rk = pd.read_sql("""
        SELECT team_name, position FROM rankings_snapshots
        WHERE captured_at = (SELECT MAX(captured_at) FROM rankings_snapshots)
    """, engine)
    rank_map = {r.team_name: r.position for _, r in rk.iterrows()}

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
    print(f"Rounds détectés : {rounds}\n")

    all_top_picks = []   # consolidated

    for round_time in rounds:
        matches = upcoming[upcoming.local == round_time]
        print(f"\n╔{'═' * 113}╗")
        print(f"║  ⏰ ROUND {round_time}  —  {len(matches)} matchs  ║")
        print(f"╚{'═' * 113}╝\n")

        round_picks_safe = []
        round_picks_pair = []
        round_picks_spec = []
        round_traps = []

        for i, (_, m) in enumerate(matches.iterrows(), 1):
            pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                       extra_markets=m.extra_markets)
            rh = rank_map.get(m.team_a); ra = rank_map.get(m.team_b)
            pred10 = predict_v10(model_v10, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets, rank_home=rh, rank_away=ra)

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
            ft_pick_v5 = pred5.get("primary_pick", "—")
            ft_p = (pred5.get("primary_p") or 0) * 100
            ht_pick_v5 = pred5.get("ht_pick", "—")
            htft = pred5.get("htft_pick") or "—"
            htft_p = (pred5.get("htft_p") or 0) * 100
            p_over_25 = sum(poisson.pmf(k, ft_total) for k in range(3, 10)) if ft_total > 0 else 0
            p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft)) if lam_h_ft > 0 else 0

            is_pair_gold_h = (m.team_a, m.team_b) in PAIR_HOME_GOLD
            is_pair_gold_a = (m.team_a, m.team_b) in PAIR_AWAY_GOLD
            is_pair_trap = (m.team_a, m.team_b) in PAIR_TRAP_HOME

            print(f"┌─ MATCH {i}  {m.team_a} vs {m.team_b}")
            print(f"│  Cotes 1X2 : {m.odds_home:.2f} / {m.odds_draw:.2f} / {m.odds_away:.2f}    Rang : {rh}-{ra}" if rh else f"│  Cotes 1X2 : {m.odds_home:.2f} / {m.odds_draw:.2f} / {m.odds_away:.2f}")
            print(f"│  HT  : {ht_pick_v5} (prédit {ht_score[0]}-{ht_score[1]})")
            print(f"│  FT  : {ft_pick_v5} ({ft_p:.0f}%)  Score modal : {score_ft}")
            print(f"│  Top 3 scores : {top3}")
            print(f"│  Buts attendus : {ft_total:.2f}    +2.5 : {p_over_25*100:.0f}%   BTTS : {p_btts*100:.0f}%")
            print(f"│  HT/FT : {htft} ({htft_p:.0f}%)")

            if is_pair_gold_h:
                pd_ = PAIR_HOME_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 PAIRE OR HOME (validé n={pd_['n']}) : Win {pd_['win']*100:.0f}%, ROI +{pd_['roi']*100:.0f}%")
                round_picks_pair.append({"match": f"{m.team_a} vs {m.team_b}", "outcome": "1",
                                          "cote": m.odds_home, "type": "PAIRE_OR_HOME",
                                          "win_hist": pd_['win'], "roi_hist": pd_['roi'], "n": pd_['n']})
            if is_pair_gold_a:
                pd_ = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
                print(f"│  💎 PAIRE OR AWAY (validé n={pd_['n']}) : Win {pd_['win']*100:.0f}%, ROI +{pd_['roi']*100:.0f}%")
                round_picks_pair.append({"match": f"{m.team_a} vs {m.team_b}", "outcome": "2",
                                          "cote": m.odds_away, "type": "PAIRE_OR_AWAY",
                                          "win_hist": pd_['win'], "roi_hist": pd_['roi'], "n": pd_['n']})
            if is_pair_trap:
                print(f"│  ❌❌ PAIRE TRAP DÉTECTÉE — ne PAS parier 1 (historique 0-15% wins)")
                round_traps.append(f"{m.team_a} vs {m.team_b}")

            # V10 multi-signal
            for outcome in ["1", "X", "2"]:
                sigs = pred10["signals"][outcome]
                agg = pred10["agg"][outcome]
                ev = pred10["ev_1x2"][outcome]
                conf = pred10["confidence"][outcome]
                cote = pred10["cotes"][outcome]
                p_model = pred10["p_model"][outcome]
                if agg.get("has_pair_trap"): continue
                if agg["n_pos"] >= 2 and ev > 0.05 and not agg.get("has_pair_gold"):
                    sig_str = ", ".join(s[0] for s in sigs if s[2] in ("+", "++"))[:80]
                    print(f"│  🔥🔥 MULTI-SIGNAL {outcome} @{cote:.2f}  EV={ev*100:+.0f}%  conf={conf}/10")
                    print(f"│       Signaux : {sig_str}")
                    if cote < 3.0:
                        round_picks_safe.append({"match": f"{m.team_a} vs {m.team_b}", "outcome": outcome,
                                                  "cote": cote, "type": "MULTI",
                                                  "ev": ev, "conf": conf, "p": p_model})

            # Spéculatifs
            for name, data in pred10["exotics"].items():
                if data["cote"] and data["ev"] and data["ev"] > 0.7:
                    print(f"│  🎰 {name} @{data['cote']:.0f}  EV+{data['ev']*100:.0f}%")
                    if data["ev"] > 1.0:
                        round_picks_spec.append({"match": f"{m.team_a} vs {m.team_b}", "outcome": name,
                                                  "cote": data["cote"], "ev": data["ev"], "p": data["p_emp"]})
            print(f"└{'─' * 113}")

        # Récap par round
        print(f"\n📋 RÉCAP ROUND {round_time}")
        print(f"   {'─' * 100}")
        if round_picks_pair:
            print(f"   💎 Paires OR à parier ({len(round_picks_pair)}):")
            for p in round_picks_pair:
                hist = f"hist {p['win_hist']*100:.0f}% wins, ROI+{p['roi_hist']*100:.0f}% sur {p['n']} matchs"
                print(f"      • {p['match']:<40} {p['outcome']} @{p['cote']:.2f}  ({hist})")
        if round_picks_safe:
            print(f"   🔥 Multi-signal ({len(round_picks_safe)}):")
            for p in sorted(round_picks_safe, key=lambda x: -x["conf"]):
                print(f"      • {p['match']:<40} {p['outcome']} @{p['cote']:.2f}  EV+{p['ev']*100:.0f}% conf {p['conf']}/10")
        if round_picks_spec:
            print(f"   🎰 Spéculatifs EV>100% ({len(round_picks_spec)}):")
            for p in sorted(round_picks_spec, key=lambda x: -x["ev"])[:3]:
                print(f"      • {p['match']:<40} {p['outcome']:<20} @{p['cote']:.0f}  EV+{p['ev']*100:.0f}%")
        if round_traps:
            print(f"   ❌ Traps évités : {', '.join(round_traps)}")

        # Suggestion mise du round
        if round_picks_pair or round_picks_safe or round_picks_spec:
            print(f"\n   💰 SUGGESTION MISE ROUND {round_time} (sur 100u) :")
            total_stake = 0; total_expected_gain = 0
            for p in round_picks_pair:
                stake = 2.0
                total_stake += stake
                exp_gain = p["win_hist"] * (p["cote"] - 1) * stake - (1 - p["win_hist"]) * stake
                total_expected_gain += exp_gain
                print(f"      💎 {p['outcome']}@{p['cote']:.2f} pour {p['match'][:30]:<30} → mise {stake:.2f}u (gain probable +{exp_gain:.2f}u)")
            for p in round_picks_safe[:2]:
                stake = 1.0
                total_stake += stake
                exp_gain = p["p"] * (p["cote"] - 1) * stake - (1 - p["p"]) * stake
                total_expected_gain += exp_gain
                print(f"      🔥 {p['outcome']}@{p['cote']:.2f} pour {p['match'][:30]:<30} → mise {stake:.2f}u (gain probable +{exp_gain:.2f}u)")
            for p in sorted(round_picks_spec, key=lambda x: -x["ev"])[:2]:
                stake = 0.10
                total_stake += stake
                exp_gain = p["p"] * (p["cote"] - 1) * stake - (1 - p["p"]) * stake
                total_expected_gain += exp_gain
                print(f"      🎰 {p['outcome']:<15} pour {p['match'][:30]:<30} → mise {stake:.2f}u (lottery +{stake*(p['cote']-1):.1f}u si OK)")
            print(f"\n      → Total round : misé {total_stake:.2f}u  gain attendu {total_expected_gain:+.2f}u")
        else:
            print(f"\n   ⚠️ AUCUN PICK SAFE ce round — SKIP recommandé")

        # Stock all
        all_top_picks.extend([{**p, "round": round_time} for p in round_picks_pair])
        all_top_picks.extend([{**p, "round": round_time} for p in round_picks_safe])
        all_top_picks.extend([{**p, "round": round_time} for p in round_picks_spec])
        print()

    # SOMMAIRE FINAL
    print(f"\n╔{'═' * 113}╗")
    print(f"║  📋 SOMMAIRE TOUS ROUNDS — Total {len(all_top_picks)} picks recommandés sur {len(rounds)} rounds")
    print(f"╚{'═' * 113}╝\n")

    pair_picks = [p for p in all_top_picks if p.get("type", "").startswith("PAIRE")]
    safe_picks = [p for p in all_top_picks if p.get("type") == "MULTI"]
    spec_picks = [p for p in all_top_picks if "ev" in p and p.get("type") not in ("MULTI", "PAIRE_OR_HOME", "PAIRE_OR_AWAY")]

    print(f"💎💎 {len(pair_picks)} PAIRES OR :  TIER 1 priority absolue")
    print(f"🔥🔥 {len(safe_picks)} MULTI-SIG : TIER 2 multi-signal validation")
    print(f"🎰 {len(spec_picks)} SPÉCULATIFS : TIER 3 lottery EV>100%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
