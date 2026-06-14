"""V10 prediction live — synthèse de tous les signaux."""
from __future__ import annotations
import argparse, sys
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from scipy.stats import poisson
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.predictor_v10 import fit_model_v10, predict_v10, _bracket_cote

MG_TZ = timezone(timedelta(hours=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--combo-cote", type=float, default=None)
    args = ap.parse_args()

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
    print(f"Loading V10 model (n_train={len(history)})...")
    model_v5 = fit_model_v5(history, ht_history=history.copy(), engine=engine, form_alpha=0.0)
    model_v10 = fit_model_v10(history)

    # Latest rankings
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
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming.expected_start, utc=True)
    upcoming = upcoming[upcoming.expected_start > now_utc].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    matches = upcoming[upcoming.local == args.round]
    if matches.empty:
        print(f"Round {args.round} introuvable.")
        return 1

    print(f"\n{'═' * 115}")
    print(f"  V10 — Round {args.round} — {len(matches)} matchs (multi-signaux)")
    print(f"{'═' * 115}\n")

    candidates = []
    for i, (_, m) in enumerate(matches.iterrows(), 1):
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                   extra_markets=m.extra_markets)
        rh = rank_map.get(m.team_a); ra = rank_map.get(m.team_b)
        pred10 = predict_v10(model_v10, m.team_a, m.team_b,
                              m.odds_home, m.odds_draw, m.odds_away,
                              extra_markets=m.extra_markets,
                              rank_home=rh, rank_away=ra)

        # Score predictions V5
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
        ht_pick_v5 = pred5.get("ht_pick", "—")
        ft_pick_v5 = pred5.get("primary_pick", "—")
        ft_p = (pred5.get("primary_p") or 0) * 100
        htft = pred5.get("htft_pick") or "—"
        htft_p = (pred5.get("htft_p") or 0) * 100
        p_over_25 = sum(poisson.pmf(k, ft_total) for k in range(3, 10)) if ft_total > 0 else 0
        p_btts = (1 - poisson.pmf(0, lam_h_ft)) * (1 - poisson.pmf(0, lam_a_ft)) if lam_h_ft > 0 else 0

        print(f"━━━ MATCH {i}  {m.team_a} vs {m.team_b}")
        print(f"     Cotes 1X2 : {m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}")
        print(f"     Rang : {rh}-{ra}" if rh and ra else "     Rang : ?")
        print(f"     HT : {ht_pick_v5} (score {ht_score[0]}-{ht_score[1]})    FT : {ft_pick_v5} ({ft_p:.0f}%) score {score_ft}")
        print(f"     Buts : {ft_total:.2f}  P(+2.5)={p_over_25*100:.0f}%  P(BTTS)={p_btts*100:.0f}%    HT/FT : {htft} ({htft_p:.0f}%)")

        # V10 signals per outcome
        for outcome in ["1", "X", "2"]:
            sigs = pred10["signals"][outcome]
            agg = pred10["agg"][outcome]
            ev = pred10["ev_1x2"][outcome]
            conf = pred10["confidence"][outcome]
            cote = pred10["cotes"][outcome]
            p = pred10["p_model"][outcome]
            if not sigs and ev < 0.02: continue

            sig_str = ", ".join(f"{s[0]}({s[1]*100:+.0f}%)" for s in sigs[:4])
            if agg["n_pos"] >= 2:
                badge = "🔥🔥 MULTI-SIGNAL"
            elif agg["n_pos"] == 1 and agg["n_neg"] == 0:
                badge = "✅ 1 signal"
            elif agg["n_neg"] >= 1:
                badge = "❌ BLOQUÉ"
            else:
                badge = "⚪"

            if ev > 0 and agg["n_neg"] == 0:
                print(f"     {badge} {outcome} @{cote:.2f}  EV={ev*100:+.1f}%  conf={conf}/10  signaux: {sig_str}")
                if (agg["n_pos"] >= 2 or (agg["n_pos"] >= 1 and cote < 2.0 and ev > 0.05)):
                    candidates.append({
                        "match": f"{m.team_a} vs {m.team_b}", "outcome": outcome,
                        "cote": cote, "p": p, "ev": ev, "conf": conf,
                        "n_signals": agg["n_pos"],
                    })

        # Spéculatifs V6
        for name, data in pred10["exotics"].items():
            if data["cote"] and data["ev"] and data["ev"] > 0.5:
                print(f"     🎰 {name} @{data['cote']:.0f}  EV+{data['ev']*100:.0f}% spéculatif")
        print()

    # Candidats
    print(f"{'═' * 115}")
    print(f"🎯 CANDIDATS V10 : {len(candidates)} (priorité multi-signaux)")
    print(f"{'═' * 115}\n")
    candidates.sort(key=lambda c: (-c["n_signals"], -c["ev"]))
    for i, c in enumerate(candidates, 1):
        sig_badge = "🔥🔥" if c["n_signals"] >= 2 else "⭐"
        print(f"  {sig_badge} [{i}] {c['match']:<40} {c['outcome']} @{c['cote']:.2f}  EV={c['ev']*100:+.1f}%  conf={c['conf']}/10 ({c['n_signals']} signaux)")

    # Combos
    if len(candidates) >= 2:
        print(f"\n💡 COMBINAISONS :\n")
        found = []
        for n in [2, 3]:
            for combo in combinations(range(len(candidates)), n):
                picks = [candidates[i] for i in combo]
                tc = np.prod([p["cote"] for p in picks])
                tp = np.prod([p["p"] for p in picks])
                ev = tc * tp - 1
                n_multi = sum(1 for p in picks if p["n_signals"] >= 2)
                found.append({"picks": picks, "cote": tc, "p": tp, "ev": ev, "n_multi": n_multi})
        # Tri par n_multi puis EV
        found.sort(key=lambda x: (-x["n_multi"], -x["ev"]))
        for i, c in enumerate(found[:5], 1):
            print(f"  COMBO {i}  cote {c['cote']:.2f}  proba {c['p']*100:.1f}%  EV {c['ev']*100:+.0f}%  ({c['n_multi']} multi-signal)")
            for p in c["picks"]:
                badge = "🔥" if p["n_signals"] >= 2 else "⭐"
                print(f"     {badge} {p['match']:<40} {p['outcome']} @{p['cote']:.2f}")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
