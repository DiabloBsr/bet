"""Prédiction V9 — intègre biais équipe + classement + forme récente.

Améliore V8 en :
1. Détectant les équipes "pièges" qui sous-performent comme home favori (Spurs, Everton, Wolves)
2. Détectant les équipes "solides" sur-performantes (Brighton, C. Palace, Brentford, Burnley)
3. Filtrant par rank difference (ROI +8% quand home moins bien classé mais favori)
4. Filtrant par combo forme (Home 4/5 + Away 4/5 = ROI +12%)

Usage :
  python scripts/predict_v9.py --round 15:55
  python scripts/predict_v9.py --round 15:55 --combo-cote 5
"""
from __future__ import annotations
import argparse, sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v8 import fit_model_v8, predict_v8

MG_TZ = timezone(timedelta(hours=3))

# Biais empiriques par équipe en tant que home favori (mesurés sur 3000+ matchs)
TEAM_HOME_FAV_BIAS = {
    # POSITIFS (over-performent) — à privilégier
    "Brighton": +0.078,
    "C. Palace": +0.077,
    "Brentford": +0.067,
    "Liverpool": +0.038,
    "N. Forest": +0.037,
    "Manchester Red": +0.026,
    "Burnley": +0.096,   # mais peu d'échantillon
    # NEUTRES
    "London Reds": +0.010,
    "Fulham": +0.009,
    "London Blues": +0.007,
    # NÉGATIFS (sous-performent) — à éviter ou éviter cotes basses
    "Bournemouth": -0.010,
    "A. Villa": -0.012,
    "Manchester Blue": -0.013,
    "Newcastle": -0.018,
    "West Ham": -0.021,
    "Spurs": -0.026,
    "Wolverhampton": -0.030,
    "Everton": -0.071,   # PIRE — à JAMAIS prendre comme home favori
}

# Risque "trap" : éviter à cote < 2.0 si bias négatif significatif
TRAP_TEAMS = {"Everton", "Wolverhampton", "Spurs"}


def get_team_bias(team):
    return TEAM_HOME_FAV_BIAS.get(team, 0.0)


def is_trap(team, cote):
    return team in TRAP_TEAMS and cote < 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True)
    ap.add_argument("--combo-cote", type=float, default=None,
                    help="Si fourni, cherche combo cote ~X")
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
    print(f"Loading model (n_train={len(history)})...")
    model = fit_model_v8(history)

    # Latest ranking snapshot
    rk = pd.read_sql("""
        SELECT team_name, position, points, history
        FROM rankings_snapshots
        WHERE captured_at = (SELECT MAX(captured_at) FROM rankings_snapshots)
    """, engine)
    rank_map = {r.team_name: {"position": r.position, "points": r.points, "history": r.history}
                for _, r in rk.iterrows()}

    # Forme récente des équipes (5 derniers matchs)
    # On compte W/L/D par équipe en regardant les matchs récents joués
    recent_matches = pd.read_sql("""
        SELECT e.team_a, e.team_b, r.score_a, r.score_b
        FROM events e
        JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL
        ORDER BY e.expected_start DESC
        LIMIT 500
    """, engine)
    team_form = defaultdict(list)
    for _, m in recent_matches.iterrows():
        if m.score_a > m.score_b:
            team_form[m.team_a].append("W"); team_form[m.team_b].append("L")
        elif m.score_a < m.score_b:
            team_form[m.team_a].append("L"); team_form[m.team_b].append("W")
        else:
            team_form[m.team_a].append("D"); team_form[m.team_b].append("D")

    def form_wins(team, n=5):
        return sum(1 for x in team_form.get(team, [])[:n] if x == "W")

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
        print(f"Round {args.round} introuvable."); return 1

    print(f"\n{'═' * 105}")
    print(f"  V9 — Round {args.round} — {len(matches)} matchs (avec biais équipe + classement + forme)")
    print(f"{'═' * 105}\n")

    candidates = []
    print(f"{'#':<3}{'Match':<32} {'Cotes':<18} {'Rk H-A':<8} {'Form':<10} {'V8 pick':<11} {'Biais':<8} {'V9 verdict'}")
    print("─" * 130)
    for i, (_, m) in enumerate(matches.iterrows(), 1):
        pred = predict_v8(model, m.odds_home, m.odds_draw, m.odds_away, extra_markets=m.extra_markets)
        rk_h = rank_map.get(m.team_a, {}).get("position", "?")
        rk_a = rank_map.get(m.team_b, {}).get("position", "?")
        rk_diff = (rk_h - rk_a) if rk_h != "?" and rk_a != "?" else None
        form_h = form_wins(m.team_a)
        form_a = form_wins(m.team_b)
        bias_h = get_team_bias(m.team_a)

        # V8 best pick
        evs = {o: pred["ev_1x2"][o] for o in ["1", "X", "2"]}
        best_outcome = max(evs, key=evs.get)
        best_ev = evs[best_outcome]
        cote = pred["cotes"][best_outcome]

        # V9 ajustements
        verdict = "—"
        reason = ""
        boost = 0
        if best_outcome == "1" and best_ev > 0.02:
            if is_trap(m.team_a, cote):
                verdict = "❌ TRAP"
                reason = f"({m.team_a} = équipe piège bias{bias_h*100:+.1f}pp)"
            else:
                # Boost selon biais équipe
                adj_ev = best_ev + bias_h
                # Boost selon rank diff
                if rk_diff is not None and rk_diff >= 10:
                    boost += 0.04
                    reason += f"+rank{rk_diff:+d}"
                # Boost selon forme (Home 4/5 + Away 4/5)
                if form_h == 4 and form_a == 4:
                    boost += 0.05; reason += " +hot4/4"
                elif form_h == 5 and form_a == 5:
                    boost -= 0.10; reason += " trop chaud"
                elif form_h <= 1 and form_a >= 3:
                    boost += 0.06; reason += " +reset"
                adj_ev_final = adj_ev + boost
                if adj_ev_final > 0.04:
                    verdict = f"✅ V9 EV{adj_ev_final*100:+.0f}%"
                elif adj_ev_final > 0:
                    verdict = f"⚠️ V9 EV{adj_ev_final*100:+.0f}% (marginal)"
                else:
                    verdict = f"❌ V9 EV{adj_ev_final*100:+.0f}%"
        elif best_outcome == "1":
            verdict = "❌ pas EV"
        else:
            # Non-home pick : on garde V8 sans ajustement biais
            if best_ev > 0.02:
                verdict = f"V8 {best_outcome} EV{best_ev*100:+.0f}%"
            else:
                verdict = "❌"

        # Garder candidats positifs V9
        if verdict.startswith("✅") and best_outcome == "1":
            adj_ev_value = best_ev + bias_h + boost
            adj_p = pred["p_model"][best_outcome] + (bias_h + boost) / 2  # adj approx
            candidates.append({
                "match": f"{m.team_a} vs {m.team_b}",
                "team_a": m.team_a, "team_b": m.team_b,
                "outcome": best_outcome, "cote": cote, "p": adj_p,
                "ev": adj_ev_value, "bias": bias_h, "rk_diff": rk_diff,
                "form_h": form_h, "form_a": form_a, "reason": reason,
            })

        match_str = f"{m.team_a} vs {m.team_b}"[:30]
        cotes_str = f"{m.odds_home:.2f}/{m.odds_draw:.2f}/{m.odds_away:.2f}"
        rk_str = f"{rk_h}-{rk_a}" if rk_diff is not None else "?"
        form_str = f"{form_h}/{form_a}"
        v8_str = f"{best_outcome}@{cote:.2f}({best_ev*100:+.0f}%)"
        bias_str = f"{bias_h*100:+.1f}pp"
        print(f"{i:<3}{match_str:<32} {cotes_str:<18} {rk_str:<8} {form_str:<10} {v8_str:<11} {bias_str:<8} {verdict} {reason}")

    print()
    print(f"=== {len(candidates)} CANDIDATS SAFE V9 ===\n")
    for i, c in enumerate(candidates, 1):
        print(f"  [{i}] {c['match']:<36} {c['outcome']} @{c['cote']:.2f}  EV={c['ev']*100:+.1f}%")
        print(f"      Biais {c['team_a']}={c['bias']*100:+.1f}pp, rank_diff={c['rk_diff']}, forme {c['form_h']}/{c['form_a']}  {c['reason']}")

    # Combo recherche
    if args.combo_cote and candidates:
        target = args.combo_cote
        print(f"\n💡 COMBINAISONS pour cote ~{target}:\n")
        found = []
        for n in [2, 3, 4]:
            for combo in combinations(range(len(candidates)), n):
                picks = [candidates[i] for i in combo]
                total_cote = np.prod([p["cote"] for p in picks])
                total_p = np.prod([p["p"] for p in picks])
                ev = total_cote * total_p - 1
                if target * 0.7 <= total_cote <= target * 1.4:
                    found.append({"picks": picks, "cote": total_cote, "p": total_p, "ev": ev, "n": n})

        found.sort(key=lambda x: -x["ev"])
        for i, combo in enumerate(found[:5], 1):
            print(f"  COMBO {i}  cote {combo['cote']:.2f}  proba {combo['p']*100:.1f}%  EV {combo['ev']*100:+.0f}%")
            for p in combo["picks"]:
                print(f"     • {p['match']:<40} {p['outcome']} @{p['cote']:.2f}  (biais{p['bias']*100:+.1f}pp)")
            print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
