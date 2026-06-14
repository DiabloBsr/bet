"""Portfolio V8 — raisonne comme un parieur professionnel.

Pour un round (10 matchs) :
  1. Analyse chaque match (V8 = calibration + exotiques + cote movement)
  2. Génère pour chaque match une JUSTIFICATION narrative
  3. Construit un PORTEFEUILLE diversifié :
       - 1 pick sécurisé (cote 1.4-1.8, signal calibration)
       - 1-2 picks modérés (cote 1.8-3, exotiques HT-X ou outsider)
       - 1-2 picks spéculatifs (cote 30-100, HT/FT comebacks)
       - 0-1 pari combo (si 2 picks sécurisés avec p > 65%)
  4. Allocation Kelly fractionnaire avec cap par catégorie
  5. Risk management : stop-loss, cap total par round

Le système EXPLIQUE chaque décision comme un parieur expérimenté.

Usage :
  python scripts/portfolio_v8.py --round 14:00 --bankroll 100
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v8 import fit_model_v8, predict_v8

MG_TZ = timezone(timedelta(hours=3))


def fmt_pct(x): return f"{x*100:.1f}%"


def categorize_pick(pick_type, cote, ev):
    """Catégorise pour diversification du portefeuille."""
    if pick_type.startswith("1X2"):
        if cote < 1.8: return "SÉCURISÉ"
        if cote < 3.0: return "MODÉRÉ"
        return "SPÉCULATIF"
    if pick_type.startswith("HT/FT"): return "SPÉCULATIF" if cote > 20 else "MODÉRÉ"
    if pick_type.startswith("Score"): return "SPÉCULATIF"
    return "MODÉRÉ"


def narrative_for_pick(team_a, team_b, pred, pick_type, pick_data):
    """Génère un raisonnement narratif comme un parieur."""
    lines = []
    cote = pick_data["cote"]
    p = pick_data["p"]
    ev = pick_data["ev"]

    if pick_type.startswith("1X2"):
        outcome = pick_type.split()[-1]
        team = team_a if outcome == "1" else (team_b if outcome == "2" else "nul")
        p_market = pred["p_market"][outcome]
        p_model = pred["p_model"][outcome]
        delta = p_model - p_market

        if outcome == "1":
            lines.append(f"Je parie HOME ({team_a}) — cote {cote:.2f}.")
            if cote < 1.8 and p > 0.60:
                lines.append(f"  Le favori cote 1.6-1.8 a une dynamique empirique de gain à {fmt_pct(p)} (vs marché {fmt_pct(p_market)}, +{delta*100:.1f}pp).")
                lines.append(f"  C'est un pari de fond : pas glorieux mais le bookmaker sous-cote ce bracket précis.")
            elif cote > 3.0:
                lines.append(f"  Le favori cote 4-5 (outsider modéré home) gagne à {fmt_pct(p)} alors que la cote dit {fmt_pct(p_market)}.")
                lines.append(f"  L'avantage du terrain est sous-évalué quand l'équipe home n'est pas favorite.")
            else:
                lines.append(f"  Probabilité empirique : {fmt_pct(p)}, marché : {fmt_pct(p_market)}, edge {delta*100:+.1f}pp.")
        elif outcome == "2":
            lines.append(f"Je parie AWAY ({team_b}) — cote {cote:.2f}.")
            lines.append(f"  L'away cote 2.5-2.9 (semi-favori) gagne à {fmt_pct(p)} vs marché {fmt_pct(p_market)}.")
        else:
            lines.append(f"Je parie NUL — cote {cote:.2f}.")

        if pred.get("movement") and pred["movement"].get("boost"):
            boost_name, boost_val = pred["movement"]["boost"]
            lines.append(f"  ⚡ BOOST : mouvement de cote détecté ({boost_name}, +{boost_val*100:.0f}pp ajustement)")

    elif pick_type.startswith("HT/FT"):
        side = pick_type.split()[-1]
        if side == "1/2":
            lines.append(f"Je parie HT/FT 1/2 ({team_a} mène à HT, {team_b} gagne au final) — cote {cote:.2f}.")
            lines.append(f"  C'est un pari spéculatif sur un comeback de l'AWAY ({team_b}).")
            lines.append(f"  Le bookmaker price ça à {fmt_pct(1/cote)} mais l'empirique sur 3400+ matchs montre {fmt_pct(p)}.")
            lines.append(f"  Edge brut : {(p - 1/cote)*100:+.1f}pp en notre faveur.")
        elif side == "2/1":
            lines.append(f"Je parie HT/FT 2/1 ({team_b} mène à HT, {team_a} gagne au final) — cote {cote:.2f}.")
            lines.append(f"  Comeback du HOME ({team_a}) attendu, p_emp {fmt_pct(p)} vs bookmaker {fmt_pct(1/cote)}.")
    elif pick_type.startswith("Score 1-0"):
        lines.append(f"Je parie Score 1-0 ({team_a} 1, {team_b} 0) — cote {cote:.2f}.")
        lines.append(f"  Cas typique : {team_b} est favori mais {team_a} l'accroche 1-0 (avantage du terrain).")
        lines.append(f"  Probabilité empirique de ce score précis quand l'away est favori : {fmt_pct(p)}.")

    lines.append(f"  → EV {ev*100:+.0f}%, confiance {pick_data.get('confidence', '?')}/10")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", required=True, help="HH:MM Mada")
    ap.add_argument("--bankroll", type=float, default=100.0)
    ap.add_argument("--max-picks", type=int, default=5)
    ap.add_argument("--ev-min-1x2", type=float, default=0.02)
    ap.add_argument("--ev-min-exotic", type=float, default=0.50)
    ap.add_argument("--max-stake-pct", type=float, default=0.10, help="Cap mise totale par round en % bankroll")
    args = ap.parse_args()

    settings = load_settings()
    engine = create_engine(settings.db_url)

    history = pd.read_sql("""
        SELECT o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.ht_score_a IS NOT NULL
    """, engine)
    model = fit_model_v8(history)

    now_utc = datetime.now(timezone.utc)
    upcoming = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.expected_start, e.id as ev_id,
               last_o.odds_home, last_o.odds_draw, last_o.odds_away, last_o.extra_markets,
               first_o.odds_home as first_oh, first_o.odds_draw as first_od, first_o.odds_away as first_oa,
               (SELECT COUNT(*) FROM odds_snapshots WHERE event_id = e.id) as n_snaps
        FROM events e
        JOIN odds_snapshots last_o  ON last_o.id  = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN odds_snapshots first_o ON first_o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        WHERE r.id IS NULL
        ORDER BY e.expected_start
    """, engine)
    upcoming["expected_start"] = pd.to_datetime(upcoming["expected_start"], utc=True, errors="coerce")
    upcoming = upcoming[upcoming.expected_start.notna() & (upcoming.expected_start > now_utc)].copy()
    upcoming["local"] = upcoming.expected_start.dt.tz_convert(MG_TZ).dt.strftime("%H:%M")
    upcoming = upcoming[upcoming.local == args.round]
    if upcoming.empty:
        print(f"Round {args.round} introuvable."); return 1

    print(f"\n{'═' * 100}")
    print(f"  PORTEFEUILLE V8 — Round {args.round} Mada — {len(upcoming)} matchs analysés")
    print(f"  Bankroll {args.bankroll:.0f}u — Cap par round : {args.max_stake_pct*100:.0f}% ({args.max_stake_pct*args.bankroll:.1f}u max)")
    print(f"{'═' * 100}\n")

    candidates = []
    for _, m in upcoming.iterrows():
        first_oh = m.first_oh if m.n_snaps >= 2 else None
        first_od = m.first_od if m.n_snaps >= 2 else None
        first_oa = m.first_oa if m.n_snaps >= 2 else None
        pred = predict_v8(model, m.odds_home, m.odds_draw, m.odds_away,
                            extra_markets=m.extra_markets,
                            first_odds_home=first_oh,
                            first_odds_draw=first_od,
                            first_odds_away=first_oa)
        # Collect all positive-EV options
        for outcome in ["1", "X", "2"]:
            ev = pred["ev_1x2"][outcome]
            if ev >= args.ev_min_1x2:
                candidates.append({
                    "match": f"{m.team_a} vs {m.team_b}",
                    "team_a": m.team_a, "team_b": m.team_b,
                    "type": f"1X2 {outcome}", "cote": pred["cotes"][outcome],
                    "p": pred["p_model"][outcome], "ev": ev,
                    "category": categorize_pick(f"1X2 {outcome}", pred["cotes"][outcome], ev),
                    "pred": pred,
                })
        for name, data in pred["exotics"].items():
            if data["cote"] is None or data["ev"] is None: continue
            if data["ev"] >= args.ev_min_exotic:
                candidates.append({
                    "match": f"{m.team_a} vs {m.team_b}",
                    "team_a": m.team_a, "team_b": m.team_b,
                    "type": name, "cote": data["cote"],
                    "p": data["p_emp"], "ev": data["ev"],
                    "category": categorize_pick(name, data["cote"], data["ev"]),
                    "pred": pred,
                })

    if not candidates:
        print("❌  Aucune opportunité détectée sur ce round.")
        return 0

    # Compute confidence /10
    for c in candidates:
        is_exotic = not c["type"].startswith("1X2")
        base = c["p"] * 10 if not is_exotic else c["p"] * 30
        bonus = min(c["ev"] * 5, 3)
        c["confidence"] = min(round(base + bonus, 1), 10)

    # === Sélection portfolio diversifié ===
    # Critère : 1 SÉCURISÉ + 1-2 MODÉRÉ + 1-2 SPÉCULATIF
    # Et : pas 2 picks sur le même match
    selected = []
    used_matches = set()
    counts = {"SÉCURISÉ": 0, "MODÉRÉ": 0, "SPÉCULATIF": 0}
    caps = {"SÉCURISÉ": 1, "MODÉRÉ": 2, "SPÉCULATIF": 2}

    # Sort by ev*confidence
    candidates.sort(key=lambda c: -(c["ev"] * c["confidence"]))
    for c in candidates:
        if c["match"] in used_matches: continue
        if counts[c["category"]] >= caps[c["category"]]: continue
        selected.append(c)
        used_matches.add(c["match"])
        counts[c["category"]] += 1
        if sum(counts.values()) >= args.max_picks: break

    # Allocation Kelly avec cap
    total_stake_cap = args.max_stake_pct * args.bankroll
    raw_stakes = []
    for c in selected:
        b = c["cote"] - 1
        kelly = (b * c["p"] - (1 - c["p"])) / b if b > 0 else 0
        # Kelly fractionnaire selon catégorie
        frac = {"SÉCURISÉ": 0.25, "MODÉRÉ": 0.12, "SPÉCULATIF": 0.05}[c["category"]]
        raw_stakes.append(max(0, kelly * frac))
    sum_raw = sum(raw_stakes) or 1
    # Si total dépasse cap, scale down
    if sum_raw * args.bankroll > total_stake_cap:
        scale = total_stake_cap / (sum_raw * args.bankroll)
        raw_stakes = [s * scale for s in raw_stakes]

    print(f"📋 {len(selected)} PARIS SÉLECTIONNÉS (diversifiés sur 3 catégories)\n")

    total_stake = total_expected = 0
    for i, (c, stake_frac) in enumerate(zip(selected, raw_stakes), 1):
        stake = stake_frac * args.bankroll
        gain_if_win = stake * (c["cote"] - 1)
        ev_gain = c["ev"] * stake
        total_stake += stake
        total_expected += ev_gain

        cat_icon = {"SÉCURISÉ": "🟢", "MODÉRÉ": "🟡", "SPÉCULATIF": "🔴"}[c["category"]]
        print(f"━━━ PARI #{i}  [{cat_icon} {c['category']}]  Confiance {c['confidence']}/10")
        print(f"     Match : {c['match']}")
        print(f"     Pari  : {c['type']}  @{c['cote']:.2f}")
        for line in narrative_for_pick(c["team_a"], c["team_b"], c["pred"], c["type"], c):
            print(f"     {line}")
        print(f"     Mise  : {stake:>5.2f}u   Gain si OK : +{gain_if_win:>5.2f}u   EV : +{ev_gain:>5.2f}u")
        print()

    print(f"{'═' * 100}")
    print(f"📊 SYNTHÈSE DU PORTEFEUILLE")
    print(f"   Total misé        : {total_stake:>6.2f}u / {args.bankroll:.0f}u ({total_stake/args.bankroll*100:.1f}%)")
    print(f"   Gain attendu (EV) : +{total_expected:>5.2f}u")
    if total_stake > 0:
        print(f"   ROI attendu       : +{total_expected/total_stake*100:.1f}%")
    print(f"   Répartition       : 🟢 {counts['SÉCURISÉ']} sécurisé(s) | 🟡 {counts['MODÉRÉ']} modéré(s) | 🔴 {counts['SPÉCULATIF']} spéculatif(s)")
    print()
    print(f"⚠️  RISK MANAGEMENT")
    print(f"   • Variance attendue : élevée sur les paris spéculatifs (acc ~4%)")
    print(f"   • Drawdown possible : 30-50 paris perdants consécutifs sont normaux")
    print(f"   • Stop-loss conseillé : si bankroll perd 20% (-{args.bankroll*0.20:.0f}u), pause de 5 rounds")
    print(f"   • Stop-win conseillé : si bankroll +30% (+{args.bankroll*0.30:.0f}u), retire 50% des gains")
    print(f"   • Ne JAMAIS combler une perte en augmentant les mises (Martingale = ruine)")
    print(f"{'═' * 100}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
