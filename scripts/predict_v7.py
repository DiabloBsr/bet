"""Prédiction V7 — synthèse : 1X2 affiné + value bets sur tous marchés.

Pour chaque match :
  • Probas vig-free du marché (1, X, 2)
  • Probas du modèle (corrigées via calibration empirique)
  • Probabilité la PLUS PROBABLE = recommandation 'prédiction pure'
  • Pour chaque marché (1, X, 2, HT/FT, scores) : EV calculée
  • Recommandation pari : marché avec EV positive maximale

Le système :
  - NE suit PAS le favori par défaut
  - NE le combat PAS par principe
  - Calcule l'EV de TOUT et recommande ce qui est value

Usage :
  python scripts/predict_v7.py --round 13:30
  python scripts/predict_v7.py --team-a "X" --team-b "Y" --cote-h 1.5 --cote-d 4 --cote-a 6
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v7 import fit_model_v7, predict_v7

MG_TZ = timezone(timedelta(hours=3))


def fmt_pct(x): return f"{x*100:.1f}%"


def value_level(ev):
    if ev is None: return "—"
    if ev < 0: return "❌ négative"
    if ev < 0.05: return "⚠️  faible"
    if ev < 0.20: return "🟡 modérée"
    if ev < 0.60: return "🟢 forte"
    return "🔥 très forte"


def confidence_score(p_model, ev, is_exotic):
    """Score /10 basé sur proba × edge."""
    if ev is None or ev < 0: return 0
    base = p_model * 10 if not is_exotic else p_model * 30
    bonus = min(ev * 5, 3)
    return min(round(base + bonus, 1), 10)


def analyze(model, team_a, team_b, oh, od, oa, em=None):
    p = predict_v7(model, oh, od, oa, em)

    print(f"\n{'━' * 95}")
    print(f"  {team_a}  vs  {team_b}")
    print(f"  Cotes : 1={oh:.2f}   X={od:.2f}   2={oa:.2f}")
    print(f"{'━' * 95}")

    # SECTION 1 : Probabilités des 3 issues
    print("\n📊 PROBABILITÉS DES 3 ISSUES (vraies probas, pas juste favori)")
    print(f"   ┌─────────┬──────────┬──────────┬──────────┬──────────┐")
    print(f"   │  Issue  │  Cote    │ p_market │ p_modèle │ EV       │")
    print(f"   ├─────────┼──────────┼──────────┼──────────┼──────────┤")
    for outcome in ["1", "X", "2"]:
        cote = p["cotes"][outcome]
        pm = p["p_market"][outcome]
        pmd = p["p_model"][outcome]
        ev = p["ev_1x2"][outcome]
        delta = pmd - pm
        delta_str = f"({delta*100:+.1f}pp)" if abs(delta) > 0.005 else "         "
        # Marquer si EV positive
        ev_marker = "✓" if ev > 0.02 else " "
        print(f"   │   {outcome:<4} │  {cote:>5.2f}   │  {pm*100:>5.1f}%  │  {pmd*100:>5.1f}% {delta_str:<10}│ {ev*100:>+5.1f}% {ev_marker}│")
    print(f"   └─────────┴──────────┴──────────┴──────────┴──────────┘")

    # Issue la plus probable
    p_model_dict = p["p_model"]
    most_likely = max(p_model_dict, key=p_model_dict.get)
    p_ml = p_model_dict[most_likely]
    print(f"\n   📌 Issue la PLUS PROBABLE : {most_likely}  ({fmt_pct(p_ml)} d'après modèle)")
    second = sorted(p_model_dict.items(), key=lambda kv: -kv[1])[1]
    print(f"   📌 2ème option           : {second[0]}  ({fmt_pct(second[1])})")
    print(f"   📌 Outsider              : {sorted(p_model_dict.items(), key=lambda kv: kv[1])[0][0]}  ({fmt_pct(sorted(p_model_dict.values())[0])})")

    # SECTION 2 : Marchés exotiques
    print("\n💎 MARCHÉS EXOTIQUES (où les vrais edges existent)")
    exotic_lines = []
    for name, data in p["exotics"].items():
        if data["cote"] is None: continue
        ev = data["ev"]
        if ev is None: continue
        marker = "✓" if ev > 0.20 else (" " if ev > -0.1 else "✗")
        exotic_lines.append({
            "name": name, "cote": data["cote"], "p": data["p_emp"], "ev": ev, "marker": marker
        })
    if not exotic_lines:
        print("   Aucun marché exotique disponible.")
    else:
        for x in sorted(exotic_lines, key=lambda l: -l["ev"]):
            p_str = f"{x['p']*100:.2f}%" if x['p'] else "—"
            print(f"   {x['marker']} {x['name']:<20}  cote {x['cote']:>5.1f}   p_emp {p_str:>7}   EV {x['ev']*100:>+6.1f}%   {value_level(x['ev'])}")

    # SECTION 3 : Recommandations finales (toutes options confondues)
    print("\n🎯 RECOMMANDATIONS (toutes options confondues, classées par EV)")
    candidates = []
    for outcome in ["1", "X", "2"]:
        ev = p["ev_1x2"][outcome]
        if ev > 0.02:  # seuil 2% pour 1X2 (overround à dépasser)
            candidates.append({
                "type": f"1X2 {outcome}", "cote": p["cotes"][outcome],
                "p": p["p_model"][outcome], "ev": ev,
                "confidence": confidence_score(p["p_model"][outcome], ev, False),
            })
    for x in exotic_lines:
        if x["ev"] > 0.20:
            candidates.append({
                "type": x["name"], "cote": x["cote"],
                "p": x["p"], "ev": x["ev"],
                "confidence": confidence_score(x["p"], x["ev"], True),
            })

    if not candidates:
        print("   ❌  AUCUNE OPPORTUNITÉ INTÉRESSANTE (EV trop faible partout)")
        print(f"       RECOMMANDATION : PASSER ce match.")
        print(f"       L'issue la plus probable reste {most_likely} mais sans value.")
        return

    candidates.sort(key=lambda c: -c["ev"])
    for i, c in enumerate(candidates[:5]):
        rank = "✅" if i == 0 else "  "
        type_label = c["type"]
        is_1x2 = type_label.startswith("1X2")
        type_tag = "SÉCURISÉ" if (is_1x2 and c["cote"] < 2.5) else ("VALUE OUTSIDER" if c["cote"] > 5 else "OPPORTUNISTE")
        print(f"   {rank} {type_label:<20}  @{c['cote']:>5.2f}   p={c['p']*100:>5.1f}%   EV {c['ev']*100:>+5.1f}%   confiance {c['confidence']}/10   [{type_tag}]")

    # Plan de mise sur le meilleur
    best = candidates[0]
    b = best["cote"] - 1
    kelly = (b * best["p"] - (1 - best["p"])) / b if b > 0 else 0
    mise = max(0, min(kelly * 100 / 8, 5))
    print(f"\n   💰 Plan de mise (Kelly 1/8, bankroll 100u) : {mise:.2f}u sur {best['type']} → gain potentiel +{mise*b:.2f}u")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-a"); ap.add_argument("--team-b")
    ap.add_argument("--cote-h", type=float); ap.add_argument("--cote-d", type=float); ap.add_argument("--cote-a", type=float)
    ap.add_argument("--round", default=None)
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
    model = fit_model_v7(history)
    print(f"\n=== V7 PRÉDICTION FINE — modèle n={model.n_train} ===")
    print(f"   Brackets 1X2 calibrés : {len(model.p_h_calibration)} home, {len(model.p_a_calibration)} away")
    print(f"   Signaux exotiques : HT/FT 1/2 ({model.p_12_global*100:.2f}%), HT/FT 2/1 ({model.p_21_global*100:.2f}%), Score 1-0 ({model.p_1_0_when_away_fav*100:.2f}%)")

    if args.team_a and args.team_b and args.cote_h:
        analyze(model, args.team_a, args.team_b, args.cote_h, args.cote_d, args.cote_a)
        return 0

    if args.round:
        now_utc = datetime.now(timezone.utc)
        upcoming = pd.read_sql("""
            SELECT e.team_a, e.team_b, e.expected_start,
                   o.odds_home, o.odds_draw, o.odds_away, o.extra_markets
            FROM events e
            JOIN odds_snapshots o ON o.id = (SELECT MAX(id) FROM odds_snapshots WHERE event_id = e.id)
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
        for _, m in upcoming.iterrows():
            analyze(model, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away, em=m.extra_markets)
        return 0

    print("Usage : --round HH:MM   OU   --team-a X --team-b Y --cote-h ... --cote-d ... --cote-a ...")
    return 1


if __name__ == "__main__":
    sys.exit(main())
