"""Re-calibrage léger à partir des résultats observés du round 13:22.

Ajustements ciblés (pas un retrain complet, juste un fine-tuning) :
1. Élargir top3 score predictions aux scores extrêmes (4-0, 5-0, 3-3...)
2. Réduire confiance PAIRE OR pour paires "ouvertes" (Fulham vs L. Blues)
3. Détecter pattern "away favori cote<2.0 → high scoring"
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    full = pd.read_sql("""
        SELECT e.team_a, e.team_b, o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    full["ft_o"] = np.where(full.score_a > full.score_b, "1",
                    np.where(full.score_a == full.score_b, "X", "2"))
    full["total"] = full.score_a + full.score_b
    full["score"] = full.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

    print(f"BDD : {len(full)} matchs")
    print()

    # ============ 1. Quelle est la VRAIE distribution top 5 scores quand cote_h = 1.79 ? ============
    # (similaire à Brighton vs Man Red)
    print("=" * 105)
    print("1️⃣  Distribution scores réelle pour cote_home ~1.79 (similaire Brighton vs ManRed)")
    print("=" * 105)
    sub = full[(full.odds_home >= 1.6) & (full.odds_home < 2.0) &
                (full.odds_away >= 3.5) & (full.odds_away < 4.5)]
    if len(sub) >= 30:
        top10 = Counter(sub.score).most_common(10)
        total = len(sub)
        print(f"\n  Sur {total} matchs similaires :")
        for s, c in top10:
            print(f"    {s:<6} : {c}/{total} = {c/total*100:.1f}%")
        # Quelle est la freq de 4-0 et 4-1 ?
        n_4_0 = (sub.score == "4-0").sum()
        n_4_1 = (sub.score == "4-1").sum()
        n_5_0 = (sub.score == "5-0").sum()
        n_3_0 = (sub.score == "3-0").sum()
        n_big = sub[(sub.score_a >= 3) & (sub.score_b <= 1)].shape[0]
        print(f"\n  Scores 'gros'  : 4-0={n_4_0}, 4-1={n_4_1}, 5-0={n_5_0}, 3-0={n_3_0}")
        print(f"  Total home win 3+ buts d'écart : {n_big} ({n_big/total*100:.0f}%)")

    # ============ 2. Y a-t-il un pattern "away favori cote<2.0 → high scoring" ? ============
    print()
    print("=" * 105)
    print("2️⃣  Pattern 'Away favori cote 1.5-2.0 → high scoring' ?")
    print("=" * 105)
    away_fav = full[(full.odds_away < full.odds_home) & (full.odds_away < full.odds_draw) &
                     (full.odds_away >= 1.5) & (full.odds_away < 2.0)]
    if len(away_fav) >= 30:
        print(f"\n  Sur {len(away_fav)} matchs away favori cote 1.5-2.0 :")
        print(f"    Over 2.5 : {(away_fav.total > 2.5).mean()*100:.1f}%")
        print(f"    Over 3.5 : {(away_fav.total > 3.5).mean()*100:.1f}%")
        print(f"    Under 2.5: {(away_fav.total <= 2.5).mean()*100:.1f}%")
        print(f"    Total moy : {away_fav.total.mean():.2f} buts")

    # Comparaison : home favori dans même bracket
    home_fav = full[(full.odds_home < full.odds_away) & (full.odds_home < full.odds_draw) &
                     (full.odds_home >= 1.5) & (full.odds_home < 2.0)]
    print(f"\n  Comparaison home favori cote 1.5-2.0 (n={len(home_fav)}) :")
    print(f"    Over 2.5 : {(home_fav.total > 2.5).mean()*100:.1f}%")
    print(f"    Over 3.5 : {(home_fav.total > 3.5).mean()*100:.1f}%")
    print(f"    Total moy : {home_fav.total.mean():.2f} buts")

    # ============ 3. Sous-estimation des NULS — quand cote_X est basse ? ============
    print()
    print("=" * 105)
    print("3️⃣  Quand X (nul) arrive-t-il plus souvent que prévu ?")
    print("=" * 105)
    # Cote_draw 3.0-3.8 = matchs équilibrés
    bracket_x = full[(full.odds_draw >= 3.0) & (full.odds_draw < 3.8)]
    if len(bracket_x) > 30:
        rate_x = (bracket_x.ft_o == "X").mean()
        cote_avg = bracket_x.odds_draw.mean()
        implied = 1 / cote_avg * 100
        print(f"\n  Cote X ∈ [3.0;3.8] : n={len(bracket_x)}")
        print(f"  Taux X réel    : {rate_x*100:.1f}%")
        print(f"  Taux implicite : {implied:.1f}% (cote moyenne {cote_avg:.2f})")
        if rate_x * 100 > implied + 2:
            print(f"  → ⭐ Nul est SOUS-COTÉ dans ce bracket (parier X = value)")

    # ============ 4. Fulham vs London Blues spécifiquement ============
    print()
    print("=" * 105)
    print("4️⃣  Historique Fulham vs London Blues — était-ce vraiment PAIRE OR ?")
    print("=" * 105)
    pair = full[(full.team_a == "Fulham") & (full.team_b == "London Blues")]
    print(f"\n  {len(pair)} matchs historiques :")
    if len(pair) > 0:
        for _, r in pair.iterrows():
            print(f"    {r.score} (cote home {r.odds_home:.2f})")
        win_rate = (pair.ft_o == "1").mean()
        avg_goals = pair.total.mean()
        x_rate = (pair.ft_o == "X").mean()
        print(f"\n  Fulham win rate : {win_rate*100:.0f}%")
        print(f"  X (nul) rate    : {x_rate*100:.0f}%")
        print(f"  Buts moyens     : {avg_goals:.2f}")
        if x_rate >= 0.30 or avg_goals >= 3.0:
            print(f"  → ⚠️  Paire 'OUVERTE' : nuls fréquents OU buts élevés → PAIRE OR à RÉDUIRE confiance")

    # ============ 5. Recommandations ============
    print()
    print("=" * 105)
    print("📋 RECOMMANDATIONS DE CALIBRAGE")
    print("=" * 105)
    print()
    print("  ✅ À CONSERVER :")
    print("     - PAIRE OR HOME pour Brighton (89% confirmé live)")
    print("     - Wolverhampton MULTI signal (validé sur Sunderland)")
    print("     - Under 3.5 quand cote_away ≥ 2.0 (Everton-Leeds OK)")
    print()
    print("  ⚠️  À AJUSTER :")
    print("     - Étendre top scores predictions aux 4-0, 4-1 si cote_home < 1.8 et away >= 3.5")
    print("     - Réduire PAIRE OR Fulham/London Blues (paire 'ouverte' avec 3-3 observé)")
    print("     - Sur away favori cote 1.5-1.9, ne PAS parier Under 3.5 (high scoring confirmé)")
    print()
    print("  ❌ À ÉVITER :")
    print("     - Combo score top 2 sur paires avec écart cote > 2 (extrême possible)")
    print("     - Pick FT '1' quand cote_X ≤ 3.2 (sous-estimation X)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
