"""Recalibration complète sur dernière BDD : paires OR, brackets, OVER/UNDER, BTTS, scores."""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    # Filtre anti-bug match_key dedup : utiliser uniquement événements où finished_at est proche d'expected_start
    full = pd.read_sql("""
        SELECT e.id, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b,
               r.finished_at
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    print(f"Avant dédup : {len(full)} matchs")
    # Anti-bug match_key dedup : garder 1 entrée par (team_a, team_b, expected_start, score)
    full = full.drop_duplicates(["team_a", "team_b", "expected_start", "score_a", "score_b"]).copy()
    print(f"Après dédup : {len(full)} matchs")
    full_clean = full.copy()
    n_min = 8  # seuil plus bas car BDD plus petite après dédup
    full_clean["ft_o"] = np.where(full_clean.score_a > full_clean.score_b, "1",
                            np.where(full_clean.score_a == full_clean.score_b, "X", "2"))
    if "ht_score_a" in full_clean.columns:
        full_clean["ht_o"] = np.where(full_clean.ht_score_a > full_clean.ht_score_b, "1",
                                np.where(full_clean.ht_score_a == full_clean.ht_score_b, "X", "2"))
    full_clean["total"] = full_clean.score_a + full_clean.score_b
    full_clean["btts"] = ((full_clean.score_a >= 1) & (full_clean.score_b >= 1)).astype(int)
    full_clean["over_25"] = (full_clean.total > 2.5).astype(int)
    full_clean["under_35"] = (full_clean.total <= 3.5).astype(int)
    full_clean["over_15"] = (full_clean.total > 1.5).astype(int)

    print()
    print("=" * 110)
    print("📊 STATS GLOBALES (BDD CLEAN)")
    print("=" * 110)
    print(f"  Home win : {(full_clean.ft_o=='1').mean()*100:.1f}%")
    print(f"  Draw     : {(full_clean.ft_o=='X').mean()*100:.1f}%")
    print(f"  Away win : {(full_clean.ft_o=='2').mean()*100:.1f}%")
    print(f"  Over 1.5 : {full_clean.over_15.mean()*100:.1f}%")
    print(f"  Over 2.5 : {full_clean.over_25.mean()*100:.1f}%")
    print(f"  Under 3.5: {full_clean.under_35.mean()*100:.1f}%")
    print(f"  BTTS     : {full_clean.btts.mean()*100:.1f}%")

    # ============ PAIRES OR HOME (parier 1) ============
    print()
    print("=" * 110)
    print("🔧 PAIRES OR HOME (parier 1, n>=12, ROI hist >=+30%)")
    print("=" * 110)
    home_fav = full_clean[(full_clean.odds_home < full_clean.odds_away) &
                            (full_clean.odds_home < full_clean.odds_draw)].copy()
    home_fav["won"] = home_fav.ft_o == "1"

    pair_home = []
    for ta in home_fav.team_a.unique():
        for tb in home_fav.team_b.unique():
            sub = home_fav[(home_fav.team_a == ta) & (home_fav.team_b == tb)]
            if len(sub) < 8: continue
            roi = np.where(sub.won, sub.odds_home - 1, -1).mean()
            if roi >= 0.30:
                pair_home.append({"home": ta, "away": tb, "n": len(sub),
                                    "win": sub.won.mean(), "roi": roi,
                                    "cote": sub.odds_home.mean()})
    pair_home.sort(key=lambda x: -x["roi"])
    print(f"\n  {len(pair_home)} paires HOME confirmées (n>=12, ROI>=30%):")
    for p in pair_home[:25]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} win={p['win']*100:>5.1f}% ROI+{p['roi']*100:>5.1f}% cote~{p['cote']:.2f}")

    # ============ PAIRES OR AWAY ============
    print()
    print("=" * 110)
    print("🔧 PAIRES OR AWAY (parier 2, n>=12, ROI hist >=+50%, win >=40%)")
    print("=" * 110)
    away_fav_pairs = []
    for ta in full_clean.team_a.unique():
        for tb in full_clean.team_b.unique():
            sub = full_clean[(full_clean.team_a == ta) & (full_clean.team_b == tb)]
            if len(sub) < 8: continue
            sub_won = sub.ft_o == "2"
            roi = np.where(sub_won, sub.odds_away - 1, -1).mean()
            win_rate = sub_won.mean()
            # Plus stricte : ROI >=50% ET win >=40%
            if roi >= 0.50 and win_rate >= 0.40:
                away_fav_pairs.append({"home": ta, "away": tb, "n": len(sub),
                                         "win": win_rate, "roi": roi,
                                         "cote": sub.odds_away.mean()})
    away_fav_pairs.sort(key=lambda x: -x["roi"])
    print(f"\n  {len(away_fav_pairs)} paires AWAY confirmées:")
    for p in away_fav_pairs[:15]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} win={p['win']*100:>5.1f}% ROI+{p['roi']*100:>5.1f}% cote~{p['cote']:.2f}")

    # ============ PAIRES TRAP HOME ============
    print()
    print("=" * 110)
    print("❌ PAIRES TRAP HOME (parier 1 perd, n>=12, ROI <= -50%)")
    print("=" * 110)
    pair_trap = []
    for ta in home_fav.team_a.unique():
        for tb in home_fav.team_b.unique():
            sub = home_fav[(home_fav.team_a == ta) & (home_fav.team_b == tb)]
            if len(sub) < 8: continue
            roi = np.where(sub.won, sub.odds_home - 1, -1).mean()
            if roi <= -0.50:
                pair_trap.append({"home": ta, "away": tb, "n": len(sub),
                                    "win": sub.won.mean(), "roi": roi})
    pair_trap.sort(key=lambda x: x["roi"])
    print(f"\n  {len(pair_trap)} paires TRAP HOME:")
    for p in pair_trap[:20]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} win={p['win']*100:>5.1f}% ROI{p['roi']*100:>+6.1f}%")

    # ============ OVER 2.5 / UNDER 2.5 par paire ============
    print()
    print("=" * 110)
    print("🔧 OVER 2.5 GOLD (paires avec ≥80% Over historique, n>=12)")
    print("=" * 110)
    pair_over = []
    pair_btts_oui = []
    pair_btts_non = []
    pair_under = []
    pair_score = []
    for ta in full_clean.team_a.unique():
        for tb in full_clean.team_b.unique():
            sub = full_clean[(full_clean.team_a == ta) & (full_clean.team_b == tb)]
            if len(sub) < 8: continue
            over_rate = sub.over_25.mean()
            btts_rate = sub.btts.mean()
            if over_rate >= 0.80:
                pair_over.append({"home": ta, "away": tb, "n": len(sub), "rate": over_rate})
            if over_rate <= 0.30:
                pair_under.append({"home": ta, "away": tb, "n": len(sub), "over_rate": over_rate})
            if btts_rate >= 0.80:
                pair_btts_oui.append({"home": ta, "away": tb, "n": len(sub), "rate": btts_rate})
            if btts_rate <= 0.25:
                pair_btts_non.append({"home": ta, "away": tb, "n": len(sub), "bts_rate": btts_rate})
            # Score dominant
            scores = sub.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
            if len(scores) > 0:
                top_score, top_count = Counter(scores).most_common(1)[0]
                rate = top_count / len(scores)
                if rate >= 0.30 and len(scores) >= 8:
                    pair_score.append({"home": ta, "away": tb, "n": len(sub), "score": top_score, "rate": rate})

    pair_over.sort(key=lambda x: -x["rate"])
    print(f"\n  {len(pair_over)} paires OVER 2.5 GOLD:")
    for p in pair_over[:20]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} rate={p['rate']*100:.1f}%")

    print()
    print("=" * 110)
    print("🔧 UNDER 2.5 GOLD (paires avec ≤30% Over historique, n>=12)")
    print("=" * 110)
    pair_under.sort(key=lambda x: x["over_rate"])
    print(f"\n  {len(pair_under)} paires UNDER 2.5 GOLD:")
    for p in pair_under[:15]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} over_rate={p['over_rate']*100:.1f}%")

    print()
    print("=" * 110)
    print("🔧 BTTS OUI GOLD (paires avec ≥80% BTTS, n>=12)")
    print("=" * 110)
    pair_btts_oui.sort(key=lambda x: -x["rate"])
    print(f"\n  {len(pair_btts_oui)} paires BTTS OUI GOLD:")
    for p in pair_btts_oui[:20]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} rate={p['rate']*100:.1f}%")

    print()
    print("=" * 110)
    print("🔧 BTTS NON GOLD (paires avec ≤25% BTTS, n>=12)")
    print("=" * 110)
    pair_btts_non.sort(key=lambda x: x["bts_rate"])
    print(f"\n  {len(pair_btts_non)} paires BTTS NON GOLD:")
    for p in pair_btts_non[:15]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} bts_rate={p['bts_rate']*100:.1f}%")

    print()
    print("=" * 110)
    print("🔧 SCORE EXACT DOMINANT GOLD (paires avec score ≥30% récurrent, n>=12)")
    print("=" * 110)
    pair_score.sort(key=lambda x: -x["rate"])
    print(f"\n  {len(pair_score)} paires SCORE EXACT GOLD:")
    for p in pair_score[:25]:
        print(f"  ({p['home']!r:<22}, {p['away']!r:<22}) n={p['n']:<3} score={p['score']!r:<6} rate={p['rate']*100:.1f}%")

    print()
    print("=" * 110)
    print("📊 BILAN RECALIBRATION:")
    print("=" * 110)
    print(f"  Paires OR HOME    : {len(pair_home)}")
    print(f"  Paires OR AWAY    : {len(away_fav_pairs)}")
    print(f"  Paires TRAP HOME  : {len(pair_trap)}")
    print(f"  Paires OVER 2.5   : {len(pair_over)}")
    print(f"  Paires UNDER 2.5  : {len(pair_under)}")
    print(f"  Paires BTTS Oui   : {len(pair_btts_oui)}")
    print(f"  Paires BTTS Non   : {len(pair_btts_non)}")
    print(f"  Paires Score      : {len(pair_score)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
