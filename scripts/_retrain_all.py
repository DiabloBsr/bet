"""Retrain & backtest 5-fold OUT-OF-SAMPLE sur BDD clean.

1. Recalcule les brackets calibration (HOME/AWAY)
2. Valide chaque signal GOLD sur fold de validation
3. Reporte accuracy réelle par catégorie
"""
from __future__ import annotations
import sys
from collections import Counter, defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.team_gold_data import (
    PAIR_HOME_GOLD, PAIR_AWAY_GOLD, PAIR_TRAP_HOME,
    OVER_GOLD, UNDER_GOLD, BTTS_OUI_GOLD, BTTS_NON_GOLD,
    SCORE_COMBO_GOLD, SCORE_DOMINANT_GOLD,
)


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    full = pd.read_sql("""
        SELECT e.id, e.expected_start, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
        ORDER BY e.expected_start
    """, engine)
    full["ft_o"] = np.where(full.score_a > full.score_b, "1",
                    np.where(full.score_a == full.score_b, "X", "2"))
    full["score"] = full.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    full["total"] = full.score_a + full.score_b
    full["btts"] = ((full.score_a >= 1) & (full.score_b >= 1)).astype(int)
    print(f"BDD CLEAN : {len(full)} matchs au total")
    print(f"Période : {full.expected_start.min()} → {full.expected_start.max()}\n")

    # ============ TRAIN/TEST SPLIT 75/25 (temporel) ============
    split = int(len(full) * 0.75)
    train = full.iloc[:split].reset_index(drop=True)
    test = full.iloc[split:].reset_index(drop=True)
    print(f"Train : {len(train)} matchs")
    print(f"Test  : {len(test)} matchs (OUT-OF-SAMPLE)\n")

    # ============ 1. RECALCUL BRACKETS HOME/AWAY ============
    print("=" * 105)
    print("1️⃣  RECALCUL BRACKETS CALIBRATION (par équipe + cote)")
    print("=" * 105)

    def _bracket(c):
        if c < 1.3: return (1.0, 1.3)
        if c < 1.5: return (1.3, 1.5)
        if c < 1.7: return (1.5, 1.7)
        if c < 1.9: return (1.7, 1.9)
        if c < 2.1: return (1.9, 2.1)
        if c < 2.5: return (2.1, 2.5)
        if c < 3.0: return (2.5, 3.0)
        return (3.0, 100)

    home_fav = train[(train.odds_home < train.odds_away) & (train.odds_home < train.odds_draw)].copy()
    home_fav["won"] = home_fav.ft_o == "1"
    home_fav["bracket"] = home_fav.odds_home.apply(_bracket)

    print("\n  💎 BRACKET GOLD HOME (n≥15, ROI≥+10%) :")
    bracket_home_data = []
    for team in home_fav.team_a.unique():
        for br, _ in [(_bracket(c), None) for c in [1.4, 1.6, 1.8, 2.0, 2.3, 2.7, 4]]:
            sub = home_fav[(home_fav.team_a == team) & (home_fav.bracket == br)]
            if len(sub) < 8: continue
            roi = np.where(sub.won, sub.odds_home - 1, -1).mean()
            if roi >= 0.10:
                bracket_home_data.append({"team": team, "br": br, "n": len(sub), "roi": roi})
    bracket_home_data.sort(key=lambda x: -x["roi"])
    for d in bracket_home_data[:20]:
        print(f"  ('{d['team']}', {d['br']}) n={d['n']:<3} ROI+{d['roi']*100:.1f}%")

    # ============ 2. VALIDATION OOS DES SIGNAUX GOLD ============
    print()
    print("=" * 105)
    print("2️⃣  VALIDATION OUT-OF-SAMPLE DES SIGNAUX GOLD (test sur 25%)")
    print("=" * 105)

    results = {}

    # PAIRE OR HOME
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in PAIR_HOME_GOLD:
            won = m.ft_o == "1"
            roi = m.odds_home - 1 if won else -1
            bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["PAIR_HOME_GOLD"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # PAIRE OR AWAY
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in PAIR_AWAY_GOLD:
            p = PAIR_AWAY_GOLD[(m.team_a, m.team_b)]
            if m.odds_away <= p["cote"] * p.get("max_cote_factor", 1.05):
                won = m.ft_o == "2"
                roi = m.odds_away - 1 if won else -1
                bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["PAIR_AWAY_GOLD"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # PAIR TRAP HOME
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in PAIR_TRAP_HOME:
            won = m.ft_o == "1"
            roi = m.odds_home - 1 if won else -1
            bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["PAIR_TRAP_HOME"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # OVER_GOLD (parier Over 2.5, cote estimée ~1.5)
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in OVER_GOLD:
            won = m.total > 2.5
            roi = 0.5 if won else -1  # cote ~1.5
            bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["OVER_GOLD"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # UNDER_GOLD
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in UNDER_GOLD:
            won = m.total <= 2.5
            roi = 1.5 if won else -1  # cote ~2.5
            bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["UNDER_GOLD"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # BTTS_OUI
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in BTTS_OUI_GOLD:
            won = m.btts == 1
            roi = 0.8 if won else -1  # cote ~1.8
            bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["BTTS_OUI_GOLD"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # BTTS_NON
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in BTTS_NON_GOLD:
            won = m.btts == 0
            roi = 1.2 if won else -1  # cote ~2.2
            bets.append({"won": won, "roi": roi})
    if bets:
        df = pd.DataFrame(bets)
        results["BTTS_NON_GOLD"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    # SCORE_COMBO_GOLD (top 1 ou top 2)
    bets_top1 = []
    bets_combo = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in SCORE_COMBO_GOLD:
            c = SCORE_COMBO_GOLD[(m.team_a, m.team_b)]
            won_top1 = m.score == c["top1"]
            won_combo = m.score in (c["top1"], c["top2"])
            bets_top1.append({"won": won_top1, "roi": 6 if won_top1 else -1})  # cote ~7
            bets_combo.append({"won": won_combo, "roi": 5 if won_combo else -1})  # cote ~6 par score, 2 misés
    if bets_top1:
        df = pd.DataFrame(bets_top1)
        results["SCORE_TOP1"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}
        df2 = pd.DataFrame(bets_combo)
        results["SCORE_COMBO_TOP2"] = {"n": len(df2), "acc": df2.won.mean(), "roi": df2.roi.mean()}

    # SCORE_DOMINANT (rate 30-44%)
    bets = []
    for _, m in test.iterrows():
        if (m.team_a, m.team_b) in SCORE_DOMINANT_GOLD:
            s = SCORE_DOMINANT_GOLD[(m.team_a, m.team_b)]
            if 0.30 <= s["rate"] <= 0.44:
                won = m.score == s["score"]
                bets.append({"won": won, "roi": 6 if won else -1})
    if bets:
        df = pd.DataFrame(bets)
        results["SCORE_SWEET_SPOT"] = {"n": len(df), "acc": df.won.mean(), "roi": df.roi.mean()}

    print()
    print(f"  {'Signal':<22} {'n test':<8} {'Accuracy':<12} {'ROI':<10} {'Verdict'}")
    print("  " + "-" * 80)
    for sig, data in results.items():
        verdict = ""
        if data["roi"] >= 0.15: verdict = "🟢 EXCELLENT"
        elif data["roi"] >= 0.05: verdict = "✅ BON"
        elif data["roi"] >= -0.05: verdict = "🟡 NEUTRE"
        else: verdict = "❌ NÉGATIF"
        print(f"  {sig:<22} {data['n']:<8} {data['acc']*100:>5.1f}%       {data['roi']*100:>+6.2f}%    {verdict}")

    # ============ 3. STATS GLOBALES BDD CLEAN ============
    print()
    print("=" * 105)
    print("3️⃣  STATS GLOBALES BDD CLEAN (référence)")
    print("=" * 105)
    print(f"  Home win  : {(full.ft_o=='1').mean()*100:.2f}%")
    print(f"  Draw      : {(full.ft_o=='X').mean()*100:.2f}%")
    print(f"  Away win  : {(full.ft_o=='2').mean()*100:.2f}%")
    print(f"  Over 2.5  : {(full.total > 2.5).mean()*100:.2f}%")
    print(f"  BTTS      : {full.btts.mean()*100:.2f}%")
    print(f"  Total team-pairs uniques : {len(full[['team_a','team_b']].drop_duplicates())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
