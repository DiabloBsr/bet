"""TIER 1 PICKER — recherche le filtre qui maximise l'ACCURACY 1X2.

Test plusieurs configurations de filtres et mesure :
- Couverture (n picks / n matchs)
- Accuracy (WR)
- ROI (pour comparaison)

Objectif : trouver un filtre qui donne 70-80% WR avec couverture raisonnable.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.strategy_engine import StrategyEngine, label_segment


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    # Données : tous les matchs finis avec cotes + HT
    df = pd.read_sql("""
        SELECT e.round_info, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
          AND e.round_info IS NOT NULL AND e.round_info != '0'
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df["segment"] = df.journee.apply(label_segment)
    df = df[df.segment.notna()].copy().reset_index(drop=True)
    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))

    print(f"📊 Test sur {len(df):,} matchs\n")

    # Split temporel : train sur premier 70%, test sur dernier 30%
    split = int(len(df) * 0.7)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    print(f"   Train: {len(train):,}  |  Test: {len(test):,}\n")

    # Fit V5
    print("⚙️  Fit V5...")
    model_v5 = fit_model_v5(train, ht_history=train.copy(), engine=engine, form_alpha=0.0)
    se = StrategyEngine()

    # Pour chaque match de test, prédire et collecter signaux
    print("🔍 Prédictions...")
    rows = []
    for i, m in test.iterrows():
        if i % 500 == 0: print(f"  ... {i}/{len(test)}")
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)
        ft_pick = pred5.get("primary_pick", "—")
        ft_p = (pred5.get("primary_p") or 0)
        ev_se = se.evaluate(m.team_a, m.team_b, int(m.journee), m.odds_home, m.odds_draw, m.odds_away)
        se_pick = ev_se.recommended_picks[0] if ev_se.recommended_picks else None
        n_traps_on_ft = sum(1 for t in ev_se.traps if t.pick == ft_pick)
        n_signals_on_ft = sum(1 for s in ev_se.base_signals if s.pick == ft_pick)
        # Cote du pick FT
        cote_ft = {"1": m.odds_home, "X": m.odds_draw, "2": m.odds_away}.get(ft_pick, None)
        rows.append({
            "segment": m.segment,
            "ft_pick": ft_pick, "ft_p": ft_p, "cote_ft": cote_ft,
            "se_pick": se_pick["pick"] if se_pick else None,
            "se_strength": se_pick["strength"] if se_pick else 0,
            "se_pick_eq_ft": se_pick and se_pick["pick"] == ft_pick,
            "n_traps_on_ft": n_traps_on_ft,
            "n_signals_on_ft": n_signals_on_ft,
            "actual": m.ft_o,
            "ft_won": ft_pick == m.ft_o,
            "se_won": se_pick and se_pick["pick"] == m.ft_o,
        })
    res = pd.DataFrame(rows)
    print(f"  Done : {len(res)} rows\n")

    # ============ Test différentes configurations de filtres ============
    print("═" * 100)
    print(f"  TEST FILTRES (sur test set n={len(res):,})")
    print("═" * 100)
    print(f"\n  {'Filtre':<55} {'n picks':<10} {'cover':<8} {'WR':<8} {'cote moy':<10} {'ROI'}")
    print(f"  {'-'*55} {'-'*10} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    filters = [
        ("FT V5 pick (no filter)",
         lambda r: True, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("FT V5 ≥ 50%",
         lambda r: r.ft_p >= 0.50, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("FT V5 ≥ 55%",
         lambda r: r.ft_p >= 0.55, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("FT V5 ≥ 60%",
         lambda r: r.ft_p >= 0.60, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("FT V5 ≥ 65%",
         lambda r: r.ft_p >= 0.65, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("FT V5 ≥ 70%",
         lambda r: r.ft_p >= 0.70, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("FT V5 ≥ 75%",
         lambda r: r.ft_p >= 0.75, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.5",
         lambda r: r.cote_ft and r.cote_ft <= 1.5, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.7 (favori solide)",
         lambda r: r.cote_ft and r.cote_ft <= 1.7, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 2.0 (favori modéré)",
         lambda r: r.cote_ft and r.cote_ft <= 2.0, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.5 ET V5 ≥ 60%",
         lambda r: r.cote_ft and r.cote_ft <= 1.5 and r.ft_p >= 0.60, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.7 ET V5 ≥ 60%",
         lambda r: r.cote_ft and r.cote_ft <= 1.7 and r.ft_p >= 0.60, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.7 ET V5 ≥ 65%",
         lambda r: r.cote_ft and r.cote_ft <= 1.7 and r.ft_p >= 0.65, lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 2.0 ET V5 ≥ 55% ET 0 TRAP",
         lambda r: r.cote_ft and r.cote_ft <= 2.0 and r.ft_p >= 0.55 and r.n_traps_on_ft == 0,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.7 ET V5 ≥ 60% ET 0 TRAP",
         lambda r: r.cote_ft and r.cote_ft <= 1.7 and r.ft_p >= 0.60 and r.n_traps_on_ft == 0,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.7 ET V5 ≥ 60% ET signal SE = FT",
         lambda r: r.cote_ft and r.cote_ft <= 1.7 and r.ft_p >= 0.60 and r.se_pick_eq_ft,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.5 ET V5 ≥ 65% ET 0 TRAP",
         lambda r: r.cote_ft and r.cote_ft <= 1.5 and r.ft_p >= 0.65 and r.n_traps_on_ft == 0,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.4 ET V5 ≥ 70%",
         lambda r: r.cote_ft and r.cote_ft <= 1.4 and r.ft_p >= 0.70,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.4 ET V5 ≥ 70% ET 0 TRAP",
         lambda r: r.cote_ft and r.cote_ft <= 1.4 and r.ft_p >= 0.70 and r.n_traps_on_ft == 0,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.4 ET V5 ≥ 70% ET signal SE = FT",
         lambda r: r.cote_ft and r.cote_ft <= 1.4 and r.ft_p >= 0.70 and r.se_pick_eq_ft,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.3 ET V5 ≥ 75% (FAVORI EXTRÊME)",
         lambda r: r.cote_ft and r.cote_ft <= 1.3 and r.ft_p >= 0.75,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
        ("Cote ≤ 1.3 ET V5 ≥ 75% ET 0 TRAP",
         lambda r: r.cote_ft and r.cote_ft <= 1.3 and r.ft_p >= 0.75 and r.n_traps_on_ft == 0,
         lambda r: r.ft_pick, lambda r: r.cote_ft, lambda r: r.ft_won),
    ]

    best = None
    for name, cond, _, _, _ in filters:
        sub = res[res.apply(cond, axis=1)]
        if len(sub) == 0:
            print(f"  {name:<55} {'0':<10} {'0%':<8} {'-':<8} {'-':<10} {'-'}")
            continue
        wr = sub.ft_won.mean()
        cover = len(sub) / len(res)
        cote_avg = sub.cote_ft.mean()
        roi = (sub.ft_won * (sub.cote_ft - 1) - (1 - sub.ft_won.astype(int))).mean()
        tag = " ⭐" if wr >= 0.70 and cover >= 0.10 else ""
        print(f"  {name:<55} {len(sub):<10} {cover*100:5.1f}%  {wr*100:5.1f}%  {cote_avg:5.2f}     {roi*100:+5.1f}%{tag}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
