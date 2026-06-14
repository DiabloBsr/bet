"""Backtest ScorePredictorV2 vs V5 actuel — mesurer top1/top3 accuracy."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import ScorePredictorV2
from scraper.strategy_engine import label_segment


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql("""
        SELECT e.id, e.round_info, e.team_a, e.team_b, e.expected_start,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
          AND e.round_info IS NOT NULL AND e.round_info != '0'
        ORDER BY e.expected_start
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df["segment"] = df.journee.apply(label_segment)
    df = df[df.segment.notna()].copy().reset_index(drop=True)
    df["actual_score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)

    print(f"📊 Test sur {len(df):,} matchs")

    # Split temporel
    split = int(len(df) * 0.7)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    print(f"   Train: {len(train):,}  |  Test: {len(test):,}\n")

    print("⚙️  Fit V5...")
    model_v5 = fit_model_v5(train, ht_history=train.copy(), engine=engine, form_alpha=0.0)
    print("⚙️  Init V2 predictor...")
    v2 = ScorePredictorV2(engine)

    # ── ANTI-LEAKAGE : PairScoreCache._load() lirait TOUTE la BDD (y compris
    # les matchs de test). On injecte un cache construit sur le TRAIN uniquement.
    print("⚙️  Build pair cache (TRAIN only — leak-free)...")
    cache: dict = {}
    tr = train.copy()
    tr["score"] = tr.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    min_n = v2.pair_cache.min_n
    for (ta, tb, seg), grp in tr.groupby(["team_a", "team_b", "segment"]):
        if len(grp) < min_n: continue
        counts = grp.score.value_counts().to_dict()
        total = sum(counts.values())
        cache[(ta, tb, seg)] = {s: c/total for s, c in counts.items()}
    for (ta, tb), grp in tr.groupby(["team_a", "team_b"]):
        if len(grp) < min_n: continue
        counts = grp.score.value_counts().to_dict()
        total = sum(counts.values())
        cache[(ta, tb, "ALL")] = {s: c/total for s, c in counts.items()}
    v2.pair_cache._cache = cache
    v2.pair_cache._loaded = True
    print(f"   {len(cache):,} entrées de cache (train only)")

    # Compare top1/top3 hit rates
    print("\n🔍 Prédictions...")
    v5_top1_hits = 0
    v5_top2_hits = 0
    v5_top3_hits = 0
    v2_top1_hits = 0
    v2_top2_hits = 0
    v2_top3_hits = 0
    n_pair_used = 0

    for i, m in test.iterrows():
        if i % 500 == 0: print(f"  ... {i}/{len(test)}")
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw, m.odds_away,
                                  extra_markets=m.extra_markets)
        v5_top5 = pred5.get("top5_scores_enriched") or []
        if not v5_top5: continue
        v5_top1 = [s for s, _ in v5_top5[:1]]
        v5_top2 = [s for s, _ in v5_top5[:2]]
        v5_top3 = [s for s, _ in v5_top5[:3]]

        # V2 ensemble : convert V5 top5 to grid dict
        v5_grid = {s: p for s, p in v5_top5}
        # Compléter avec quelques scores plausibles à 0
        if len(v5_grid) < 20:
            common = ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2","3-0","0-3","3-1","1-3","3-2","2-3"]
            for s in common:
                if s not in v5_grid: v5_grid[s] = 0.01
            total = sum(v5_grid.values())
            v5_grid = {s: p/total for s, p in v5_grid.items()}

        v2_top5 = v2.predict(m.team_a, m.team_b, int(m.journee),
                              v5_score_grid=v5_grid,
                              extra_markets=m.extra_markets,
                              odds_h=m.odds_home, odds_a=m.odds_away,
                              top_n=5)
        v2_top1 = [s for s, _, _ in v2_top5[:1]]
        v2_top2 = [s for s, _, _ in v2_top5[:2]]
        v2_top3 = [s for s, _, _ in v2_top5[:3]]

        # Pair availability
        pair_dist, _ = v2.pair_cache.get(m.team_a, m.team_b, m.segment)
        if pair_dist:
            n_pair_used += 1

        actual = m.actual_score
        if actual in v5_top1: v5_top1_hits += 1
        if actual in v5_top2: v5_top2_hits += 1
        if actual in v5_top3: v5_top3_hits += 1
        if actual in v2_top1: v2_top1_hits += 1
        if actual in v2_top2: v2_top2_hits += 1
        if actual in v2_top3: v2_top3_hits += 1

    print(f"\n📊 Pair empirical disponible : {n_pair_used:,}/{len(test):,} matchs ({n_pair_used/len(test)*100:.1f}%)")

    print()
    print("═" * 80)
    print(f"  RÉSULTATS — Comparaison V5 vs V2 (n_test={len(test):,})")
    print("═" * 80)
    print(f"\n  {'Métrique':<25} {'V5 actuel':<15} {'V2 ensemble':<15} {'Δ'}")
    print(f"  {'-'*25} {'-'*15} {'-'*15} {'-'*8}")
    for label, v5_hits, v2_hits in [
        ("Top 1 accuracy", v5_top1_hits, v2_top1_hits),
        ("Top 2 accuracy", v5_top2_hits, v2_top2_hits),
        ("Top 3 accuracy", v5_top3_hits, v2_top3_hits),
    ]:
        v5_acc = v5_hits / len(test) * 100
        v2_acc = v2_hits / len(test) * 100
        delta = v2_acc - v5_acc
        print(f"  {label:<25} {v5_acc:5.2f}% ({v5_hits})   {v2_acc:5.2f}% ({v2_hits})   {delta:+.2f}pp")

    return 0


if __name__ == "__main__":
    sys.exit(main())
