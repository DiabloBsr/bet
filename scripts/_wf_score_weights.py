"""Tuning des poids ensemble ScorePredictorV2 — leak-free (pair cache = train only).

Duplique le backtest _backtest_score_v2.py mais :
- précalcule les 4 sources (v5/pair/profile/market) UNE fois par match de test
- évalue ensuite N configs de poids en pur python (rapide)
- stocke le VRAI n (nb de matchs) par paire -> variante shrinkage n/(n+k)
- fallback sans pair : redistribution PROPORTIONNELLE des poids restants
  (la classe redistribue en dur 55/0/25/20 — ici paramétrique pour comparaison équitable)
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.predictor_v5 import fit_model_v5, predict_match_v5
from scraper.score_predictor_v2 import market_score_distribution
from scraper.strategy_engine import label_segment, PROFILE_SCORES, classify_profile


def build_pair_cache(train: pd.DataFrame, min_n: int = 5):
    """Cache (dist, n_matchs) construit sur TRAIN uniquement."""
    cache: dict = {}
    tr = train.copy()
    tr["score"] = tr.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    for (ta, tb, seg), grp in tr.groupby(["team_a", "team_b", "segment"]):
        if len(grp) < min_n: continue
        counts = grp.score.value_counts().to_dict()
        total = sum(counts.values())
        cache[(ta, tb, seg)] = ({s: c/total for s, c in counts.items()}, len(grp))
    for (ta, tb), grp in tr.groupby(["team_a", "team_b"]):
        if len(grp) < min_n: continue
        counts = grp.score.value_counts().to_dict()
        total = sum(counts.values())
        cache[(ta, tb, "ALL")] = ({s: c/total for s, c in counts.items()}, len(grp))
    return cache


def get_pair(cache, ta, tb, seg):
    hit = cache.get((ta, tb, seg))
    if hit: return hit
    hit = cache.get((ta, tb, "ALL"))
    if hit: return hit
    return {}, 0


def combine(sources: dict, weights: dict, pair_n: int = 0, shrink_k: float | None = None,
            top_n: int = 3) -> list[str]:
    """Ensemble pondéré. sources = {name: dist}. Redistribution proportionnelle
    des poids des sources vides. Option shrinkage du poids pair : w_pair * n/(n+k)."""
    w = dict(weights)
    if shrink_k is not None and sources.get("pair"):
        w["pair"] = w["pair"] * (pair_n / (pair_n + shrink_k))
    # zero-out sources vides puis renormaliser les poids
    for name in list(w):
        if not sources.get(name):
            w[name] = 0.0
    tw = sum(w.values())
    if tw <= 0: return []
    w = {k: v/tw for k, v in w.items()}

    all_scores = set()
    for d in sources.values():
        all_scores |= set(d)
    ens = {s: sum(w[name] * sources[name].get(s, 0) for name in w if sources.get(name))
           for s in all_scores}
    return [s for s, _ in sorted(ens.items(), key=lambda x: -x[1])[:top_n]]


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

    split = int(len(df) * 0.7)
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    print(f"📊 {len(df):,} matchs — Train {len(train):,} | Test {len(test):,}")

    print("⚙️  Fit V5 (train only)...")
    model_v5 = fit_model_v5(train, ht_history=train.copy(), engine=engine, form_alpha=0.0)
    pair_cache = build_pair_cache(train)
    print(f"   Pair cache (train only) : {len(pair_cache):,} entrées")

    # ── Précalcul des sources par match de test ──────────────────────────
    print("🔍 Précalcul des sources (1 passage V5)...")
    rows = []  # (actual, sources, pair_n, v5_top3)
    for i, m in test.iterrows():
        if i % 500 == 0: print(f"  ... {i}/{len(test)}")
        pred5 = predict_match_v5(model_v5, m.team_a, m.team_b, m.odds_home, m.odds_draw,
                                 m.odds_away, extra_markets=m.extra_markets)
        v5_top5 = pred5.get("top5_scores_enriched") or []
        if not v5_top5: continue
        v5_grid = {s: p for s, p in v5_top5}
        if len(v5_grid) < 20:
            common = ["0-0","1-0","0-1","1-1","2-0","0-2","2-1","1-2","2-2",
                      "3-0","0-3","3-1","1-3","3-2","2-3"]
            for s in common:
                if s not in v5_grid: v5_grid[s] = 0.01
            total = sum(v5_grid.values())
            v5_grid = {s: p/total for s, p in v5_grid.items()}

        pair_dist, pair_n = get_pair(pair_cache, m.team_a, m.team_b, m.segment)

        profile_dist = {}
        if m.odds_home and m.odds_away:
            profile = classify_profile(m.odds_home, m.odds_away)
            pdata = PROFILE_SCORES.get(m.segment, {}).get(profile)
            if pdata:
                profile_dist = pdata["top3"].copy()

        market_dist = market_score_distribution(m.extra_markets)

        sources = {"v5": v5_grid, "pair": pair_dist, "profile": profile_dist,
                   "market": market_dist}
        rows.append((m.actual_score, sources, pair_n, [s for s, _ in v5_top5[:3]]))

    n_eval = len(rows)
    print(f"   {n_eval:,} matchs évaluables\n")

    # ── Baseline V5 ──────────────────────────────────────────────────────
    v5_hits = [0, 0, 0]
    for actual, _, _, v5_top3 in rows:
        for k in range(3):
            if actual in v5_top3[:k+1]: v5_hits[k] += 1

    # ── Configs de poids ─────────────────────────────────────────────────
    configs = [
        ("V2 défaut 40/30/15/15",      {"v5": .40, "pair": .30, "profile": .15, "market": .15}, None),
        ("25/45/10/20",                {"v5": .25, "pair": .45, "profile": .10, "market": .20}, None),
        ("30/30/10/30",                {"v5": .30, "pair": .30, "profile": .10, "market": .30}, None),
        ("50/0/0/50 (v5+market)",      {"v5": .50, "pair": .0,  "profile": .0,  "market": .50}, None),
        ("30/15/5/50 (market-heavy)",  {"v5": .30, "pair": .15, "profile": .05, "market": .50}, None),
        ("40/30/15/15 + shrink k=10",  {"v5": .40, "pair": .30, "profile": .15, "market": .15}, 10.0),
        ("25/45/10/20 + shrink k=10",  {"v5": .25, "pair": .45, "profile": .10, "market": .20}, 10.0),
    ]

    print("═" * 86)
    print(f"  RÉSULTATS LEAK-FREE (n_test={n_eval:,}) — pair cache train-only, "
          f"fallback proportionnel")
    print("═" * 86)
    print(f"  {'Config':<30} {'Top1':>8} {'Top2':>8} {'Top3':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'V5 seul (baseline)':<30} "
          f"{v5_hits[0]/n_eval*100:7.2f}% {v5_hits[1]/n_eval*100:7.2f}% "
          f"{v5_hits[2]/n_eval*100:7.2f}%")

    best = None
    for name, w, shrink in configs:
        hits = [0, 0, 0]
        for actual, sources, pair_n, _ in rows:
            top3 = combine(sources, w, pair_n=pair_n, shrink_k=shrink, top_n=3)
            for k in range(3):
                if actual in top3[:k+1]: hits[k] += 1
        accs = [h/n_eval*100 for h in hits]
        print(f"  {name:<30} {accs[0]:7.2f}% {accs[1]:7.2f}% {accs[2]:7.2f}%")
        if best is None or accs[0] > best[1][0]:
            best = (name, accs)

    print()
    print(f"🏆 Meilleure config (top1) : {best[0]} — "
          f"top1 {best[1][0]:.2f}% / top2 {best[1][1]:.2f}% / top3 {best[1][2]:.2f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
