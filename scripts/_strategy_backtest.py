"""BACKTEST historique du strategy_engine sur tous les matchs.

Pour chaque match :
1. Appliquer engine.evaluate()
2. Vérifier si pick recommandé correspond au résultat réel
3. Calculer ROI par segment, par catégorie de signal, et global
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings
from scraper.strategy_engine import StrategyEngine, label_segment


def main():
    settings = load_settings()
    engine_db = create_engine(settings.db_url)
    df = pd.read_sql("""
        SELECT e.round_info, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND e.round_info IS NOT NULL
    """, engine_db)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df["segment"] = df.journee.apply(label_segment)
    df = df[df.segment.notna()].copy()
    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))
    df["score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    df["total"] = df.score_a + df.score_b
    df["btts"] = ((df.score_a >= 1) & (df.score_b >= 1)).astype(int)

    print(f"📊 Backtest sur {len(df):,} matchs\n")

    se = StrategyEngine()

    # ============ BACKTEST ============
    results_per_seg = defaultdict(list)
    results_per_cat = defaultdict(list)
    results_global = []
    skipped = defaultdict(int)
    cnt = 0
    for _, m in df.iterrows():
        cnt += 1
        if cnt % 500 == 0: print(f"  ... {cnt}/{len(df)}")
        ev = se.evaluate(m.team_a, m.team_b, int(m.journee),
                          float(m.odds_home), float(m.odds_draw), float(m.odds_away))
        if not ev.recommended_picks:
            skipped[m.segment] += 1
            continue
        pick = ev.recommended_picks[0]
        if pick["pick"] not in ("1", "X", "2"):
            continue
        won = m.ft_o == pick["pick"]
        roi = (pick["cote"] - 1) if won else -1
        record = {
            "segment": m.segment, "team_a": m.team_a, "team_b": m.team_b,
            "journee": int(m.journee),
            "odds_h": m.odds_home, "odds_a": m.odds_away,
            "pick": pick["pick"], "cote": pick["cote"], "strength": pick["strength"],
            "actual": m.ft_o, "won": won, "roi": roi,
            "n_supporting": pick["n_supporting"],
        }
        results_per_seg[m.segment].append(record)
        results_global.append(record)
        # Catégorise par strength
        if pick["strength"] >= 1.5:
            results_per_cat["very_strong (≥1.5)"].append(record)
        elif pick["strength"] >= 1.0:
            results_per_cat["strong (1.0-1.5)"].append(record)
        elif pick["strength"] >= 0.5:
            results_per_cat["medium (0.5-1.0)"].append(record)
        else:
            results_per_cat["weak (<0.5)"].append(record)

    # ============ RAPPORT ============
    print()
    print("═" * 100)
    print("  RÉSULTATS GLOBAUX")
    print("═" * 100)
    g = pd.DataFrame(results_global)
    if len(g) > 0:
        print(f"\n  🎯 Total picks       : {len(g):,} (sur {len(df):,} matchs = {len(g)/len(df)*100:.1f}% couverture)")
        print(f"  ✅ Wins              : {g.won.sum():,} ({g.won.mean()*100:.1f}%)")
        print(f"  💰 ROI moyen         : {g.roi.mean()*100:+.2f}%")
        print(f"  💵 ROI cumulé        : {g.roi.sum():+.0f}u (si 1u par pari)")
        print(f"  📊 Cote moyenne pick : {g.cote.mean():.2f}")
    else:
        print("\n  ❌ Aucun pick généré!")

    print()
    print("═" * 100)
    print("  PAR SEGMENT")
    print("═" * 100)
    print(f"\n  {'Segment':<12} {'n picks':<10} {'wins':<10} {'WR':<8} {'cote moy':<10} {'ROI':<10} {'cumul':<10} {'skip n'}")
    print(f"  {'='*12} {'='*10} {'='*10} {'='*8} {'='*10} {'='*10} {'='*10} {'='*8}")
    for seg in ["DS", "MS_early", "MS_mid", "MS_late", "FS"]:
        recs = pd.DataFrame(results_per_seg.get(seg, []))
        if len(recs) == 0:
            print(f"  {seg:<12} {'0':<10} {'-':<10} {'-':<8} {'-':<10} {'-':<10} {'-':<10} {skipped[seg]}")
            continue
        sk = skipped.get(seg, 0)
        print(f"  {seg:<12} {len(recs):<10} {recs.won.sum():<10} {recs.won.mean()*100:5.1f}%   {recs.cote.mean():5.2f}     {recs.roi.mean()*100:+5.1f}%    {recs.roi.sum():+5.0f}u    {sk}")

    print()
    print("═" * 100)
    print("  PAR STRENGTH (conviction)")
    print("═" * 100)
    print(f"\n  {'Strength':<25} {'n picks':<10} {'WR':<8} {'cote moy':<10} {'ROI':<10}")
    for cat in ["very_strong (≥1.5)", "strong (1.0-1.5)", "medium (0.5-1.0)", "weak (<0.5)"]:
        recs = pd.DataFrame(results_per_cat.get(cat, []))
        if len(recs) == 0: continue
        print(f"  {cat:<25} {len(recs):<10} {recs.won.mean()*100:5.1f}%   {recs.cote.mean():5.2f}     {recs.roi.mean()*100:+5.1f}%")

    # ============ DETAIL DS / FS (segments les plus intéressants) ============
    for seg in ["DS", "FS"]:
        recs = pd.DataFrame(results_per_seg.get(seg, []))
        if len(recs) == 0: continue
        print()
        print(f"  📌 DÉTAIL {seg} :")
        print(f"     Top 10 wins :")
        wins = recs[recs.won].nlargest(10, "cote")
        for _, w in wins.head(5).iterrows():
            print(f"       {w['team_a']:<22} vs {w['team_b']:<22}  pick {w['pick']} @{w['cote']:.2f} → {w['actual']}  ✅ +{w['cote']-1:.2f}u")
        print(f"     Top 5 losses (cote la plus basse perdue) :")
        losses = recs[~recs.won].nsmallest(5, "cote")
        for _, l in losses.iterrows():
            print(f"       {l['team_a']:<22} vs {l['team_b']:<22}  pick {l['pick']} @{l['cote']:.2f} → {l['actual']}  ❌")

    # Save full results CSV
    out = Path(__file__).parent.parent / "exports" / "backtest_results.csv"
    g.to_csv(out, index=False)
    print(f"\n💾 Détail complet : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
