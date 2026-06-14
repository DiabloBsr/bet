"""EXTENSION : Over/Under, BTTS, Scores combos par segment + bucket de cote.

Pour chaque segment :
- Over 2.5 rate par bucket de cote (total cote 1X2 combo en proxy de "openness")
- BTTS rate par profil de match
- Top scores combos par segment (toutes paires confondues)
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from collections import Counter
from sqlalchemy import create_engine
from scraper.config import load_settings

SEGMENTS = [("DS", 1, 3), ("MS_early", 4, 12), ("MS_mid", 13, 25), ("MS_late", 26, 33), ("FS", 34, 38)]

def label_segment(j):
    if pd.isna(j): return None
    j = int(j)
    for name, lo, hi in SEGMENTS:
        if lo <= j <= hi: return name
    return None


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)
    df = pd.read_sql("""
        SELECT e.round_info, e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
          AND e.round_info IS NOT NULL
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df["segment"] = df.journee.apply(label_segment)
    df = df[df.segment.notna()].copy()
    df["total"] = df.score_a + df.score_b
    df["btts"] = ((df.score_a >= 1) & (df.score_b >= 1)).astype(int)
    df["score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    df["fav_strength"] = df[["odds_home", "odds_away"]].min(axis=1)  # cote favori absolu
    df["balance"] = abs(df.odds_home - df.odds_away)  # déséquilibre cotes
    df["sum_cotes"] = df.odds_home + df.odds_draw + df.odds_away  # proxy openness

    print(f"📊 Train : {len(df):,} matchs\n")

    config = {"segments": {}}

    # ============ OVER 2.5 / 3.5 par cote favori × segment ============
    print("═" * 100)
    print("  OVER/UNDER PAR PROFIL FAVORI × SEGMENT")
    print("═" * 100)

    fav_buckets = [
        ("fav_extreme", 1.00, 1.30),
        ("fav_solide", 1.30, 1.60),
        ("fav_modere", 1.60, 2.00),
        ("equilibre",  2.00, 2.50),
    ]

    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        if len(sub) < 50: continue
        print(f"\n┌─ Segment {seg_name}")
        print(f"│  {'Bucket favori':<22} {'n':<5} {'O1.5':<7} {'O2.5':<7} {'O3.5':<7} {'O4.5':<7} {'BTTS':<7} {'Buts avg'}")
        seg_data = {"over_buckets": {}, "btts_buckets": {}}
        for bname, lo, hi in fav_buckets:
            bsub = sub[(sub.fav_strength >= lo) & (sub.fav_strength < hi)]
            if len(bsub) < 10: continue
            o15 = (bsub.total > 1.5).mean()
            o25 = (bsub.total > 2.5).mean()
            o35 = (bsub.total > 3.5).mean()
            o45 = (bsub.total > 4.5).mean()
            btts = bsub.btts.mean()
            avg = bsub.total.mean()
            print(f"│  {bname:<22} {len(bsub):<5} {o15*100:5.0f}%  {o25*100:5.0f}%  {o35*100:5.0f}%  {o45*100:5.0f}%  {btts*100:5.0f}%  {avg:.2f}")
            seg_data["over_buckets"][bname] = {"n": len(bsub),
                "over_15": float(o15), "over_25": float(o25),
                "over_35": float(o35), "over_45": float(o45),
                "btts": float(btts), "avg_total": float(avg)}
        config["segments"][seg_name] = seg_data

    # ============ TOP SCORES PAR SEGMENT (toutes paires) ============
    print()
    print("═" * 100)
    print("  TOP 10 SCORES PAR SEGMENT (probabilités globales)")
    print("═" * 100)

    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        if len(sub) < 50: continue
        scores = sub.score.value_counts(normalize=True).head(10)
        ht_scores = sub.apply(lambda r: f"{int(r.ht_score_a)}-{int(r.ht_score_b)}", axis=1).value_counts(normalize=True).head(5)
        print(f"\n┌─ {seg_name}")
        print(f"│  Top 10 FT scores :")
        for s, r in scores.items():
            print(f"│    {s:<6} : {r*100:.1f}%")
        print(f"│  Top 5 HT scores :")
        for s, r in ht_scores.items():
            print(f"│    {s:<6} : {r*100:.1f}%")
        config["segments"][seg_name]["top_scores"] = {s: float(r) for s, r in scores.items()}
        config["segments"][seg_name]["top_ht_scores"] = {s: float(r) for s, r in ht_scores.items()}

    # ============ Score profile selon CONTEXTE COTES ============
    print()
    print("═" * 100)
    print("  SCORES PAR PROFIL DE COTES × SEGMENT")
    print("═" * 100)

    profiles = [
        ("home_crush",  lambda r: r.odds_home < 1.3 and r.odds_away > 7),    # crush home
        ("home_strong", lambda r: r.odds_home < 1.6 and r.odds_away > 4),    # home fort
        ("home_slight", lambda r: 1.6 <= r.odds_home < 2.2 and r.odds_away >= 2.5),  # home léger fav
        ("balanced",    lambda r: 1.9 <= r.odds_home < 2.5 and 1.9 <= r.odds_away < 2.5),  # match équilibré
        ("away_slight", lambda r: 1.6 <= r.odds_away < 2.2 and r.odds_home >= 2.5),
        ("away_strong", lambda r: r.odds_away < 1.6 and r.odds_home > 4),
        ("away_crush",  lambda r: r.odds_away < 1.3 and r.odds_home > 7),
    ]

    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        if len(sub) < 50: continue
        print(f"\n┌─ {seg_name}")
        seg_profile = {}
        for pname, pred in profiles:
            psub = sub[sub.apply(pred, axis=1)]
            if len(psub) < 10: continue
            top3 = psub.score.value_counts(normalize=True).head(3)
            o25 = (psub.total > 2.5).mean()
            btts = psub.btts.mean()
            avg = psub.total.mean()
            top3_str = ", ".join(f"{s}({r*100:.0f}%)" for s, r in top3.items())
            print(f"│  {pname:<14} n={len(psub):<4} O2.5={o25*100:.0f}%  BTTS={btts*100:.0f}%  buts={avg:.2f}  top3: {top3_str}")
            seg_profile[pname] = {
                "n": len(psub),
                "over_25": float(o25), "btts": float(btts), "avg_total": float(avg),
                "top3_scores": {s: float(r) for s, r in top3.items()},
            }
        config["segments"][seg_name]["score_profiles"] = seg_profile

    out = Path(__file__).parent.parent / "exports" / "strategy_extension.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Config sauvegardé : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
