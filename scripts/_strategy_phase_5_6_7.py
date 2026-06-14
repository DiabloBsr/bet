"""PHASES 5+6+7 :
- Phase 5 : Favoris par bucket de cote par segment (cote 1.2-2.5)
- Phase 6 : NON-favoris (cote ≥ 2.5, upsets) par segment
- Phase 7 : Minute du premier but par segment
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
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
               r.score_a, r.score_b, r.ht_score_a, r.ht_score_b, r.goals_json
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL AND r.ht_score_a IS NOT NULL
          AND e.round_info IS NOT NULL
    """, engine)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")
    df["segment"] = df.journee.apply(label_segment)
    df = df[df.segment.notna()].copy()
    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))
    df["won_home"] = (df.ft_o == "1").astype(int)
    df["won_away"] = (df.ft_o == "2").astype(int)
    df["drew"] = (df.ft_o == "X").astype(int)
    df["total"] = df.score_a + df.score_b

    print(f"📊 Train : {len(df):,} matchs\n")

    config = {"segments": {}}

    # ============ PHASE 5+6 — COTES PAR SEGMENT ============
    # Buckets cote home (favori et non-favori)
    BUCKETS = [
        ("⭐⭐⭐ favori EXTRÊME", 1.0, 1.30),
        ("⭐⭐ favori SOLIDE",   1.30, 1.50),
        ("⭐ favori MODÉRÉ",   1.50, 1.80),
        ("⚖️ léger favori",    1.80, 2.20),
        ("⚖️ équilibré",       2.20, 2.70),
        ("🎰 non-favori léger", 2.70, 3.50),
        ("🎰🎰 underdog",       3.50, 5.00),
        ("🎰🎰🎰 long shot",    5.00, 50.00),
    ]

    print("═" * 110)
    print("  PHASE 5+6 — PERFORMANCE PAR BUCKET DE COTE × SEGMENT")
    print("═" * 110)

    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        if len(sub) < 50: continue
        print(f"\n┌─ Segment {seg_name}  (n={len(sub):,})")
        print(f"│  {'Bucket':<28} {'COTE HOME':<14} {'n':<6} {'WR_h':<8} {'implicite':<11} {'EDGE':<10} {'ROI':<10}")
        print(f"│  {'-'*28} {'-'*14} {'-'*6} {'-'*8} {'-'*11} {'-'*10} {'-'*10}")
        seg_buckets = []
        for bname, lo, hi in BUCKETS:
            bsub = sub[(sub.odds_home >= lo) & (sub.odds_home < hi)]
            if len(bsub) < 5: continue
            wr_h = bsub.won_home.mean()
            avg_cote = bsub.odds_home.mean()
            implied = 1 / avg_cote
            edge = wr_h - implied
            roi = (wr_h * (avg_cote - 1) - (1 - wr_h))
            tag = "🔥" if edge >= 0.05 else ("❄️" if edge <= -0.05 else "  ")
            print(f"│  {bname:<28} {f'[{lo:.2f}-{hi:.2f})':<14} {len(bsub):<6} {wr_h*100:5.1f}%  {implied*100:5.1f}%      {edge*100:+5.1f}pp {tag}  {roi*100:+5.1f}%")
            seg_buckets.append({"bucket": bname, "lo": lo, "hi": hi, "n": int(len(bsub)),
                                "wr_h": float(wr_h), "implied": float(implied), "edge": float(edge), "roi": float(roi)})
        # Idem AWAY
        print(f"│")
        print(f"│  AWAY :")
        print(f"│  {'Bucket':<28} {'COTE AWAY':<14} {'n':<6} {'WR_a':<8} {'implicite':<11} {'EDGE':<10} {'ROI':<10}")
        for bname, lo, hi in BUCKETS:
            bsub = sub[(sub.odds_away >= lo) & (sub.odds_away < hi)]
            if len(bsub) < 5: continue
            wr_a = bsub.won_away.mean()
            avg_cote = bsub.odds_away.mean()
            implied = 1 / avg_cote
            edge = wr_a - implied
            roi = (wr_a * (avg_cote - 1) - (1 - wr_a))
            tag = "🔥" if edge >= 0.05 else ("❄️" if edge <= -0.05 else "  ")
            print(f"│  {bname:<28} {f'[{lo:.2f}-{hi:.2f})':<14} {len(bsub):<6} {wr_a*100:5.1f}%  {implied*100:5.1f}%      {edge*100:+5.1f}pp {tag}  {roi*100:+5.1f}%")

        config["segments"][seg_name] = {"buckets_home": seg_buckets}

    # ============ PHASE 7 — MINUTE DU PREMIER BUT ============
    print()
    print("═" * 110)
    print("  PHASE 7 — MINUTE DU PREMIER BUT PAR SEGMENT")
    print("═" * 110)

    def first_goal_minute(goals_json_str):
        try:
            if not goals_json_str or goals_json_str == "null": return None
            arr = json.loads(goals_json_str)
            if not arr: return None
            return min(g["minute"] for g in arr if "minute" in g)
        except:
            return None

    df["first_goal"] = df.goals_json.apply(first_goal_minute)
    df["has_first_goal"] = df.first_goal.notna()

    print(f"\n  Matchs avec goals_json valides : {df.has_first_goal.sum():,} / {len(df):,}\n")

    fg_per_seg = {}
    for seg_name, _, _ in SEGMENTS:
        sub = df[(df.segment == seg_name) & df.has_first_goal]
        if len(sub) < 30: continue
        fg = sub.first_goal.dropna().astype(int)
        # Distribution
        no_goal = (sub.total == 0).sum() / len(sub) * 100
        print(f"\n┌─ {seg_name}  (n={len(sub):,}, sans but : {no_goal:.1f}%)")
        print(f"│  Minute médiane : {fg.median():.0f}'  Moyenne : {fg.mean():.1f}'")
        print(f"│  Pct cumulatif :")
        for mn in [5, 10, 15, 20, 25, 30, 40, 45, 60, 75]:
            pct = (fg <= mn).mean() * 100
            print(f"│    1er but ≤ {mn:>3}' : {pct:5.1f}%")
        # Window
        windows = [(0,15),(0,30),(0,45),(15,30),(15,45),(30,45),(45,60),(60,90)]
        print(f"│  Fenêtres (1er but dedans) :")
        for w0, w1 in windows:
            pct = ((fg > w0) & (fg <= w1)).mean() * 100
            print(f"│    ({w0:>2}'-{w1:<2}'] : {pct:5.1f}%")
        fg_per_seg[seg_name] = {
            "n": int(len(sub)),
            "no_goal_rate": float(no_goal/100),
            "median": float(fg.median()),
            "mean": float(fg.mean()),
            "pct_under_15": float((fg<=15).mean()),
            "pct_under_30": float((fg<=30).mean()),
            "pct_under_45": float((fg<=45).mean()),
        }

    config["first_goal_per_segment"] = fg_per_seg

    out = Path(__file__).parent.parent / "exports" / "strategy_phase_5_6_7.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Config sauvegardé : {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
