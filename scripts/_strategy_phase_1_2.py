"""PHASES 1 + 2 : Base rates 1X2/Goals/BTTS/HT/FT par segment de saison.

Segments :
- DS (Début) : J1-J3
- MS Early  : J4-J12
- MS Mid    : J13-J25
- MS Late   : J26-J33
- FS (Fin)  : J34-J38
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np, pandas as pd
from sqlalchemy import create_engine
from scraper.config import load_settings

SEGMENTS = [
    ("DS",      1, 3),
    ("MS_early", 4, 12),
    ("MS_mid",  13, 25),
    ("MS_late", 26, 33),
    ("FS",      34, 38),
]

def label_segment(j):
    if pd.isna(j): return None
    j = int(j)
    for name, lo, hi in SEGMENTS:
        if lo <= j <= hi:
            return name
    return None


def main():
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql("""
        SELECT e.id, e.round_info, e.team_a, e.team_b,
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

    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))
    df["ht_o"] = np.where(df.ht_score_a > df.ht_score_b, "1",
                  np.where(df.ht_score_a == df.ht_score_b, "X", "2"))
    df["score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    df["ht_score"] = df.apply(lambda r: f"{int(r.ht_score_a)}-{int(r.ht_score_b)}", axis=1)
    df["total"] = df.score_a + df.score_b
    df["ht_total"] = df.ht_score_a + df.ht_score_b
    df["btts"] = ((df.score_a >= 1) & (df.score_b >= 1)).astype(int)
    df["ht_btts"] = ((df.ht_score_a >= 1) & (df.ht_score_b >= 1)).astype(int)

    print(f"📊 Train exploitable : {len(df):,} matchs")
    print(f"   Période : {df.journee.min():.0f} → {df.journee.max():.0f}")
    print()

    config = {"segments": {}}

    print("═" * 100)
    print("  PHASE 1 — BASE RATES PAR SEGMENT")
    print("═" * 100)

    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        if len(sub) == 0: continue
        print(f"\n┌─ {seg_name}  (n = {len(sub):,})")
        print(f"│  📊 1X2 : 1={(sub.ft_o=='1').mean()*100:.1f}%  X={(sub.ft_o=='X').mean()*100:.1f}%  2={(sub.ft_o=='2').mean()*100:.1f}%")
        print(f"│  🥅 Buts moy : {sub.total.mean():.2f}  (HT : {sub.ht_total.mean():.2f})")
        print(f"│  📈 Over : O0.5={(sub.total>0.5).mean()*100:.0f}%  O1.5={(sub.total>1.5).mean()*100:.0f}%  O2.5={(sub.total>2.5).mean()*100:.0f}%  O3.5={(sub.total>3.5).mean()*100:.0f}%  O4.5={(sub.total>4.5).mean()*100:.0f}%")
        print(f"│  📉 Under: U1.5={(sub.total<=1).mean()*100:.0f}%  U2.5={(sub.total<=2).mean()*100:.0f}%  U3.5={(sub.total<=3).mean()*100:.0f}%")
        print(f"│  🎯 BTTS : OUI={sub.btts.mean()*100:.1f}%  NON={(1-sub.btts.mean())*100:.1f}%")
        print(f"│  ⏰ HT BTTS : {sub.ht_btts.mean()*100:.1f}%")
        print(f"│  📐 HT 1X2 : 1={(sub.ht_o=='1').mean()*100:.1f}%  X={(sub.ht_o=='X').mean()*100:.1f}%  2={(sub.ht_o=='2').mean()*100:.1f}%")

        # Top 10 scores
        top10 = sub.score.value_counts().head(10)
        print(f"│  ⭐ TOP 10 scores : {', '.join(f'{s}({c/len(sub)*100:.1f}%)' for s,c in top10.items())}")

        # Top 5 HT scores
        top5_ht = sub.ht_score.value_counts().head(5)
        print(f"│  ⏰ TOP 5 HT scores : {', '.join(f'{s}({c/len(sub)*100:.1f}%)' for s,c in top5_ht.items())}")

        config["segments"][seg_name] = {
            "n": int(len(sub)),
            "rate_1": float((sub.ft_o=="1").mean()),
            "rate_X": float((sub.ft_o=="X").mean()),
            "rate_2": float((sub.ft_o=="2").mean()),
            "avg_total": float(sub.total.mean()),
            "avg_ht_total": float(sub.ht_total.mean()),
            "over_15": float((sub.total>1.5).mean()),
            "over_25": float((sub.total>2.5).mean()),
            "over_35": float((sub.total>3.5).mean()),
            "btts": float(sub.btts.mean()),
            "ht_btts": float(sub.ht_btts.mean()),
            "top_scores": {s: float(c/len(sub)) for s,c in top10.items()},
            "top_ht_scores": {s: float(c/len(sub)) for s,c in top5_ht.items()},
            "ht_rate_1": float((sub.ht_o=="1").mean()),
            "ht_rate_X": float((sub.ht_o=="X").mean()),
            "ht_rate_2": float((sub.ht_o=="2").mean()),
        }

    print()
    print("═" * 100)
    print("  PHASE 2 — TRANSITIONS HT → FT PAR SEGMENT")
    print("═" * 100)

    for seg_name, _, _ in SEGMENTS:
        sub = df[df.segment == seg_name]
        if len(sub) == 0: continue
        print(f"\n┌─ {seg_name}")
        # Markov HT → FT
        print(f"│  {'HT/FT':<10} {'1':<8} {'X':<8} {'2':<8} {'TOTAL':<6}")
        ht_ft_trans = {}
        for ht in ["1", "X", "2"]:
            row = sub[sub.ht_o == ht]
            if len(row) == 0: continue
            p_1 = (row.ft_o=="1").mean()
            p_x = (row.ft_o=="X").mean()
            p_2 = (row.ft_o=="2").mean()
            print(f"│  HT={ht:<8} {p_1*100:.0f}%      {p_x*100:.0f}%      {p_2*100:.0f}%      n={len(row)}")
            ht_ft_trans[ht] = {"1": float(p_1), "X": float(p_x), "2": float(p_2), "n": int(len(row))}
        config["segments"][seg_name]["ht_ft_transitions"] = ht_ft_trans

        # Goals 2e mi-temps
        sub2 = sub.copy()
        sub2["sh_total"] = sub2.total - sub2.ht_total
        print(f"│  Buts 2nd mi-tps moy : {sub2.sh_total.mean():.2f}  (vs 1ère {sub2.ht_total.mean():.2f})")
        config["segments"][seg_name]["avg_sh_total"] = float(sub2.sh_total.mean())

    # Save config
    out = Path(__file__).parent.parent / "exports" / "strategy_phase_1_2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Config sauvegardé : {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
