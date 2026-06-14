"""Analyse spéciale DÉBUT DE SAISON (DS = Journée 0/1).

Hypothèses :
1. Certaines équipes sont systématiquement fortes/faibles en DS
2. Les cotes en DS peuvent être mal calibrées (l'opérateur applique les stats globales)
3. Comparer perf DS vs perf globale → trouver paires GOLD spécifiques au DS
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
    df = pd.read_sql("""
        SELECT e.team_a, e.team_b, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        JOIN results r ON r.event_id = e.id
        WHERE r.score_a IS NOT NULL
    """, engine)
    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))
    df["total"] = df.score_a + df.score_b
    df["btts"] = ((df.score_a >= 1) & (df.score_b >= 1)).astype(int)
    df["score"] = df.apply(lambda r: f"{int(r.score_a)}-{int(r.score_b)}", axis=1)
    df["journee"] = pd.to_numeric(df.round_info, errors="coerce")

    # ============ FILTRE DS : J0, J1, J2 ============
    ds = df[df.journee.isin([0, 1, 2])].copy()
    global_n = len(df)
    ds_n = len(ds)
    print(f"BDD globale : {global_n} matchs finis")
    print(f"BDD DS (J0+J1+J2) : {ds_n} matchs ({ds_n/global_n*100:.1f}%)")
    print()

    # ============ 1. PROFIL GLOBAL DS vs HORS-DS ============
    print("=" * 95)
    print("1️⃣  PROFIL GLOBAL : DS (J0/J1/J2) vs HORS-DS")
    print("=" * 95)
    non_ds = df[~df.journee.isin([0, 1, 2])]
    print(f"\n  {'Métrique':<22} {'DS':<14} {'Hors-DS':<14} {'Delta'}")
    print("  " + "-" * 60)
    metrics = [
        ("Home win", (ds.ft_o == "1").mean(), (non_ds.ft_o == "1").mean()),
        ("Draw", (ds.ft_o == "X").mean(), (non_ds.ft_o == "X").mean()),
        ("Away win", (ds.ft_o == "2").mean(), (non_ds.ft_o == "2").mean()),
        ("Over 2.5", (ds.total > 2.5).mean(), (non_ds.total > 2.5).mean()),
        ("Over 3.5", (ds.total > 3.5).mean(), (non_ds.total > 3.5).mean()),
        ("BTTS", ds.btts.mean(), non_ds.btts.mean()),
        ("Buts moy", ds.total.mean()/100, non_ds.total.mean()/100),  # divisé pour formater
    ]
    for name, d, n in metrics:
        if name == "Buts moy":
            print(f"  {name:<22} {d*100:>5.2f}         {n*100:>5.2f}         {(d-n)*100:+.2f}")
        else:
            print(f"  {name:<22} {d*100:>5.1f}%        {n*100:>5.1f}%        {(d-n)*100:+.1f}pp")

    # ============ 2. PERF ÉQUIPE EN DS ============
    print()
    print("=" * 95)
    print("2️⃣  PERF ÉQUIPES EN DS (vs perf globale)")
    print("=" * 95)
    # Home in DS vs global
    ds_home = ds.groupby("team_a").agg(
        n_ds=("ft_o", "count"),
        wr_ds=("ft_o", lambda s: (s == "1").mean()),
        avg_g_ds=("total", "mean"),
    )
    glob_home = df.groupby("team_a").agg(
        wr_g=("ft_o", lambda s: (s == "1").mean()),
    )
    merged = ds_home.join(glob_home, how="inner")
    merged["delta"] = merged.wr_ds - merged.wr_g
    merged = merged[merged.n_ds >= 5].sort_values("delta", ascending=False)

    print(f"\n  ⭐ ÉQUIPES qui PERFORMENT MIEUX en DS :")
    print(f"  {'Équipe':<22} {'n_DS':<6} {'WR DS':<10} {'WR global':<12} {'Delta':<10} {'Buts/match'}")
    print("  " + "-" * 78)
    for team, row in merged.head(10).iterrows():
        marker = " 🔥" if row["delta"] > 0.08 else ""
        print(f"  {team:<22} {int(row['n_ds']):<6} {row['wr_ds']*100:>5.1f}%    {row['wr_g']*100:>5.1f}%       {row['delta']*100:+5.1f}pp    {row['avg_g_ds']:.2f}{marker}")

    print(f"\n  ❄️  ÉQUIPES qui SOUS-PERFORMENT en DS :")
    for team, row in merged.tail(10).iterrows():
        marker = " ⚠️" if row["delta"] < -0.08 else ""
        print(f"  {team:<22} {int(row['n_ds']):<6} {row['wr_ds']*100:>5.1f}%    {row['wr_g']*100:>5.1f}%       {row['delta']*100:+5.1f}pp    {row['avg_g_ds']:.2f}{marker}")

    # ============ 3. DS MATCH 14:21 — ANALYSE PAR MATCH ============
    print()
    print("=" * 95)
    print("3️⃣  MATCHS DU 14:21 — ANALYSE DS DÉDIÉE")
    print("=" * 95)
    matches_1421 = [
        ("C. Palace", "Manchester Blue", 4.28, 3.72, 1.79),
        ("London Reds", "Everton", 1.28, 5.46, 10.44),
        ("Liverpool", "Manchester Red", 1.29, 5.54, 9.34),
        ("Leeds", "Burnley", 3.33, 3.95, 1.98),
        ("Sunderland", "Bournemouth", 5.30, 4.32, 1.56),
        ("A. Villa", "Newcastle", 1.86, 4.18, 3.52),
        ("London Blues", "Fulham", 1.74, 3.84, 4.46),
        ("Wolverhampton", "Brentford", 2.81, 3.37, 2.46),
        ("Spurs", "N. Forest", 1.54, 4.71, 5.00),
        ("West Ham", "Brighton", 2.59, 3.94, 2.38),
    ]

    print()
    for h, a, co_h, co_d, co_a in matches_1421:
        print(f"\n  ┌─ {h} vs {a}  (cotes {co_h}/{co_d}/{co_a})")
        # Stats DS HOME
        sub_h = ds[ds.team_a == h]
        if len(sub_h) > 0:
            wr_h_ds = (sub_h.ft_o == "1").mean()
            n_h = len(sub_h)
            print(f"  │  {h} home en DS  : {wr_h_ds*100:.0f}% wr ({n_h} matchs DS), {sub_h.total.mean():.2f} buts/match")
        # Stats DS AWAY
        sub_a = ds[ds.team_b == a]
        if len(sub_a) > 0:
            wr_a_ds = (sub_a.ft_o == "2").mean()
            n_a = len(sub_a)
            print(f"  │  {a} away en DS : {wr_a_ds*100:.0f}% wr ({n_a} matchs DS), {sub_a.total.mean():.2f} buts/match")
        # H2H DS
        h2h = ds[(ds.team_a == h) & (ds.team_b == a)]
        if len(h2h) >= 2:
            wr_h2h = (h2h.ft_o == "1").mean()
            top_score = Counter(h2h.score).most_common(1)[0]
            avg_g = h2h.total.mean()
            print(f"  │  H2H DS  : {len(h2h)} matchs — {h} win {wr_h2h*100:.0f}%, top {top_score[0]} ({top_score[1]}x), {avg_g:.2f} buts/match")

        # Distribution scores en H2H DS
        if len(h2h) >= 3:
            scores = Counter(h2h.score).most_common(3)
            print(f"  │  H2H DS scores top 3 : {', '.join(f'{s}({c})' for s,c in scores)}")

        # Recommandation
        implied_1 = 1/co_h*100
        implied_x = 1/co_d*100
        implied_2 = 1/co_a*100
        # Combiner DS HOME wr + global wr (50/50)
        if len(sub_h) >= 5:
            wr_h_ds = (sub_h.ft_o == "1").mean()
            wr_h_g = (df[df.team_a == h].ft_o == "1").mean() if len(df[df.team_a == h]) > 0 else 0.5
            wr_blend_1 = 0.6 * wr_h_ds + 0.4 * wr_h_g
            if wr_blend_1 * 100 > implied_1 + 5:
                print(f"  │  💎 VALUE 1 : prob estimée {wr_blend_1*100:.0f}% vs implicite {implied_1:.0f}% → {h} 1 @{co_h}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
