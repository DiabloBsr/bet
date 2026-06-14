"""Détection des frontières de saison dans la BDD Bet261 virtual.

Hypothèses à tester :
1. Une saison = X journées (probablement 38 comme la Premier League réelle)
2. Une saison ≈ un bloc temporel (24h, 7j, 30j ?)
3. Les forces d'équipes changent d'une saison à l'autre
"""
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
    df = pd.read_sql("""
        SELECT e.id, e.expected_start, e.competition, e.round_info,
               e.team_a, e.team_b,
               o.odds_home, o.odds_draw, o.odds_away,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.id = (SELECT MIN(id) FROM odds_snapshots WHERE event_id = e.id)
        LEFT JOIN results r ON r.event_id = e.id
        ORDER BY e.expected_start
    """, engine)
    df["ft_o"] = np.where(df.score_a > df.score_b, "1",
                  np.where(df.score_a == df.score_b, "X", "2"))
    df["expected_start"] = pd.to_datetime(df.expected_start)

    print(f"BDD : {len(df)} events")
    print(f"Période : {df.expected_start.min()} → {df.expected_start.max()}")
    print()

    # ============ 1. Format de round_info ============
    print("=" * 90)
    print("1️⃣  Format de round_info — exemples")
    print("=" * 90)
    samples = df.round_info.dropna().drop_duplicates().head(20).tolist()
    for s in samples:
        print(f"   {s!r}")

    # Tenter d'extraire numéro de journée
    df["journee"] = df.round_info.str.extract(r"(\d+)").astype(float)
    print(f"\n   Journées détectées : min={df.journee.min()}, max={df.journee.max()}")
    print(f"   Distribution top 10 :")
    for j, c in Counter(df.journee.dropna().astype(int)).most_common(10):
        print(f"     Journée {j}  : {c} matchs")

    # ============ 2. Reset journée 1 = nouvelle saison ============
    print()
    print("=" * 90)
    print("2️⃣  Detection des resets (journée X → journée 1 = nouvelle saison ?)")
    print("=" * 90)
    df_sorted = df.dropna(subset=["journee"]).sort_values("expected_start").reset_index(drop=True)
    df_sorted["prev_j"] = df_sorted.journee.shift(1)
    df_sorted["jump_back"] = (df_sorted.prev_j > df_sorted.journee + 5)  # journée chute brutalement

    resets = df_sorted[df_sorted.jump_back]
    print(f"\n   {len(resets)} resets détectés")
    if len(resets) > 0:
        print(f"\n   10 premiers resets :")
        for _, r in resets.head(10).iterrows():
            print(f"     {r.expected_start}  J{int(r.prev_j)} → J{int(r.journee)}")

        # Numéroter les saisons
        df_sorted["season_id"] = df_sorted.jump_back.cumsum()
        seasons = df_sorted.groupby("season_id").agg(
            start=("expected_start", "min"),
            end=("expected_start", "max"),
            n_matchs=("id", "count"),
            j_min=("journee", "min"),
            j_max=("journee", "max"),
        )
        print(f"\n   {len(seasons)} saisons détectées :")
        print(seasons.head(15).to_string())
        avg_n = seasons.n_matchs.mean()
        avg_dur_h = ((seasons.end - seasons.start).dt.total_seconds() / 3600).mean()
        print(f"\n   Saison moyenne : {avg_n:.0f} matchs, {avg_dur_h:.1f} heures")

    # ============ 3. Stabilité des forces d'équipes ENTRE saisons ============
    print()
    print("=" * 90)
    print("3️⃣  Les forces d'équipes changent-elles ENTRE saisons ?")
    print("=" * 90)
    if "season_id" in df_sorted.columns:
        df_with_score = df_sorted[df_sorted.score_a.notna()].copy()
        if len(df_with_score) > 100:
            # Calculer win rate home par équipe par saison
            df_with_score["won_home"] = (df_with_score.ft_o == "1").astype(int)
            pivot = df_with_score.groupby(["team_a", "season_id"]).agg(
                n=("id", "count"),
                wr=("won_home", "mean"),
            ).reset_index()
            pivot = pivot[pivot.n >= 5]

            # Variance intra-équipe inter-saison
            team_var = pivot.groupby("team_a").wr.agg(["mean", "std", "count"])
            team_var = team_var[team_var["count"] >= 2].sort_values("std", ascending=False)

            print(f"\n   Équipes avec PLUS GRANDE variation win rate home entre saisons :")
            print(f"   (std élevée = perf change beaucoup d'une saison à l'autre)")
            print(f"\n   {'Équipe':<22} {'Moy WR':<10} {'Écart-type':<12} {'Nb saisons'}")
            print("   " + "-" * 60)
            for team, row in team_var.head(15).iterrows():
                print(f"   {team:<22} {row['mean']*100:>5.1f}%    ±{row['std']*100:.1f}pp        {int(row['count'])}")

            avg_std = team_var["std"].mean()
            print(f"\n   📊 Écart-type moyen WR home inter-saison : ±{avg_std*100:.1f}pp")
            if avg_std > 0.08:
                print(f"   ⭐ FORTE variation → saisonnalité TRÈS importante")
            elif avg_std > 0.04:
                print(f"   ⚠️  Variation modérée → saison à prendre en compte")
            else:
                print(f"   🟢 Faible variation → forces stables")

    # ============ 4. Performance saison récente vs globale ============
    print()
    print("=" * 90)
    print("4️⃣  Saison la plus récente : nouvelles paires fortes apparues ?")
    print("=" * 90)
    if "season_id" in df_sorted.columns and len(seasons) > 0:
        last_season_id = df_sorted.season_id.max()
        last_season = df_sorted[df_sorted.season_id == last_season_id].copy()
        last_season_done = last_season[last_season.score_a.notna()]
        print(f"\n   Dernière saison : ID={last_season_id}, n={len(last_season)} matchs (dont {len(last_season_done)} finis)")
        if len(last_season_done) >= 10:
            # Top paires home (n>=3 dans saison)
            last_season_done["won_home"] = (last_season_done.ft_o == "1").astype(int)
            pair_stats = last_season_done.groupby(["team_a", "team_b"]).agg(
                n=("id", "count"),
                wr=("won_home", "mean"),
            ).reset_index()
            top_recent = pair_stats[(pair_stats.n >= 2) & (pair_stats.wr >= 0.75)].sort_values("wr", ascending=False)
            print(f"\n   Top paires home dernière saison (n≥2, wr≥75%) :")
            for _, r in top_recent.head(15).iterrows():
                print(f"     {r.team_a:<20} vs {r.team_b:<20}  {int(r.n)} matchs  WR {r.wr*100:.0f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
