"""Deeper analysis: bookmaker margin, odds calibration, market-by-market hit rate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sqlalchemy import create_engine

from scraper.config import load_settings


def main() -> int:
    settings = load_settings()
    engine = create_engine(settings.db_url)

    df = pd.read_sql(
        """
        SELECT e.id AS event_id, e.team_a, e.team_b, e.round_info,
               o.odds_home, o.odds_draw, o.odds_away, o.extra_markets,
               r.score_a, r.score_b
        FROM events e
        JOIN odds_snapshots o ON o.event_id = e.id
        JOIN results r ON r.event_id = e.id
        """,
        engine,
    )

    if df.empty:
        print("no joined matches yet")
        return 0

    df["total_goals"] = df["score_a"] + df["score_b"]
    df["btts"] = (df["score_a"] > 0) & (df["score_b"] > 0)
    df["outcome"] = df.apply(
        lambda r: "1" if r.score_a > r.score_b else ("X" if r.score_a == r.score_b else "2"),
        axis=1,
    )
    df["implied_h"] = 1 / df["odds_home"]
    df["implied_d"] = 1 / df["odds_draw"]
    df["implied_a"] = 1 / df["odds_away"]
    df["overround"] = df["implied_h"] + df["implied_d"] + df["implied_a"]
    # vig-free probabilities
    df["p_h"] = df["implied_h"] / df["overround"]
    df["p_d"] = df["implied_d"] / df["overround"]
    df["p_a"] = df["implied_a"] / df["overround"]

    n = len(df)
    print(f"=== sample: {n} matchs joints (cotes + score) ===\n")

    # 1. Bookmaker margin
    print("--- Marge bookmaker (1X2) ---")
    print(f"  overround moyen : {df['overround'].mean():.4f}  (1.0 = marché parfait)")
    print(f"  marge implicite : {(df['overround'].mean() - 1) * 100:.2f}%")
    print()

    # 2. Calibration : do quoted probabilities match observed frequencies?
    print("--- Calibration des cotes 1X2 (proba implicite vs réalisée) ---")
    for col_p, col_o, label in [("p_h", "outcome", "1 (home)"),
                                  ("p_d", "outcome", "X (draw)"),
                                  ("p_a", "outcome", "2 (away)")]:
        target = label[0]
        mean_p = df[col_p].mean()
        observed = (df[col_o] == target).mean()
        gap = (observed - mean_p) * 100
        print(f"  {label:<10} proba moy={mean_p:.3f}  observée={observed:.3f}  écart={gap:+.1f} pp")
    print()

    # 3. Calibration by odds bucket for home win
    print("--- Calibration par tranche de cote domicile (parier sur 1) ---")
    df["bucket"] = pd.cut(df["odds_home"], bins=[0, 1.5, 2.0, 2.5, 3.5, 100],
                          labels=["<1.5", "1.5-2.0", "2.0-2.5", "2.5-3.5", ">3.5"])
    by_bucket = df.groupby("bucket", observed=True).agg(
        n=("event_id", "count"),
        win_rate=("outcome", lambda s: (s == "1").mean()),
        avg_odds=("odds_home", "mean"),
    )
    by_bucket["roi_unit_stake"] = (by_bucket["win_rate"] * by_bucket["avg_odds"]) - 1
    print(by_bucket.round(3).to_string())
    print()

    # parse extra_markets once per row
    def parse_markets(em):
        if not em:
            return {}
        try:
            return em if isinstance(em, dict) else json.loads(em)
        except Exception:
            return {}
    df["em"] = df["extra_markets"].apply(parse_markets)

    # 4a. Over/Under via "+/-"
    print("--- Over/Under (clé +/-) ---")
    ou_rows = []
    for _, row in df.iterrows():
        m = row["em"].get("+/-")
        if not isinstance(m, dict):
            continue
        for line, odds in m.items():
            s = str(line).strip()
            try:
                if s.startswith(">"):
                    th = float(s.lstrip("> ").strip().replace(",", "."))
                    hit = 1 if row["total_goals"] > th else 0
                    ou_rows.append({"line": f"Over {th}", "odds": odds, "hit": hit})
                elif s.startswith("<"):
                    th = float(s.lstrip("< ").strip().replace(",", "."))
                    hit = 1 if row["total_goals"] < th else 0
                    ou_rows.append({"line": f"Under {th}", "odds": odds, "hit": hit})
            except ValueError:
                continue
    if ou_rows:
        ou = pd.DataFrame(ou_rows)
        agg = ou.groupby("line").agg(n=("hit", "count"),
                                       hit_rate=("hit", "mean"),
                                       avg_odds=("odds", "mean"))
        agg["implied_p"] = 1 / agg["avg_odds"]
        agg["edge_pp"] = (agg["hit_rate"] - agg["implied_p"]) * 100
        agg["roi"] = (agg["hit_rate"] * agg["avg_odds"]) - 1
        print(agg.round(3).to_string())
    else:
        print("  aucun Over/Under exploitable")
    print()

    # 4b. Total de buts EXACT (multi-goals)
    print("--- Nombre exact de buts (clé 'Total de buts') ---")
    tg_rows = []
    for _, row in df.iterrows():
        m = row["em"].get("Total de buts")
        if not isinstance(m, dict):
            continue
        for line, odds in m.items():
            try:
                n_goals = int(str(line).strip().rstrip("+"))
                # certains marchés cappent à "6+" ; on traite "6" comme >= 6
                if str(line).strip().endswith("+"):
                    hit = 1 if row["total_goals"] >= n_goals else 0
                else:
                    hit = 1 if row["total_goals"] == n_goals else 0
                tg_rows.append({"goals_line": str(line), "odds": odds, "hit": hit})
            except ValueError:
                continue
    if tg_rows:
        tg = pd.DataFrame(tg_rows)
        agg = tg.groupby("goals_line").agg(n=("hit", "count"),
                                             hit_rate=("hit", "mean"),
                                             avg_odds=("odds", "mean"))
        agg["implied_p"] = 1 / agg["avg_odds"]
        agg["edge_pp"] = (agg["hit_rate"] - agg["implied_p"]) * 100
        agg["roi"] = (agg["hit_rate"] * agg["avg_odds"]) - 1
        print(agg.sort_index().round(3).to_string())
    else:
        print("  aucun marché 'Total de buts' trouvé")
    print()

    # 5. BTTS (G/NG)
    print("--- Both Teams To Score (clé G/NG) ---")
    btts_rows = []
    for _, row in df.iterrows():
        m = row["em"].get("G/NG")
        if not isinstance(m, dict):
            continue
        for outcome, odds in m.items():
            hit = (
                int(row["btts"]) if outcome.lower().startswith("oui")
                else (1 - int(row["btts"])) if outcome.lower().startswith("non")
                else None
            )
            if hit is None:
                continue
            btts_rows.append({"outcome": outcome, "odds": odds, "hit": hit})
    if btts_rows:
        btts = pd.DataFrame(btts_rows)
        agg = btts.groupby("outcome").agg(n=("hit", "count"),
                                            hit_rate=("hit", "mean"),
                                            avg_odds=("odds", "mean"))
        agg["implied_p"] = 1 / agg["avg_odds"]
        agg["edge_pp"] = (agg["hit_rate"] - agg["implied_p"]) * 100
        agg["roi"] = (agg["hit_rate"] * agg["avg_odds"]) - 1
        print(agg.round(3).to_string())
    else:
        print("  aucun marché G/NG trouvé")
    print()

    # 6. Distribution des écarts de cote (favori marqué vs égalité)
    print("--- Cas où le score reflète/contredit la cote ---")
    df["fav_won"] = (
        ((df["odds_home"] < df["odds_away"]) & (df["outcome"] == "1")) |
        ((df["odds_home"] > df["odds_away"]) & (df["outcome"] == "2"))
    )
    df["draw"] = df["outcome"] == "X"
    df["upset"] = (~df["fav_won"]) & (~df["draw"])
    print(f"  favori gagne : {df['fav_won'].mean()*100:.1f}%  ({df['fav_won'].sum()}/{n})")
    print(f"  nul          : {df['draw'].mean()*100:.1f}%  ({df['draw'].sum()}/{n})")
    print(f"  upset        : {df['upset'].mean()*100:.1f}%  ({df['upset'].sum()}/{n})")
    print()

    # 7. ROI strategy: bet 1 unit on the favorite every match
    print("--- ROI fictif de stratégies basiques (mise 1 par match) ---")
    df["fav_odds"] = df[["odds_home", "odds_away"]].min(axis=1)
    df["fav_side"] = df.apply(
        lambda r: "1" if r.odds_home < r.odds_away else "2", axis=1
    )
    df["fav_payoff"] = df.apply(
        lambda r: r.fav_odds - 1 if r.outcome == r.fav_side else -1, axis=1
    )
    df["dog_odds"] = df[["odds_home", "odds_away"]].max(axis=1)
    df["dog_side"] = df.apply(
        lambda r: "1" if r.odds_home > r.odds_away else "2", axis=1
    )
    df["dog_payoff"] = df.apply(
        lambda r: r.dog_odds - 1 if r.outcome == r.dog_side else -1, axis=1
    )
    df["draw_payoff"] = df.apply(
        lambda r: r.odds_draw - 1 if r.outcome == "X" else -1, axis=1
    )
    for label, col in [("toujours favori", "fav_payoff"),
                        ("toujours sous-coté", "dog_payoff"),
                        ("toujours nul", "draw_payoff")]:
        roi = df[col].sum() / n * 100
        wins = (df[col] > 0).sum()
        print(f"  {label:<22} ROI={roi:+.1f}%  ({wins}/{n} gains)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
